"""``ScoreDecomposer`` — explain why a product scores what it scores.

The native scorer answers ``what is the score?``; this class answers
``where does the score come from?``. Given a product key and a method,
it factors the canonical LCIA expression

.. math::

    \\text{score} = Q B A^{-1} d

into three labelled tables that all sum back to the same scalar:

* :attr:`Decomposition.flow_contributions` — one row per characterised
  biosphere flow row in :math:`B`. Carries the inventory amount
  (``B @ supply`` for that flow), the CF, and ``cf * inventory`` as the
  per-flow contribution.
* :attr:`Decomposition.activity_contributions` — one row per non-zero
  ``supply[j]`` column. The contribution is ``(Q @ B)[j] * supply[j]``,
  i.e. how much the score moves if this single activity disappears
  from the supply chain.
* :attr:`Decomposition.edge_contributions` — one row per
  ``(flow, activity)`` pair where ``B[i, j] != 0`` and ``Q[i] != 0`` and
  ``supply[j] != 0``. Contribution is ``Q[i] * B[i, j] * supply[j]``.
  Use this to find ``which kg of CO2 from which activity is dominating
  the score?``

By design the decomposer is **read-only** — it does not modify the
package or the catalogs. It is built once per ``ScoringPackage`` and
``ProductCatalog`` and reused across products and methods, with the
LU factorisation cached on the underlying ``NativeLciaScorer``.

The biosphere catalog is required only to label rows by name and
sub-compartment; passing an empty DataFrame still produces correct
numbers but the ``flow_name`` / ``sub_compartment`` columns will be
filled with the bare flow id.

Construction is intentionally not lazy — pass instantiated
``ScoringPackage``, ``ProductCatalog``, and biosphere catalog
DataFrame so all I/O happens at the boundary and the decomposer
itself stays a pure (matrix → frame) function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp

from core.logging import Logging
from scoring.product_catalog import ProductCatalog
from scoring.scoring_package import ScoringPackage

try:
    from pypardiso import PyPardisoSolver as _PyPardisoSolver
except ImportError:
    _PyPardisoSolver = None

from scipy.sparse.linalg import factorized as _scipy_factorized


@dataclass(frozen=True)
class Decomposition:
    """Result of one ``decompose(product, method)`` call.

    :attr:`score` matches :class:`~scoring.native_scorer.NativeLciaScorer`:
    it is ``(Q @ inventory) + (correction @ supply)`` so regionalised
    methods (water-use, ecotoxicity) reconcile with the backtest CSV.
    :attr:`global_score` exposes the un-corrected ``Q @ inventory`` part
    and :attr:`correction_score` the per-activity adjustment, so callers
    can see what the regional correction shifted the number by.

    The three contribution frames sum within float tolerance to
    :attr:`global_score` for the flow / edge views (which only know
    about the global ``Q`` row) and to :attr:`score` for the activity
    view (which folds the correction into its ``contribution`` column).
    """

    product_key: tuple[str, str]
    method: tuple[str, ...]
    score: float
    flow_contributions: pd.DataFrame
    activity_contributions: pd.DataFrame
    edge_contributions: pd.DataFrame
    global_score: float = 0.0
    correction_score: float = 0.0


@dataclass(frozen=True)
class ScoreDecomposer:
    """Decompose ``Q @ B @ A^{-1} @ d`` into per-flow / per-activity / per-edge contributions.

    ``package`` carries the matrices and id maps. ``product_catalog``
    resolves a ``(database, code)`` request into the integer
    ``product_id`` that lives in the technosphere row map. The
    biosphere and ecoinvent catalogs supply human-readable names for
    the resulting tables; without them the decomposer still works but
    the labels collapse to bare integer ids.
    """

    package: ScoringPackage
    product_catalog: ProductCatalog
    biosphere_catalog: pd.DataFrame
    ecoinvent_catalog: pd.DataFrame | None = None
    use_pardiso: bool = False
    _solver_cache: list[Any] = field(default_factory=list, init=False, repr=False)
    _flow_label_cache: list[dict[int, tuple[str, str, str]]] = field(
        default_factory=list, init=False, repr=False
    )
    _activity_label_cache: list[dict[int, tuple[str, str, str]]] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------

    def decompose(
        self,
        product_key: tuple[str, str],
        method: tuple[str, ...],
        *,
        top_n: int | None = None,
        flow_only: bool = False,
    ) -> Decomposition:
        """Decompose the score for *product_key* under *method*.

        ``top_n`` truncates each contribution table to the *N* rows
        with largest ``|contribution|``. ``None`` (default) keeps all
        non-zero contributions.

        ``flow_only=True`` skips the activity and edge contribution
        frames (they remain as empty DataFrames in the returned
        :class:`Decomposition`). Use for batch flow-only callers like
        :class:`reporting.flow_decomposition.FlowDecompositionEmitter`
        where the edge view's per-flow × per-activity iteration is the
        dominant cost.
        """
        if method not in self.package.methods:
            raise ValueError(
                f"method={method!r} not registered in this ScoringPackage. "
                f"Available: {list(self.package.methods)}"
            )
        results = self.decompose_methods(product_key, [method], top_n=top_n, flow_only=flow_only)
        return results[method]

    def decompose_methods(
        self,
        product_key: tuple[str, str],
        methods: list[tuple[str, ...]],
        *,
        top_n: int | None = None,
        flow_only: bool = False,
    ) -> dict[tuple[str, ...], Decomposition]:
        """Decompose *product_key* under every method in *methods*.

        Solves ``A^{-1} d`` once for the product and projects against
        each method's ``Q`` row — same numeric result as 19 calls to
        :meth:`decompose` but ~10× faster on the agribalyse matrix
        where the linear solve dominates per-method projection.

        Returns a ``{method_tuple → Decomposition}`` mapping. Methods
        absent from ``package.methods`` are silently dropped (callers
        like ``FlowDecompositionEmitter`` already log them upstream).
        """
        product_id = self.product_catalog.product_id_for(product_key[0], product_key[1])
        if product_id is None:
            raise ValueError(
                f"product_key={product_key!r} not in product catalog. "
                f"Check that linking has run for this database."
            )

        a_csc = self.package.technosphere.matrix.tocsc()
        b: sp.csr_matrix = self.package.biosphere.matrix
        n = a_csc.shape[0]

        row_map = self.package.technosphere.row_id_to_idx
        if product_id not in row_map:
            raise ValueError(
                f"product_id for {product_key!r} ({product_id}) is absent from the "
                f"technosphere row map. The product is in the catalog but the matrix "
                f"has no production edge for it — typical of unlinked stub products."
            )

        demand = np.zeros(n, dtype="float64")
        demand[row_map[product_id]] = 1.0
        supply = self._solve(a_csc, demand)
        inventory = b @ supply

        out: dict[tuple[str, ...], Decomposition] = {}
        for method in methods:
            q = self.package.methods.get(method)
            if q is None:
                continue
            global_score = float((q @ inventory)[0])
            correction = self.package.corrections.get(method)
            correction_score = float((correction @ supply)[0]) if correction is not None else 0.0
            score = global_score + correction_score

            flow_contribs = self._flow_contributions(q, inventory, global_score, top_n=top_n)
            if flow_only:
                activity_contribs = self._empty_activity_frame()
                edge_contribs = self._empty_edge_frame()
            else:
                activity_contribs = self._activity_contributions(
                    q, b, supply, score, correction=correction, top_n=top_n
                )
                edge_contribs = self._edge_contributions(q, b, supply, global_score, top_n=top_n)

            out[method] = Decomposition(
                product_key=product_key,
                method=method,
                score=score,
                flow_contributions=flow_contribs,
                activity_contributions=activity_contribs,
                edge_contributions=edge_contribs,
                global_score=global_score,
                correction_score=correction_score,
            )
        return out

    # ------------------------------------------------------------------

    def inventory(
        self,
        product_key: tuple[str, str],
        *,
        method: tuple[str, ...] | None = None,
        top_n: int | None = None,
        nonzero_only: bool = True,
    ) -> pd.DataFrame:
        """Return the full ``B @ supply`` inventory vector as a DataFrame.

        One row per biosphere flow with ``inventory_amount != 0``.
        When *method* is given the CF column is filled in from that
        method's characterisation row (zero where uncharacterised), so
        the caller can spot flows that are emitted but not scored —
        the canonical signature of "right amount, wrong sub-compartment
        in bio3 catalog so CF lookup misses".
        """
        product_id = self.product_catalog.product_id_for(product_key[0], product_key[1])
        if product_id is None:
            raise ValueError(f"product_key={product_key!r} not in product catalog.")

        a_csc = self.package.technosphere.matrix.tocsc()
        b: sp.csr_matrix = self.package.biosphere.matrix
        n = a_csc.shape[0]
        row_map = self.package.technosphere.row_id_to_idx
        if product_id not in row_map:
            raise ValueError(
                f"product_id for {product_key!r} ({product_id}) absent from technosphere row map."
            )
        demand = np.zeros(n, dtype="float64")
        demand[row_map[product_id]] = 1.0
        supply = self._solve(a_csc, demand)
        inventory = b @ supply

        flow_id_to_idx = self.package.biosphere.row_id_to_idx
        idx_to_flow_id = {idx: fid for fid, idx in flow_id_to_idx.items()}
        n_flows = inventory.shape[0]

        nz = np.nonzero(inventory)[0] if nonzero_only else np.arange(n_flows)
        if nz.size == 0:
            return pd.DataFrame(
                columns=[
                    "flow_id",
                    "flow_name",
                    "sub_compartment",
                    "compartment",
                    "inventory_amount",
                    "cf",
                    "contribution",
                ]
            )
        flow_ids = [idx_to_flow_id.get(int(i), -1) for i in nz]
        labels = self._label_flows(flow_ids)

        if method is None:
            cfs = np.zeros(nz.size)
        else:
            if method not in self.package.methods:
                raise ValueError(f"method={method!r} not registered.")
            q_dense = self.package.methods[method].toarray()[0]
            cfs = q_dense[nz]
        contributions = inventory[nz] * cfs

        df = pd.DataFrame(
            {
                "flow_id": flow_ids,
                "flow_name": labels["name"],
                "sub_compartment": labels["sub_compartment"],
                "compartment": labels["compartment"],
                "inventory_amount": inventory[nz],
                "cf": cfs,
                "contribution": contributions,
            }
        )
        df = df.reindex(df["inventory_amount"].abs().sort_values(ascending=False).index)
        if top_n is not None:
            df = df.head(top_n)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------

    def _solve(self, a_csc: sp.csc_matrix, demand: np.ndarray) -> np.ndarray:
        if not self._solver_cache:
            if self.use_pardiso and _PyPardisoSolver is not None:
                solver = _PyPardisoSolver()
                solver.factorize(a_csc)
                self._solver_cache.append(lambda b, _s=solver, _a=a_csc: _s.solve(_a, b))
            else:
                self._solver_cache.append(_scipy_factorized(a_csc))
        return self._solver_cache[0](demand)

    # ------------------------------------------------------------------

    def _flow_contributions(
        self,
        q: sp.csr_matrix,
        inventory: np.ndarray,
        score: float,
        *,
        top_n: int | None,
    ) -> pd.DataFrame:
        """Per-flow contribution: ``Q[i] * inventory[i]`` for each
        characterised flow."""
        # Q is a 1-by-n_flows sparse row; pull non-zero CFs and zip with inventory.
        q_dense = q.toarray()[0]  # cheap: 1 row
        nz = np.nonzero(q_dense)[0]
        if nz.size == 0:
            return self._empty_flow_frame()

        cfs = q_dense[nz]
        inv = inventory[nz]
        contributions = cfs * inv

        # Drop ones with zero contribution (uncharacterised flows are
        # already excluded by the ``nz`` mask, but a flow can be
        # characterised yet not emitted, giving 0).
        keep = contributions != 0.0
        if not keep.any():
            return self._empty_flow_frame()

        nz, cfs, inv, contributions = nz[keep], cfs[keep], inv[keep], contributions[keep]

        # Resolve flow ids → labels.
        flow_id_to_idx = self.package.biosphere.row_id_to_idx
        idx_to_flow_id = {idx: fid for fid, idx in flow_id_to_idx.items()}
        flow_ids = [idx_to_flow_id.get(int(i), -1) for i in nz]
        labels = self._label_flows(flow_ids)

        df = pd.DataFrame(
            {
                "flow_id": flow_ids,
                "flow_name": labels["name"],
                "sub_compartment": labels["sub_compartment"],
                "compartment": labels["compartment"],
                "inventory_amount": inv,
                "cf": cfs,
                "contribution": contributions,
                "share": contributions / score if score else 0.0,
            }
        )
        df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
        if top_n is not None:
            df = df.head(top_n)
        return df.reset_index(drop=True)

    def _activity_contributions(
        self,
        q: sp.csr_matrix,
        b: sp.csr_matrix,
        supply: np.ndarray,
        score: float,
        *,
        correction: sp.csr_matrix | None = None,
        top_n: int | None,
    ) -> pd.DataFrame:
        """Per-activity contribution: ``((Q @ B) + correction)[j] * supply[j]``.

        The regional correction (1, n_activities) carries the per-
        activity adjustment ``Σ_i (regional_cf[i, loc(j)] - global_cf[i]) * B[i,j]``;
        folding it into the activity row keeps the activity view consistent
        with the corrected ``score``.
        """
        qb = (q @ b).toarray()[0]  # shape (n_activities,)
        if correction is not None and correction.nnz:
            correction_dense = correction.toarray()[0]
        else:
            correction_dense = np.zeros_like(qb)
        effective = qb + correction_dense
        contributions = effective * supply
        nz = np.nonzero(contributions)[0]
        if nz.size == 0:
            return self._empty_activity_frame()

        col_id_to_idx = self.package.technosphere.col_id_to_idx
        idx_to_col_id = {idx: aid for aid, idx in col_id_to_idx.items()}
        activity_ids = [idx_to_col_id.get(int(j), -1) for j in nz]
        labels = self._label_activities(activity_ids)

        df = pd.DataFrame(
            {
                "activity_id": activity_ids,
                "activity_database": labels["database"],
                "activity_code": labels["code"],
                "activity_name": labels["name"],
                "supply": supply[nz],
                "qb": qb[nz],
                "regional_delta": correction_dense[nz],
                "contribution": contributions[nz],
                "share": contributions[nz] / score if score else 0.0,
            }
        )
        df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
        if top_n is not None:
            df = df.head(top_n)
        return df.reset_index(drop=True)

    def _edge_contributions(
        self,
        q: sp.csr_matrix,
        b: sp.csr_matrix,
        supply: np.ndarray,
        score: float,
        *,
        top_n: int | None,
    ) -> pd.DataFrame:
        """Per (flow, activity) edge contribution: ``Q[i] * B[i, j] * supply[j]``.

        Iterates only over biosphere rows that have a CF (typically a
        few hundred), then over their non-zero columns. With CSR row
        access this is O(nnz on characterised flows), which is far less
        than dense ``q.T * b * diag(supply)``.
        """
        q_dense = q.toarray()[0]
        nz_flow = np.nonzero(q_dense)[0]
        if nz_flow.size == 0 or supply.size == 0:
            return self._empty_edge_frame()

        # Restrict to biosphere rows that are characterised.
        b_csr = b.tocsr()
        flow_rows = []
        activity_cols = []
        edge_amounts = []
        cfs = []
        supplies = []

        flow_id_to_idx = self.package.biosphere.row_id_to_idx
        idx_to_flow_id = {idx: fid for fid, idx in flow_id_to_idx.items()}
        col_id_to_idx = self.package.technosphere.col_id_to_idx
        idx_to_col_id = {idx: aid for aid, idx in col_id_to_idx.items()}

        for i in nz_flow:
            cf = q_dense[i]
            row = b_csr.getrow(int(i))
            cols = row.indices
            data = row.data
            for col_idx, edge in zip(cols, data, strict=True):
                s = supply[int(col_idx)]
                if s == 0 or edge == 0:
                    continue
                flow_rows.append(idx_to_flow_id.get(int(i), -1))
                activity_cols.append(idx_to_col_id.get(int(col_idx), -1))
                edge_amounts.append(float(edge))
                cfs.append(float(cf))
                supplies.append(float(s))

        if not flow_rows:
            return self._empty_edge_frame()

        flow_labels = self._label_flows(flow_rows)
        activity_labels = self._label_activities(activity_cols)

        contributions = np.asarray(cfs) * np.asarray(edge_amounts) * np.asarray(supplies)
        df = pd.DataFrame(
            {
                "activity_id": activity_cols,
                "activity_database": activity_labels["database"],
                "activity_code": activity_labels["code"],
                "activity_name": activity_labels["name"],
                "flow_id": flow_rows,
                "flow_name": flow_labels["name"],
                "sub_compartment": flow_labels["sub_compartment"],
                "compartment": flow_labels["compartment"],
                "edge_amount": edge_amounts,
                "supply": supplies,
                "cf": cfs,
                "contribution": contributions,
                "share": contributions / score if score else 0.0,
            }
        )
        df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
        if top_n is not None:
            df = df.head(top_n)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Label resolution.

    def _label_flows(self, flow_ids: list[int]) -> dict[str, list]:
        """Resolve biosphere flow ids → (name, sub_compartment, compartment).

        The biosphere catalog is keyed by ``(database, code)`` whereas
        the matrix is keyed by integer flow ids. ``ExchangeFrameBuilder``
        assigns the same hash, so we re-derive ``flow_id_for(db, code)``
        for every catalog row, build a reverse map, and look up.

        The full ``flow_id → label`` map is cached on the decomposer
        after the first call — building it iterates the entire
        biosphere catalog (~4 k rows) and dominates batch-emitter cost.
        """
        from scoring.exchange_frame_builder import ExchangeFrameBuilder

        if self.biosphere_catalog.empty:
            return {
                "name": [str(i) for i in flow_ids],
                "sub_compartment": ["" for _ in flow_ids],
                "compartment": ["" for _ in flow_ids],
            }

        if not self._flow_label_cache:
            bc = self.biosphere_catalog
            flow_id_to_label: dict[int, tuple[str, str, str]] = {}
            for row in bc.itertuples(index=False):
                db = str(getattr(row, "database", ""))
                code = str(getattr(row, "code", ""))
                if not db or not code:
                    continue
                fid = ExchangeFrameBuilder.flow_id_for((db, code))
                cats = getattr(row, "categories", None)
                top, sub = self._split_compartment(cats)
                flow_id_to_label[fid] = (str(getattr(row, "name", "") or ""), top, sub)
            self._flow_label_cache.append(flow_id_to_label)
        flow_id_to_label = self._flow_label_cache[0]

        names, subs, tops = [], [], []
        for fid in flow_ids:
            label = flow_id_to_label.get(int(fid))
            if label is None:
                names.append(str(fid))
                subs.append("")
                tops.append("")
            else:
                names.append(label[0])
                tops.append(label[1])
                subs.append(label[2])
        return {"name": names, "sub_compartment": subs, "compartment": tops}

    def _label_activities(self, activity_ids: list[int]) -> dict[str, list]:
        """Resolve activity ids → (database, code, name).

        The technosphere column id is the *activity* id — the
        production edge's ``output_id`` in ``ExchangeFrame`` terms,
        equal to ``flow_id_for((activity_database, activity_code))``.
        ``ProductCatalog`` is keyed by ``(database, code)`` of the
        activity, so we re-derive the activity id from those columns
        and build the lookup.

        The merged ``activity_id → label`` map is cached on the
        decomposer so batch-emitter calls do not re-sort the full
        product catalog on every method.
        """
        from scoring.exchange_frame_builder import ExchangeFrameBuilder

        if not self._activity_label_cache:
            pc_df = self.product_catalog._df
            df = pc_df.copy()
            # Prefer 'process' / 'multifunctional' over bare 'product' so
            # the label carries the activity name rather than a stub.
            rank = {
                "process": 0,
                "multifunctional": 0,
                "processwithreferenceproduct": 0,
                "product": 1,
            }
            df["_rank"] = df["type"].map(rank).fillna(2).astype("int64")
            df = df.sort_values(["database", "code", "_rank"])
            first = df.drop_duplicates(subset=["database", "code"], keep="first")
            lookup: dict[int, tuple[str, str, str]] = {}
            for row in first.itertuples(index=False):
                db = str(row.database)
                code = str(row.code)
                aid = ExchangeFrameBuilder.flow_id_for((db, code))
                lookup[aid] = (db, code, str(row.name))

            # Optionally overlay ecoinvent activity names.
            if self.ecoinvent_catalog is not None and not self.ecoinvent_catalog.empty:
                for row in self.ecoinvent_catalog.itertuples(index=False):
                    db = str(getattr(row, "database", "") or "")
                    code = str(getattr(row, "code", "") or "")
                    if not db or not code:
                        continue
                    aid = ExchangeFrameBuilder.flow_id_for((db, code))
                    # Product-catalog hit wins over ecoinvent overlay.
                    lookup.setdefault(aid, (db, code, str(getattr(row, "name", "") or "")))
            self._activity_label_cache.append(lookup)
        lookup = self._activity_label_cache[0]

        dbs, codes, names = [], [], []
        for aid in activity_ids:
            label = lookup.get(int(aid))
            if label is None:
                dbs.append("")
                codes.append("")
                names.append(str(aid))
            else:
                dbs.append(label[0])
                codes.append(label[1])
                names.append(label[2])
        return {"database": dbs, "code": codes, "name": names}

    # ------------------------------------------------------------------

    @staticmethod
    def _split_compartment(cats: object) -> tuple[str, str]:
        """Return ``(top_compartment, sub_compartment)`` from a categories list."""
        if cats is None:
            return ("", "")
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if not isinstance(cats, (list, tuple)) or not cats:
            return ("", "")
        top = str(cats[0])
        sub = str(cats[1]) if len(cats) > 1 else ""
        return (top, sub)

    @staticmethod
    def _empty_flow_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "flow_id",
                "flow_name",
                "sub_compartment",
                "compartment",
                "inventory_amount",
                "cf",
                "contribution",
                "share",
            ]
        )

    @staticmethod
    def _empty_activity_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "activity_id",
                "activity_database",
                "activity_code",
                "activity_name",
                "supply",
                "qb",
                "regional_delta",
                "contribution",
                "share",
            ]
        )

    @staticmethod
    def _empty_edge_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "activity_id",
                "activity_database",
                "activity_code",
                "activity_name",
                "flow_id",
                "flow_name",
                "sub_compartment",
                "compartment",
                "edge_amount",
                "supply",
                "cf",
                "contribution",
                "share",
            ]
        )
