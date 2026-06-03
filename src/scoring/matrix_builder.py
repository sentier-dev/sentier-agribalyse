"""``TechnosphereBuilder`` / ``BiosphereBuilder`` — pure DataFrame → CSR.

The builders take an :class:`~scoring.exchange_frame.ExchangeFrame` and
emit a sparse matrix plus the ``id → row/col`` dictionaries needed to
map a demand vector or a characterization vector back to integer
positions. No bw2data, no SQLite, no allocation.

Sign convention (matches Brightway / ``bw_processing``):

* Production edges contribute on the diagonal, positive: a 1-unit
  product output is encoded as ``A[product_row, activity_col] = +amount``.
* Technosphere consumption rows contribute negatively: a 1-unit input
  draw becomes ``A[input_row, activity_col] = -amount``.
* Substitution edges contribute *positively* — same sign as production.
  ``bw_simapro_csv`` (and AGB upstream) emit substitution amounts as
  positive numbers meaning "this activity displaces N units of the
  named product elsewhere"; the canonical bw_processing rule sets
  ``flip=False`` for substitution rows, so the matrix entry stays
  positive and the linear solve subtracts the avoided burden. Treating
  substitution as ``-amount`` (the original L4 implementation) flipped
  the credit into a debit and amplified upstream impacts — that was
  the dominant cause of the post-refactor score over-estimation.

Squareness is a precondition: the builder asserts
``len(product_id_to_row) == len(activity_id_to_col)`` before returning,
because ``bc.LCA`` would otherwise fall through to a least-squares
pseudo-solution whose impacts are physically meaningless. This is the
same invariant ``LciaScorer._assert_square_technosphere`` enforces at
the scoring boundary — but here we catch it earlier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse as sp

from scoring.exchange_frame import ExchangeFrame


@dataclass(frozen=True)
class BuiltMatrix:
    """Result of one build pass — sparse CSR + id → index dicts."""

    matrix: sp.csr_matrix
    row_id_to_idx: dict[int, int]
    col_id_to_idx: dict[int, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape


@dataclass(frozen=True)
class TechnosphereBuilder:
    """Builds the technosphere matrix A from production + consumption
    edges. ``A`` is square: rows = products, cols = activities, and for
    a square supply chain there is exactly one production edge per
    activity (``A[i, i] > 0``).

    Multifunctional activities (more production rows than activity
    cols) MUST have been allocated upstream — the
    :class:`~scoring.allocator.Allocator` is responsible for that. If
    we receive an un-allocated frame, we raise rather than producing a
    rank-deficient matrix."""

    def build(self, frame: ExchangeFrame) -> BuiltMatrix:
        tech = frame.technosphere
        if tech.empty:
            raise ValueError(
                "TechnosphereBuilder cannot build from an empty technosphere "
                "slice — the ExchangeFrame contains no production/consumption edges."
            )

        # Pair each product with its (canonical, single-after-dedup) producer.
        # Sorting *that pairing* by ``input_id`` makes the production amounts
        # land on the diagonal: row i and col i refer to the same
        # (product, activity) pair. Building ``row_id_to_idx`` and
        # ``col_id_to_idx`` independently from sorted ids drifted them apart
        # and produced a structurally singular matrix (every off-diagonal
        # production entry shifted to a row whose actual product had no
        # producer in the same column).
        #
        # ``amount > 0`` guard: ``DanglingEdgePruner`` already drops
        # zero-production rows, but a stub ``type='product'`` entry can
        # acquire a synthetic zero-amount production row through
        # link-time rewrites that the pre-matrix pipeline doesn't see.
        # Including those would inject zero-diagonal rows that pypardiso
        # pivots through and turns into 1e+8 supply vectors. Filtering
        # at the builder is the last line of defence — it mirrors
        # ``MatrixPurger._orphan_products`` running after ``Database.process()``.
        production = tech.loc[
            tech["edge_type"].isin(("production", "generic production")) & (tech["amount"] > 0)
        ]
        production_pairs = (
            production[["input_id", "output_id"]]
            .drop_duplicates(subset="input_id")
            .sort_values("input_id", kind="stable")
            .reset_index(drop=True)
        )
        if len(production_pairs) != len(frame.activities):
            raise ValueError(
                f"Technosphere is non-square: {len(production_pairs)} products vs "
                f"{len(frame.activities)} activities. Run the allocator before building."
            )

        row_id_to_idx = {int(pid): i for i, pid in enumerate(production_pairs["input_id"])}
        col_id_to_idx = {int(aid): i for i, aid in enumerate(production_pairs["output_id"])}

        rows = tech["input_id"].map(row_id_to_idx).to_numpy()
        cols = tech["output_id"].map(col_id_to_idx).to_numpy()

        # Drop edges that survived the upstream pruner but reference
        # products/activities the ``amount > 0`` filter just removed.
        # These are typically consumption edges pointing at stub
        # ``type='product'`` rows that no real activity produces.
        valid = ~(pd.isna(rows) | pd.isna(cols))
        if not valid.all():
            tech = tech.loc[valid]
            rows = rows[valid.to_numpy()]
            cols = cols[valid.to_numpy()]

        # Sign convention: production *and* substitution are positive
        # contributions on the activity column (the activity yields a
        # product output / displaces an external product), every other
        # technosphere edge is a negative consumption coefficient. This
        # mirrors ``bw_processing``'s ``flip`` flag, where production
        # and substitution rows have ``flip=False``.
        positive_signs = ("production", "generic production", "substitution")
        is_positive = tech["edge_type"].isin(positive_signs).to_numpy()
        sign = np.where(is_positive, 1.0, -1.0)
        data = tech["amount"].to_numpy(dtype="float64") * sign

        n = len(production_pairs)
        coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        return BuiltMatrix(
            matrix=coo.tocsr(),
            row_id_to_idx=row_id_to_idx,
            col_id_to_idx=col_id_to_idx,
        )


@dataclass(frozen=True)
class BiosphereBuilder:
    """Builds the biosphere matrix B. Rows = biosphere flows, cols =
    activities. ``B[i, j]`` is the amount of biosphere flow ``i``
    emitted (or consumed, if negative) per unit output of activity ``j``.

    No squareness constraint here — B is rectangular by design (more
    flows than activities, typically)."""

    def build(
        self,
        frame: ExchangeFrame,
        col_id_to_idx: dict[int, int] | None = None,
    ) -> BuiltMatrix:
        """Build B with columns indexed consistently with the technosphere.

        ``col_id_to_idx`` should be the technosphere's column map so the
        product ``B @ supply`` is meaningful: ``supply`` comes out of
        ``A`` keyed by that column ordering. When omitted (e.g. unit
        tests with no technosphere), we fall back to sorted-activities
        which is fine in isolation but **not** combinable with a
        separately-built A.
        """
        bio = frame.biosphere
        flow_ids = frame.biosphere_flows
        if col_id_to_idx is None:
            activity_ids = frame.activities
            col_id_to_idx = {int(aid): j for j, aid in enumerate(activity_ids)}
        n_cols = len(col_id_to_idx)
        row_id_to_idx = {fid: i for i, fid in enumerate(flow_ids)}

        if bio.empty:
            return BuiltMatrix(
                matrix=sp.csr_matrix((len(flow_ids), n_cols)),
                row_id_to_idx=row_id_to_idx,
                col_id_to_idx=col_id_to_idx,
            )

        rows = bio["input_id"].map(row_id_to_idx).to_numpy()
        cols = bio["output_id"].map(col_id_to_idx).to_numpy()
        data = bio["amount"].to_numpy(dtype="float64")

        # Biosphere edges may reference activities that the dedup /
        # dangling-edge pruner removed from the technosphere. Drop
        # those rows here too, otherwise ``coo_matrix`` raises on NaN
        # column indices.
        mask = ~(pd.isna(rows) | pd.isna(cols))
        if not mask.all():
            rows = rows[mask]
            cols = cols[mask]
            data = data[mask]

        coo = sp.coo_matrix(
            (data, (rows.astype("int64"), cols.astype("int64"))),
            shape=(len(flow_ids), n_cols),
        )
        return BuiltMatrix(
            matrix=coo.tocsr(),
            row_id_to_idx=row_id_to_idx,
            col_id_to_idx=col_id_to_idx,
        )


@dataclass(frozen=True)
class CharacterizationBuilder:
    """Builds a per-method characterization vector ``Q``.

    For one method, ``Q`` is a 1×N row vector with one CF per
    biosphere flow (and zeros for unmatched flows). The LCIA score is
    ``Q @ B @ supply``.

    Input ``cf_df`` columns: ``flow_id (int)``, ``cf (float)``."""

    def build(self, cf_df: pd.DataFrame, biosphere: BuiltMatrix) -> sp.csr_matrix:
        n_flows = biosphere.matrix.shape[0]
        if cf_df.empty or n_flows == 0:
            return sp.csr_matrix((1, n_flows))

        flow_to_row = biosphere.row_id_to_idx
        rows = np.zeros(len(cf_df), dtype="int64")
        cols = cf_df["flow_id"].map(flow_to_row).to_numpy()
        data = cf_df["cf"].to_numpy(dtype="float64")

        # Drop flows we don't have a row for (e.g. an EF method that
        # characterises elements not present in this DB).
        mask = ~pd.isna(cols)
        if not mask.all():
            rows = rows[mask]
            cols = cols[mask].astype("int64")
            data = data[mask]
        else:
            cols = cols.astype("int64")

        return sp.coo_matrix((data, (rows, cols)), shape=(1, n_flows)).tocsr()
