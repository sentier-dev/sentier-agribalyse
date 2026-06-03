"""``NativeLciaScorer`` — score directly from a ``ScoringPackage``.

Phase 5 of the SQLite refactor. ``LciaScorer`` (the legacy class) goes
through ``bw2calc.LCA``, which is fine but adds:

* an extra layer of indirection (``data_objs`` plumbing),
* an internal demand-vector dictionary keyed by activity id,
* a method-switch hook that re-binds the characterization on every
  method change.

The native scorer skips all three: with the matrices already loaded as
``scipy.sparse`` CSR, scoring is just ``Q @ B @ spsolve(A, demand)``.
Factorization is cached on the first call so subsequent products reuse
the LU. With pypardiso the factorization is shared via its singleton
solver; with scipy we cache an LU on this scorer instance.

Crucially, this scorer never touches ``bw2data``. With it in place,
the runtime image of the agent doesn't need bw2data, peewee, or the
SQLite ``databases.db`` file. ``bw2data`` is build-time only — used by
``SimaProImporter`` to parse the input CSV and by ``ExchangeFrameBuilder``
to produce the input frame, but neither runs at scoring time."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import factorized as _scipy_factorized

from core.logging import Logging
from scoring.scorer import ScoringResult
from scoring.scoring_package import ScoringPackage, ScoringPackageStore

try:
    from pypardiso import spsolve as _pypardiso_spsolve
except ImportError:
    _pypardiso_spsolve = None


@dataclass
class NativeLciaScorer:
    """Score products against methods using only ``scipy.sparse``.

    Build it with a fully-loaded ``ScoringPackage``; call ``score(...)``
    with a sequence of ``(process_key, product_id)`` tuples. The scorer
    caches the LU after the first solve so subsequent products are
    millisecond-scale."""

    package: ScoringPackage
    use_pardiso: bool = True
    _solver: Any = field(default=None, init=False, repr=False)

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def score(
        self,
        products: Sequence[tuple[tuple[str, str], int]],
        methods: Sequence[tuple[str, ...]],
    ) -> dict[tuple[str, str], ScoringResult]:
        """Score every product against every method.

        ``products`` is a sequence of ``(process_key, product_id)``: the
        key indexes the result dict (matching the legacy ``LciaScorer``
        contract); the id is the integer that lives in the technosphere
        matrix's row map.
        """
        if not methods:
            raise ValueError("methods must be non-empty")
        unknown = [m for m in methods if m not in self.package.methods]
        if unknown:
            raise ValueError(
                f"NativeLciaScorer: package does not carry CFs for {unknown}. "
                f"Available: {list(self.package.methods)}"
            )

        a = self.package.technosphere.matrix.tocsc()
        b = self.package.biosphere.matrix
        row_map = self.package.technosphere.row_id_to_idx
        n = a.shape[0]

        results: dict[tuple[str, str], ScoringResult] = {}
        for process_key, product_id in products:
            t0 = time.time()
            if product_id not in row_map:
                results[process_key] = ScoringResult(
                    process_key=process_key,
                    scores=None,
                    skip_reason="product id not in technosphere",
                    elapsed_s=time.time() - t0,
                )
                continue

            demand = np.zeros(n, dtype="float64")
            demand[row_map[product_id]] = 1.0
            supply = self._solve(a, demand)
            inventory = b @ supply

            scores: dict[tuple[str, ...], float] = {}
            for method in methods:
                q = self.package.methods[method]
                score = float((q @ inventory)[0])
                # Per-activity regional correction (currently water-use
                # only). The correction row is shape (1, n_activities)
                # and captures sum_i (regional_cf[i, location(j)] -
                # global_cf[i]) * B[i, j] for each j, so multiplying by
                # ``supply`` contracts to a scalar add — same algebraic
                # shape as ``q @ inventory`` from the scorer's POV.
                correction = self.package.corrections.get(method)
                if correction is not None:
                    score += float((correction @ supply)[0])
                scores[method] = score

            results[process_key] = ScoringResult(
                process_key=process_key,
                scores=scores,
                skip_reason=None,
                elapsed_s=time.time() - t0,
            )
        return results

    # ------------------------------------------------------------------

    def _solve(self, a_csc: sp.csc_matrix, demand: np.ndarray) -> np.ndarray:
        """Solve ``A @ supply = demand``, caching the LU.

        With pypardiso, the singleton solver caches its own factorization
        keyed by the matrix bytes — we just call ``spsolve`` and let
        Pardiso's internals do the right thing. With scipy, we factorize
        once on this instance and reuse for subsequent demands."""
        if self.use_pardiso and _pypardiso_spsolve is not None:
            return _pypardiso_spsolve(a_csc, demand)
        if self._solver is None:
            self._solver = _scipy_factorized(a_csc)
        return self._solver(demand)


@dataclass(frozen=True)
class NativeWorkerPayload:
    """Pickle-friendly worker input for native scoring — store paths + product ids only."""

    store_root: Path
    content_hash: str
    products: tuple[tuple[tuple[str, str], int], ...]
    methods: tuple[tuple[str, ...], ...]
    use_pardiso: bool


@dataclass(frozen=True)
class NativeScoreWorker:
    """Callable worker — picklable under the 'spawn' context.

    A frozen dataclass with ``__call__`` satisfies both the OOP rule (no module-level
    functions doing real work) and ProcessPoolExecutor's pickle requirement."""

    def __call__(self, payload: NativeWorkerPayload) -> dict[tuple[str, str], ScoringResult]:
        pkg = ScoringPackageStore(root=payload.store_root).read(payload.content_hash)
        scorer = NativeLciaScorer(package=pkg, use_pardiso=payload.use_pardiso)
        return scorer.score(list(payload.products), list(payload.methods))
