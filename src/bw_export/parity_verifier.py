"""``ParityVerifier`` — prove the datapackage equals the ScoringPackage.

Three guards: (1) the technosphere is square, (2) it factorizes (no
singular-matrix error), (3) scores computed from the emitted datapackages
match ``NativeLciaScorer`` within tolerance for a sample of products across
all methods.

The technosphere factorization is the expensive step, and it depends only
on the product demand — not the LCIA method. So we factorize **once per
product** through a single ``bw2calc`` LCA (built with one "anchor" method),
reuse the resulting inventory vector to characterise every other method by a
cheap dot product, and assert that this reused dot reproduces ``bw2calc``'s
own ``lcia`` for the anchor method (so the dot path is provably equivalent to
stock bw2calc). This turns an ``n_products x n_methods`` factorization loop
into ``n_products`` factorizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bw2calc as bc
import bw_processing as bwp
import numpy as np
from scipy.sparse.linalg import factorized

from bw_export.datapackage_writer import WriteResult
from scoring.native_scorer import NativeLciaScorer
from scoring.scoring_package import ScoringPackage

try:
    from pypardiso import spsolve as _pypardiso_spsolve
except ImportError:
    _pypardiso_spsolve = None


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    n_checked: int
    max_rel_error: float
    tolerance: float


@dataclass(frozen=True)
class ParityVerifier:
    tolerance: float = 1e-9

    def assert_square(self, package: ScoringPackage) -> None:
        a = package.technosphere.matrix
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"Technosphere is not square: {a.shape[0]}x{a.shape[1]}.")

    def assert_solvable(self, package: ScoringPackage) -> None:
        # The real technosphere carries zero-diagonal placeholder activities
        # that scipy SuperLU rejects as "Factor is exactly singular" even
        # though pypardiso's pivoting factorizes them fine (see the solver
        # note in CLAUDE.md). Prefer pardiso when installed — the same
        # policy as NativeLciaScorer — and fall back to scipy otherwise.
        a = package.technosphere.matrix.tocsc()
        if _pypardiso_spsolve is not None:
            _pypardiso_spsolve(a, np.ones(a.shape[0], dtype="float64"))
            return
        factorized(a)

    def verify(
        self,
        *,
        package: ScoringPackage,
        written: WriteResult,
        product_ids: list[int],
        methods: list[tuple[str, ...]],
    ) -> ParityResult:
        self.assert_square(package)
        self.assert_solvable(package)

        # ``use_pardiso=True`` falls back to scipy when pypardiso is absent, but
        # when it is installed this avoids a slow scipy SuperLU factorization of
        # the full technosphere (minutes on a ~20k-activity system).
        native = NativeLciaScorer(package=package, use_pardiso=True)
        native_scores = native.score(
            [((str(pid), str(pid)), pid) for pid in product_ids],
            methods,
        )

        inv_dp = bwp.load_datapackage(
            bwp.generic_directory_filesystem(dirpath=written.inventory_path)
        )
        if not methods:
            return ParityResult(
                passed=True, n_checked=0, max_rel_error=0.0, tolerance=self.tolerance
            )

        # Read each method's written CF table (flow_id -> cf) straight from the
        # emitted datapackage arrays, so we verify the files, not the in-memory
        # build. The anchor method drives the single per-product factorization.
        method_cfs = {m: self._load_cfs(written.method_paths[m]) for m in methods}
        anchor = methods[0]
        anchor_dp = bwp.load_datapackage(
            bwp.generic_directory_filesystem(dirpath=written.method_paths[anchor])
        )

        max_rel = 0.0
        n_checked = 0
        for pid in product_ids:
            native_res = native_scores[(str(pid), str(pid))]
            assert native_res.scores is not None, f"native scorer skipped {pid}"

            # One factorization per product: stock bw2calc solves the
            # technosphere and assembles the inventory for the anchor method.
            lca = bc.LCA({pid: 1.0}, data_objs=[inv_dp, anchor_dp])
            lca.lci()
            lca.lcia()
            inventory_vector = np.asarray(lca.inventory.sum(axis=1)).ravel()
            biosphere_row = lca.dicts.biosphere

            # Anchor: the reused dot must reproduce bw2calc's own lcia for the
            # anchor method — proving the dot path equals stock bw2calc before
            # we trust it for the remaining methods.
            anchor_manual = self._characterise(method_cfs[anchor], inventory_vector, biosphere_row)
            if abs(lca.score - anchor_manual) > 1e-6 * max(1.0, abs(lca.score)):
                raise RuntimeError(
                    f"parity anchor failed for product {pid}: reused dot "
                    f"{anchor_manual:.6g} != bw2calc lcia {lca.score:.6g}"
                )

            for method in methods:
                score = (
                    lca.score
                    if method == anchor
                    else self._characterise(method_cfs[method], inventory_vector, biosphere_row)
                )
                expected = native_res.scores[method]
                denom = abs(expected) if expected != 0 else 1.0
                rel = abs(score - expected) / denom
                max_rel = max(max_rel, rel)
                n_checked += 1

        return ParityResult(
            passed=max_rel <= self.tolerance,
            n_checked=n_checked,
            max_rel_error=max_rel,
            tolerance=self.tolerance,
        )

    @staticmethod
    def _load_cfs(method_dir: Path) -> dict[int, float]:
        """Read a written characterization datapackage's diagonal CFs."""
        indices = np.load(Path(method_dir) / "characterization.indices.npy")
        data = np.load(Path(method_dir) / "characterization.data.npy")
        return dict(zip(indices["row"].tolist(), data.tolist(), strict=True))

    @staticmethod
    def _characterise(
        cfs: dict[int, float], inventory_vector: np.ndarray, biosphere_row: dict[int, int]
    ) -> float:
        """``sum_f cf_f * inventory_f`` — the LCIA score from a prebuilt inventory."""
        total = 0.0
        for flow_id, cf in cfs.items():
            row = biosphere_row.get(flow_id)
            if row is not None:
                total += cf * float(inventory_vector[row])
        return total
