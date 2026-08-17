"""``dds-build-bw-package`` — export Brightway-ready datapackages.

End-to-end: load the ScoringPackage recorded by the last ``dds-link-all`` run,
embed the AWARE corrections as synthetic biosphere flows, write the
bw_processing datapackages + importer-sufficient metadata, copy in the
standalone ``import_into_brightway.py``, and verify parity before declaring
success.

The output contains ecoinvent LCI amounts composed from the locally
regenerated ``source/`` artifacts — it is licence-gated like ``source/``
itself: gitignored, blocked by the pre-commit EULA guard, never redistributed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd

from cli._base import BaseCli
from scoring.method_slug import MethodSlug
from scoring.native_scorer import NativeLciaScorer
from scoring.scoring_package import ScoringPackage
from scoring.scoring_package_locator import ScoringPackageLocator

try:
    from bw_export.catalog_key_resolver import CatalogKeyResolver
    from bw_export.correction_embedder import CorrectionEmbedder
    from bw_export.datapackage_writer import DatapackageWriter
    from bw_export.metadata_emitter import MetadataEmitter
    from bw_export.parity_verifier import ParityVerifier
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "dds-build-bw-package needs bw_processing + bw2calc. "
        "Install the extra: pip install -e '.[bw]'"
    ) from exc

# The standalone importer copied verbatim into the export root.
_IMPORTER_SRC = Path(__file__).resolve().parents[1] / "bw_import" / "import_into_brightway.py"
# Tolerance written to parity_samples.json when the export-time parity check is
# skipped (with parity on, the verifier's own tighter tolerance is written).
_DEFAULT_TOLERANCE = 1e-6


@dataclass
class BuildBwPackageCli(BaseCli):
    PROG: ClassVar[str] = "dds-build-bw-package"
    DESCRIPTION: ClassVar[str] = (
        "Export the linked Agribalyse 3.2 x ecoinvent x EF v3.1 system as "
        "bw_processing datapackages (runnable directly through bw2calc) plus a "
        "standalone import_into_brightway.py for Brightway / Activity Browser."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--out",
            type=Path,
            default=Path("bw_package"),
            help="Output root (default: bw_package/ — gitignored; the "
            "export contains licensed ecoinvent amounts, never "
            "commit or redistribute it).",
        )
        p.add_argument(
            "--parity-n",
            type=int,
            default=3,
            help="Number of products to parity-check (default: 3).",
        )
        p.add_argument(
            "--parity-full", action="store_true", help="Parity-check every mapped product."
        )
        p.add_argument(
            "--skip-parity",
            action="store_true",
            help="Skip the bw2calc round-trip (still checks squareness).",
        )
        return p

    def sample_product_ids(
        self, package: ScoringPackage, n: int | None, full: bool = False
    ) -> list[int]:
        all_ids = sorted(package.technosphere.row_id_to_idx)
        if full:
            return all_ids
        count = min(n or 0, len(all_ids))
        if count <= 0:
            return []
        if count >= len(all_ids):
            return all_ids
        step = max(1, len(all_ids) // count)
        return all_ids[::step][:count]

    def execute(self, args: argparse.Namespace) -> None:
        package = ScoringPackageLocator(settings=self.settings).load()
        embedded = CorrectionEmbedder().embed(package)
        written = DatapackageWriter().write(embedded, out_root=args.out)

        product_ids = self.sample_product_ids(package, n=args.parity_n, full=args.parity_full)
        parity: dict = {"skipped": True}
        tolerance = _DEFAULT_TOLERANCE
        if not args.skip_parity:
            result = ParityVerifier().verify(
                package=package,
                written=written,
                product_ids=product_ids,
                methods=list(package.methods),
            )
            tolerance = result.tolerance
            parity = {
                "passed": result.passed,
                "n_checked": result.n_checked,
                "max_rel_error": result.max_rel_error,
                "tolerance": result.tolerance,
            }
            if not result.passed:
                raise RuntimeError(
                    f"Parity check FAILED: max rel error {result.max_rel_error:.3e} "
                    f"> tolerance {result.tolerance:.1e}. Datapackage not trustworthy."
                )

        paths = self.settings.paths
        product_catalog = pd.read_parquet(paths.registry_product_catalog)
        biosphere_catalog = pd.read_parquet(paths.registry_biosphere_catalog)
        ecoinvent_catalog = pd.read_parquet(paths.registry_ecoinvent_catalog)
        # Per-column label catalog; when present it is authoritative (it also
        # names Allocator multifunctional-split columns the hash join can't).
        activity_catalog = (
            pd.read_parquet(paths.registry_activity_catalog)
            if paths.registry_activity_catalog.exists()
            else None
        )

        resolver = CatalogKeyResolver(
            product_catalog=product_catalog,
            biosphere_catalog=biosphere_catalog,
            ecoinvent_catalog=ecoinvent_catalog,
            activity_catalog=activity_catalog,
            synthetic_flow_codes={
                fid: f"aware-{MethodSlug.encode(method)}"
                for method, fid in embedded.synthetic_flow_ids.items()
            },
        )
        # Native scores for the sampled products feed parity_samples.json, which
        # the shipped importer's --verify re-checks against the user's own
        # bw2calc. Skipped when parity is skipped (no trusted scores to embed).
        parity_scores = {} if args.skip_parity else self._native_parity_scores(package, product_ids)

        manifest_extra: dict = {
            "content_hash": package.content_hash,
            "ecoinvent_version": self.settings.resolved_ecoinvent_version,
            "built_at": datetime.now(UTC).isoformat(),
        }

        MetadataEmitter().emit(
            out_root=args.out,
            embedded=embedded,
            written=written,
            activity_resolver=resolver.activity,
            bio_resolver=resolver.biosphere,
            product_catalog=product_catalog,
            parity_scores=parity_scores,
            parity_tolerance=tolerance,
            manifest_extra=manifest_extra,
            parity=parity,
        )
        shutil.copy2(_IMPORTER_SRC, args.out / "import_into_brightway.py")
        self._print_export_summary(args.out, parity)

    @staticmethod
    def _native_parity_scores(
        package: ScoringPackage, product_ids: list[int]
    ) -> dict[int, dict[tuple[str, ...], float]]:
        if not product_ids:
            return {}
        scorer = NativeLciaScorer(package=package, use_pardiso=True)
        results = scorer.score(
            [((str(pid), str(pid)), pid) for pid in product_ids],
            list(package.methods),
        )
        out: dict[int, dict[tuple[str, ...], float]] = {}
        for pid in product_ids:
            res = results[(str(pid), str(pid))]
            if res.scores is not None:
                out[pid] = dict(res.scores)
        return out

    def _print_export_summary(self, out: Path, parity: dict) -> None:
        print(f"\nbw_processing datapackages written to {out}/")
        if parity.get("skipped"):
            print("  parity check: SKIPPED")
        else:
            print(
                f"  parity check: PASS ({parity['n_checked']} scores, "
                f"max rel error {parity['max_rel_error']:.2e})"
            )
        print(f"  score directly: cd {out} && python run_example.py")
        print(f"  import into Brightway: cd {out} && python import_into_brightway.py")
        print(
            "  NOTE: the export contains licensed ecoinvent amounts — do not "
            "commit or redistribute it."
        )


def main() -> int:
    return BuildBwPackageCli().run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
