"""``dds-build-flow-decomp`` — one-shot build of per-product flow JSONs.

Loads the cached :class:`~scoring.scoring_package.ScoringPackage` (the
same one ``dds-backtest`` / ``dds-decompose-score`` consume) and the
backtest ``scores.parquet`` (filter ``mapped == True``), then walks
every product × every short-id method in
:class:`~reporting.backtest_dashboard_csv.BacktestPass1Emitter` and
writes one JSON per product to ``--out`` (default
``dashboard/decomp/``). The dashboard fetches these lazily on cell
click.

Standalone by design — not wired into ``dds-backtest`` so a backtest
run does not pay the 3-5 minute decomposition cost. Re-run this CLI
after the registry or scoring package changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from cli._base import BaseCli
from cli.decompose_score import MethodAliases
from core.logging import Logging
from reporting.backtest_dashboard_csv import BacktestPass1Emitter
from reporting.flow_decomposition import FlowDecompositionEmitter
from reporting.simapro_cf_lookup import SimaProCfLookup
from scoring.decomposer import ScoreDecomposer
from scoring.product_catalog import ProductCatalog
from scoring.scoring_package import ScoringPackageStore


@dataclass
class BuildFlowDecompCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_flow_decomp"
    DESCRIPTION: ClassVar[str] = (
        "Build per-product flow-decomposition JSONs under dashboard/decomp/ "
        "for the backtest dashboard's drill-down panel. One-shot; rerun "
        "after the registry or scoring package changes."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--top-n",
            type=int,
            default=25,
            help="Per (product, method), keep this many flows by |contribution| (default: 25).",
        )
        p.add_argument(
            "--out",
            type=Path,
            default=None,
            help="Output directory (default: <dashboard>/decomp).",
        )
        p.add_argument(
            "--solver",
            choices=("scipy", "pardiso"),
            default="pardiso",
            help=(
                "Linear solver for A^-1 d. 'pardiso' is the project default "
                "because scipy SuperLU rejects the post-refactor matrix's "
                "37 zero-diagonal placeholder activities."
            ),
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        package = self._load_package()
        product_catalog = ProductCatalog.load(self.settings.paths.registry_product_catalog)
        biosphere = pd.read_parquet(self.settings.paths.registry_biosphere_catalog)
        scores_df = self._load_scores_df()
        out_dir = args.out if args.out is not None else (self.settings.paths.dashboard / "decomp")

        unknown = set(BacktestPass1Emitter.SHORT_ORDER) - set(MethodAliases.SHORT_TO_FULL)
        if unknown:
            raise ValueError(
                f"BacktestPass1Emitter.SHORT_ORDER has aliases not in MethodAliases: "
                f"{sorted(unknown)}. Add them to MethodAliases.SHORT_TO_FULL."
            )
        method_short_to_full = {
            short: MethodAliases.resolve(short) for short in BacktestPass1Emitter.SHORT_ORDER
        }

        decomposer = ScoreDecomposer(
            package=package,
            product_catalog=product_catalog,
            biosphere_catalog=biosphere,
            use_pardiso=(args.solver == "pardiso"),
        )
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=biosphere,
            method_short_to_full=method_short_to_full,
            out_dir=out_dir,
            top_n=args.top_n,
            simapro_cf=self._load_simapro_cf(),
        )
        emitter.write(scores_df)

    # ------------------------------------------------------------------

    def _load_package(self):
        report_path = self.settings.paths.dashboard_run_report
        if not report_path.exists():
            raise FileNotFoundError(
                f"run_report.json not found at {report_path}. Run dds-link-all first."
            )
        report = json.loads(report_path.read_text())
        content_hash = report["stages"]["scoring_package"]["content_hash"]
        return ScoringPackageStore(root=self.settings.paths.scoring_packages_root).read(
            content_hash
        )

    def _load_scores_df(self) -> pd.DataFrame:
        scores_path = self.settings.paths.dashboard_backtest_dir / "scores.parquet"
        if not scores_path.exists():
            raise FileNotFoundError(
                f"scores.parquet not found at {scores_path}. Run dds-backtest first."
            )
        return pd.read_parquet(scores_path)

    def _load_simapro_cf(self) -> SimaProCfLookup | None:
        """Per-flow SimaPro CFs for the side-by-side comparison column.

        Sourced from the properly-matched CF-comparison sidecar. Optional: if
        the sidecar is missing the panel still renders (SimaPro column shows
        em-dashes). Rebuild it with ``dds-compare-cfs``.
        """
        by_code_path = self.settings.paths.registry_cf_comparison_by_code
        if not by_code_path.exists():
            Logging.get(self.PROG).warning(
                "flow_decomp.simapro_cf_missing",
                path=str(by_code_path),
                hint="run dds-compare-cfs to enable the SimaPro CF column",
            )
            return None
        return SimaProCfLookup.from_parquet(by_code_path)


def main() -> int:
    return BuildFlowDecompCli().run()


if __name__ == "__main__":
    sys.exit(main())
