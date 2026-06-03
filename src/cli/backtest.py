"""``python -m cli.backtest`` — backtest LCIA scores against ADEME's reference."""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from pipelines import BacktestOptions, BacktestPipeline


class BacktestCli(BaseCli):
    PROG: ClassVar[str] = "cli.backtest"
    DESCRIPTION: ClassVar[str] = "Score every mapped product against ADEME and write parquet diffs."

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument("--solver", choices=("scipy", "pardiso"), default="scipy")
        p.add_argument(
            "--workers",
            type=int,
            default=1,
            help=(
                "Number of parallel scoring workers. Each spawns its own brightway "
                "project + factorized LCA, so memory scales with N. 1 = serial."
            ),
        )
        p.add_argument(
            "--n-products",
            type=int,
            default=None,
            help="Score only the first N reference rows (deterministic). Default: all.",
        )
        p.add_argument(
            "--exact-name-only",
            action="store_true",
            help=(
                "Restrict the backtest to ADEME reference rows whose 'LCI Name' "
                "exactly matches an AGB DB process/product name (the SimaPro-"
                "restored name). No CIQUAL or substring fallback."
            ),
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        BacktestPipeline(
            self.settings,
            options=BacktestOptions(
                solver=args.solver,
                n_workers=args.workers,
                n_products=args.n_products,
                match_mode=("exact_name" if args.exact_name_only else "all"),
            ),
        ).run()


def main() -> int:
    return BacktestCli().run()


if __name__ == "__main__":
    sys.exit(main())
