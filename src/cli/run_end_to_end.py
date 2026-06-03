"""``python -m cli.run_end_to_end`` — link → register methods → score sample."""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from pipelines import EndToEndOptions, EndToEndPipeline


class RunEndToEndCli(BaseCli):
    PROG: ClassVar[str] = "cli.run_end_to_end"
    DESCRIPTION: ClassVar[str] = "Link AGB-3.2, register LCIA, score a sample."

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument("--skip-linking", action="store_true")
        p.add_argument("--skip-ecoinvent", action="store_true")
        p.add_argument("--solver", choices=("scipy", "pardiso"), default="scipy")
        p.add_argument("--n-products", type=int, default=5)
        return p

    def execute(self, args: argparse.Namespace) -> None:
        opts = EndToEndOptions(
            skip_linking=args.skip_linking,
            skip_ecoinvent=args.skip_ecoinvent,
            solver=args.solver,
            n_sample_products=args.n_products,
        )
        EndToEndPipeline(self.settings, options=opts).run()


def main() -> int:
    return RunEndToEndCli().run()


if __name__ == "__main__":
    sys.exit(main())
