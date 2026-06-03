"""``python -m cli.link_all`` — run the full link pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from pipelines import LinkAllOptions, LinkAllPipeline


class LinkAllCli(BaseCli):
    PROG: ClassVar[str] = "cli.link_all"
    DESCRIPTION: ClassVar[str] = "Run the full AGB-3.2 linking pipeline."

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--skip-ecoinvent",
            action="store_true",
            help="Skip ecoinvent download/import (must already be loaded).",
        )
        p.add_argument(
            "--no-llm",
            action="store_true",
            help="Disable LLM overrides AND curated synonym fallback (fix 1.j).",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        settings = self.settings.with_no_llm() if args.no_llm else self.settings
        opts = LinkAllOptions(skip_ecoinvent=args.skip_ecoinvent)
        LinkAllPipeline(settings, options=opts).run()


def main() -> int:
    return LinkAllCli().run()


if __name__ == "__main__":
    sys.exit(main())
