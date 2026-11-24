"""``python -m cli.link_all`` — run the full link pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from config import Settings
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
        p.add_argument(
            "--csv",
            choices=Settings.AGRIBALYSE_CSV_CHOICES,
            default="auto",
            help=(
                "Which Agribalyse CSV to read. 'auto' (default) reads ADEME's full "
                "export when it is in source/, else the ecoinvent-free CSV shipped "
                "with the repository; 'ecoinvent-free' reads that one although the "
                "full export is there (to build a package for dds-build-skeleton). "
                "Commands without --csv (dds-set-parameter, dds-list-parameters, ...) "
                "read what the last dds-link-all asked for."
            ),
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        settings = self.settings.with_no_llm() if args.no_llm else self.settings
        settings = settings.with_agribalyse_csv(args.csv)
        opts = LinkAllOptions(skip_ecoinvent=args.skip_ecoinvent)
        LinkAllPipeline(settings, options=opts).run()


def main() -> int:
    return LinkAllCli().run()


if __name__ == "__main__":
    sys.exit(main())
