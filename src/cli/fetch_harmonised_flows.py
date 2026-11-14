"""``dds-fetch-harmonised-flows`` — download the pinned Brightway Flows list.

Writes ``source/harmonised-flows-simple.json.gz`` (46 MB, gitignored) and
verifies its sha256. Run once per clone, before ``dds-build-registry``.
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from registry import HarmonisedFlowsFetcher


class FetchHarmonisedFlowsCli(BaseCli):
    PROG: ClassVar[str] = "cli.fetch_harmonised_flows"
    DESCRIPTION: ClassVar[str] = (
        "Download the pinned Brightway Flows release into source/ and verify its sha256."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--force",
            action="store_true",
            help="Replace an existing file that is not the pinned release.",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        fetcher = HarmonisedFlowsFetcher(settings=self.settings)
        outcome = fetcher.fetch(force=args.force)
        release = fetcher.release
        print(
            f"Brightway Flows {release.version} ({outcome}): {fetcher.path} "
            f"[doi:{release.doi}, ODbL]"
        )


def main() -> int:
    return FetchHarmonisedFlowsCli().run()


if __name__ == "__main__":
    sys.exit(main())
