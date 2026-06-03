"""``python -m cli.mappings_comparison`` — regenerate ``to_review/mappings_comparison.xlsx``.

Standalone reader: re-runs ``MappingsComparisonExporter`` against an
already-linked ``sp.data`` cached from ``dds-link-all``. After
REFACTOR_FINAL F5 there's no bw2data project to bootstrap.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from typing import ClassVar

from cli._base import BaseCli
from exports import MappingsComparisonExporter


class MappingsComparisonCli(BaseCli):
    PROG: ClassVar[str] = "cli.mappings_comparison"
    DESCRIPTION: ClassVar[str] = (
        "Generate to_review/mappings_comparison.xlsx from the most recent "
        "linked AGB CSV (loaded from cache/importer_cache.pkl)."
    )

    def execute(self, args: argparse.Namespace) -> None:
        sp_data = self._load_sp_data()
        out = MappingsComparisonExporter(self.settings, sp_data=sp_data).export()
        print(out)

    def _load_sp_data(self) -> list[dict]:
        cache = self.settings.paths.importer_cache_pkl
        if not cache.exists():
            raise FileNotFoundError(
                f"importer cache {cache} not found. Run dds-link-all first to "
                f"materialise the parsed AGB CSV."
            )
        sp = pickle.loads(cache.read_bytes())
        return list(sp.data)


def main() -> int:
    return MappingsComparisonCli().run()


if __name__ == "__main__":
    sys.exit(main())
