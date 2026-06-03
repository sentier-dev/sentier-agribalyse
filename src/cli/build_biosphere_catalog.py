"""``dds-build-biosphere-catalog`` — emit ``registry/biosphere_catalog.parquet``.

Reads ``source/biosphere3-flows.json``,
``source/ecoinvent-3.9.1-biosphere-flows.json`` (optional), and
``registry/ef_flows.parquet`` to produce the runtime biosphere catalog.
No bw2data, no SQLite (REFACTOR_FINAL F6).
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from matching import BiosphereRegistryBuilder


class BuildBiosphereCatalogCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_biosphere_catalog"
    DESCRIPTION: ClassVar[str] = (
        "Materialise registry/biosphere_catalog.parquet from the snapshotted "
        "biosphere3 + ecoinvent biosphere JSON files plus registry/ef_flows.parquet."
    )

    def execute(self, args: argparse.Namespace) -> None:
        BiosphereRegistryBuilder(settings=self.settings).build()


def main() -> int:
    return BuildBiosphereCatalogCli().run()


if __name__ == "__main__":
    sys.exit(main())
