"""``dds-build-ecoinvent-catalog`` — emit ``registry/ecoinvent_catalog.parquet``.

After REFACTOR_FINAL F6 the builder reads from
``source/ecoinvent-3.9.1-cutoff-activities.json`` (a one-time bw2data
snapshot). No bw2data, no SQLite.
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from matching import EcoinventCatalogBuilder


class BuildEcoinventCatalogCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_ecoinvent_catalog"
    DESCRIPTION: ClassVar[str] = (
        "Materialise registry/ecoinvent_catalog.parquet from the snapshotted "
        "ecoinvent activity JSON file."
    )

    def execute(self, args: argparse.Namespace) -> None:
        EcoinventCatalogBuilder(settings=self.settings).build()


def main() -> int:
    return BuildEcoinventCatalogCli().run()


if __name__ == "__main__":
    sys.exit(main())
