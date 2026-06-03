"""``python -m cli.build_packages`` — author the publishable randonneur datapackages."""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from exports import RandonneurPackagesExporter


class BuildPackagesCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_packages"
    DESCRIPTION: ClassVar[str] = (
        "Author the AGB-3.2 randonneur datapackages: delete-aggregated, "
        "restore-names, and the residuals review xlsx."
    )

    def execute(self, args: argparse.Namespace) -> None:
        out = RandonneurPackagesExporter(self.settings).export()
        for k, v in out.items():
            print(f"{k}: {v}")


def main() -> int:
    return BuildPackagesCli().run()


if __name__ == "__main__":
    sys.exit(main())
