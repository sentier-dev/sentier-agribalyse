"""``python -m cli.build_registry`` — build the ``registry/*.parquet`` files."""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from pipelines import RegistryBuildPipeline


class BuildRegistryCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_registry"
    DESCRIPTION: ClassVar[str] = "Build the sentier_agribalyse mapping registry parquets."

    def execute(self, args: argparse.Namespace) -> None:
        RegistryBuildPipeline(self.settings).run()


def main() -> int:
    return BuildRegistryCli().run()


if __name__ == "__main__":
    sys.exit(main())
