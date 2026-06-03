"""``dds-build-ef-flows-registry`` — emit ``registry/ef_flows.parquet``.

Reads the registry's target index plus the snapshotted EF v3.1 method
list at ``source/ef-v31-methods.json``. No bw2data, no SQLite
(REFACTOR_FINAL F6).
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from ef import EfFlowsRegistryBuilder
from registry import MappingRegistry


class BuildEfFlowsRegistryCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_ef_flows_registry"
    DESCRIPTION: ClassVar[str] = (
        "Materialise registry/ef_flows.parquet — the hygiene-filtered EF flow universe."
    )

    def execute(self, args: argparse.Namespace) -> None:
        s = self.settings
        registry = MappingRegistry.load(s)
        EfFlowsRegistryBuilder(settings=s, registry=registry).build()


def main() -> int:
    return BuildEfFlowsRegistryCli().run()


if __name__ == "__main__":
    sys.exit(main())
