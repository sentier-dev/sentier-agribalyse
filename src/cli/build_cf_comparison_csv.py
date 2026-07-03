"""``dds-build-cf-comparison-csv`` — re-flatten the SimaPro-vs-registry CF
comparison join into the dashboard's ``cf_comparison.csv``.

Reads ``registry/cf_comparison_join.parquet`` (the complete per-flow join of
SimaPro adapted EF 3.1 CFs against our built registry CFs, written by
``dds-compare-cfs``) and re-serialises the matched rows as a flat CSV the static
dashboard fetches for its CF-comparison tab.

``dds-compare-cfs`` already writes ``cf_comparison.csv`` directly; this CLI is a
standalone re-flatten of an existing join parquet (e.g. after editing the row
filter). No comparison maths happen here — the parquet is the single source of
truth; this only projects the columns into a stable, comparison-first order.
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from core.logging import Logging
from reporting import CfComparisonCsvEmitter, CfComparisonJoinLoader


class BuildCfComparisonCsvCli(BaseCli):
    """``dds-build-cf-comparison-csv`` entry point."""

    PROG: ClassVar[str] = "cli.build_cf_comparison_csv"
    DESCRIPTION: ClassVar[str] = (
        "Flatten registry/cf_comparison_join.parquet into dashboard/cf_comparison.csv "
        "for the CF-comparison dashboard tab."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        return argparse.ArgumentParser(prog=cls.PROG, description=cls.DESCRIPTION)

    def execute(self, args: argparse.Namespace) -> None:
        paths = self.settings.paths
        join = CfComparisonJoinLoader(paths.registry_cf_comparison_join).load()
        out_path = CfComparisonCsvEmitter(
            out_path=paths.dashboard_cf_comparison_csv,
        ).write(join)
        Logging.get(self.PROG).info(
            "cf_comparison.dashboard.emitted",
            path=str(out_path),
            n_rows=len(join),
        )


def main() -> int:
    return BuildCfComparisonCsvCli().run()


if __name__ == "__main__":
    sys.exit(main())
