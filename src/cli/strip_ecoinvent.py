"""``dds-strip-ecoinvent`` — write the ecoinvent-free Agribalyse CSV.

Maintainer-side: needs ADEME's full export (``source/AGB32_final.CSV``).
Removes every ``Copied from Ecoinvent`` process block as text and writes
``source/AGB32_ecoinvent_free.CSV.xz`` plus the manifest of removed
blocks. Fails if any process on the registry's deletion list survives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import ClassVar

from cli._base import BaseCli
from transforms import EcoinventStripper


class StripEcoinventCli(BaseCli):
    PROG: ClassVar[str] = "cli.strip_ecoinvent"
    DESCRIPTION: ClassVar[str] = (
        "Strip the embedded ecoinvent copies from the full SimaPro export "
        "and write the committed ecoinvent-free CSV and its manifest."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--full-csv",
            type=Path,
            default=None,
            help="Full SimaPro export (default: source/AGB32_final.CSV).",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        paths = self.settings.paths
        stripper = EcoinventStripper(required_identifiers=self._deletion_list())
        result = stripper.write(
            full_csv=args.full_csv or paths.agribalyse_csv,
            stripped_xz=paths.agribalyse_csv_ecoinvent_free_xz,
            manifest_json=paths.stripped_processes_json,
        )
        print(
            f"Removed {len(result.removed)} of {result.processes_total} process blocks; "
            f"{result.processes_kept} kept."
        )
        print(f"Wrote {paths.agribalyse_csv_ecoinvent_free_xz}")
        print(f"Wrote {paths.stripped_processes_json}")

    def _deletion_list(self) -> frozenset[str]:
        """Process identifiers of the aggregated ecoinvent copies (cross-check)."""
        payload = json.loads(
            self.settings.paths.delete_aggregated_processes_json.read_text(encoding="utf-8")
        )
        return frozenset(row["source"]["identifier"] for row in payload["delete"])


def main() -> int:
    return StripEcoinventCli().run()


if __name__ == "__main__":
    sys.exit(main())
