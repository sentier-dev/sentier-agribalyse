"""``python -m cli.snapshot_ecoinvent_exchanges`` — bootstrap-time exchange dump.

Reads a ``bw2data``-shaped SQLite (``activitydataset`` + ``exchangedataset``
tables) and emits ``source/ecoinvent-3.9.1-cutoff-exchanges.parquet`` —
the long-form snapshot the runtime ScoringPackage builder concatenates
with the AGB frame to form a full square technosphere.

The runtime never reads SQLite. This CLI is bootstrap-only: it lifts
the exchanges out of a Brightway project once, then the runtime works
from the parquet alone. Pure ``stdlib + pandas + pyarrow`` — no
``bw2data`` import — so the snapshot can be regenerated even on a
machine that only has the bootstrap deps absent.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from cli._base import BaseCli
from core.logging import Logging


@dataclass(frozen=True)
class EcoinventSqliteReader:
    """Read all exchanges owned by activities of one ``database`` from a
    bw2data SQLite. The ``data`` BLOB is a pickled dict; we only pull
    the fields the runtime actually needs (input ref, amount, type) so
    the parquet stays compact."""

    sqlite_path: Path
    db_name: str

    def read_exchanges(self) -> pd.DataFrame:
        if not self.sqlite_path.exists():
            raise FileNotFoundError(
                f"Brightway SQLite not found: {self.sqlite_path}. "
                f"This snapshotter is bootstrap-only — point it at the "
                f"`databases.db` of a project that already loaded ecoinvent."
            )
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT e.output_database, e.output_code,
                       e.input_database,  e.input_code,
                       e.type, e.data
                FROM exchangedataset e
                JOIN activitydataset a
                  ON a.database = e.output_database
                 AND a.code     = e.output_code
                WHERE a.database = ?
                """,
                (self.db_name,),
            ).fetchall()
        records: list[dict[str, object]] = []
        for od, oc, idb, ic, typ, blob in rows:
            payload = pickle.loads(blob)
            records.append(
                {
                    "output_database": od,
                    "output_code": oc,
                    "input_database": idb,
                    "input_code": ic,
                    "amount": float(payload.get("amount", 0.0)),
                    "type": typ,
                }
            )
        df = pd.DataFrame.from_records(records)
        # Stable sort so the parquet is byte-identical across runs;
        # downstream content-hashing of the ScoringPackage benefits.
        if not df.empty:
            df = df.sort_values(
                ["output_database", "output_code", "input_database", "input_code", "type"],
                kind="stable",
            ).reset_index(drop=True)
        return df


@dataclass(frozen=True)
class EcoinventExchangesSnapshotter:
    """One-shot writer. Bootstrapping responsibility: reader → parquet."""

    reader: EcoinventSqliteReader
    output_path: Path

    def snapshot(self) -> dict[str, int]:
        df = self.reader.read_exchanges()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, engine="pyarrow", compression="snappy")
        type_counts = df["type"].value_counts().to_dict() if not df.empty else {}
        return {
            "n_rows": len(df),
            "by_type": {str(k): int(v) for k, v in type_counts.items()},
            "path": str(self.output_path),
        }


@dataclass
class SnapshotEcoinventExchangesCli(BaseCli):
    PROG: ClassVar[str] = "cli.snapshot_ecoinvent_exchanges"
    DESCRIPTION: ClassVar[str] = (
        "One-time bootstrap dump of ecoinvent exchanges from a bw2data "
        "SQLite to source/ecoinvent-*.parquet (no bw2data dep)."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--sqlite",
            type=Path,
            required=True,
            help="Path to a bw2data project's lci/databases.db.",
        )
        p.add_argument(
            "--db-name",
            default="ecoinvent-3.9.1-cutoff",
            help="Brightway database name to dump (default: ecoinvent-3.9.1-cutoff).",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        log = Logging.get(self.PROG)
        reader = EcoinventSqliteReader(sqlite_path=args.sqlite, db_name=args.db_name)
        snapshotter = EcoinventExchangesSnapshotter(
            reader=reader,
            output_path=self.settings.paths.ecoinvent_exchanges_parquet,
        )
        report = snapshotter.snapshot()
        log.info("ecoinvent.exchanges.snapshot", **report)
        print(json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    return SnapshotEcoinventExchangesCli().run()


if __name__ == "__main__":
    sys.exit(main())
