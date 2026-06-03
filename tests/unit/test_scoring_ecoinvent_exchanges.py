"""Unit tests for ``EcoinventExchangesIngester`` and the bootstrap
``EcoinventSqliteReader`` / ``EcoinventExchangesSnapshotter``.

The runtime ingester is the cold path that turns
``source/ecoinvent-3.9.1-cutoff-exchanges.parquet`` into the long-form
schema the matrix builders consume. The snapshotter is one-shot
bootstrap glue — pure ``stdlib + pandas`` so the runtime image stays
``bw2data``-free.
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from cli.snapshot_ecoinvent_exchanges import (
    EcoinventExchangesSnapshotter,
    EcoinventSqliteReader,
)
from scoring.ecoinvent_exchanges import EcoinventExchangesIngester
from scoring.exchange_frame_builder import ExchangeFrameBuilder


def _seed_sqlite(path: Path, db_name: str, exchanges: list[dict]) -> None:
    """Build a minimal bw2data-shaped SQLite for the snapshotter tests."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE activitydataset (
            id INTEGER PRIMARY KEY, data BLOB NOT NULL,
            code TEXT NOT NULL, database TEXT NOT NULL,
            location TEXT, name TEXT, product TEXT, type TEXT
        );
        CREATE TABLE exchangedataset (
            id INTEGER PRIMARY KEY, data BLOB NOT NULL,
            input_code TEXT NOT NULL, input_database TEXT NOT NULL,
            output_code TEXT NOT NULL, output_database TEXT NOT NULL,
            type TEXT NOT NULL
        );
        """
    )
    # Owning activities — one per distinct (db, code) on the output side.
    seen_acts: set[tuple[str, str]] = set()
    for e in exchanges:
        key = (e["output_database"], e["output_code"])
        if key in seen_acts:
            continue
        seen_acts.add(key)
        conn.execute(
            "INSERT INTO activitydataset (data, code, database, type) VALUES (?, ?, ?, ?)",
            (pickle.dumps({"code": key[1]}), key[1], key[0], "process"),
        )
    for e in exchanges:
        payload = {"amount": e["amount"], "type": e["type"]}
        conn.execute(
            "INSERT INTO exchangedataset "
            "(data, input_code, input_database, output_code, output_database, type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                pickle.dumps(payload),
                e["input_code"],
                e["input_database"],
                e["output_code"],
                e["output_database"],
                e["type"],
            ),
        )
    conn.commit()
    conn.close()


class TestEcoinventSqliteReader:
    def test_extracts_exchanges_for_named_database_only(self, tmp_path: Path):
        sqlite = tmp_path / "lci.db"
        _seed_sqlite(
            sqlite,
            "ecoinvent",
            [
                # Two activities in 'ecoinvent', one in 'other-db' that must be ignored.
                {
                    "output_database": "ecoinvent",
                    "output_code": "act1",
                    "input_database": "ecoinvent",
                    "input_code": "act1",
                    "amount": 1.0,
                    "type": "production",
                },
                {
                    "output_database": "ecoinvent",
                    "output_code": "act1",
                    "input_database": "biosphere",
                    "input_code": "co2",
                    "amount": 0.5,
                    "type": "biosphere",
                },
                {
                    "output_database": "other-db",
                    "output_code": "outsider",
                    "input_database": "other-db",
                    "input_code": "outsider",
                    "amount": 9.0,
                    "type": "production",
                },
            ],
        )
        reader = EcoinventSqliteReader(sqlite_path=sqlite, db_name="ecoinvent")
        df = reader.read_exchanges()

        assert set(df["output_database"]) == {"ecoinvent"}
        assert len(df) == 2
        assert {
            "output_database",
            "output_code",
            "input_database",
            "input_code",
            "amount",
            "type",
        } <= set(df.columns)
        # Stable sort key — same dataframe across runs (this matters for the
        # downstream content-hash of the ScoringPackage parquet).
        assert df["amount"].iloc[0] == pytest.approx(0.5)
        assert df["amount"].iloc[1] == pytest.approx(1.0)

    def test_missing_sqlite_raises_actionable_error(self, tmp_path: Path):
        reader = EcoinventSqliteReader(sqlite_path=tmp_path / "missing.db", db_name="x")
        with pytest.raises(FileNotFoundError, match="bootstrap-only"):
            reader.read_exchanges()


class TestEcoinventExchangesSnapshotter:
    def test_round_trip_through_parquet(self, tmp_path: Path):
        sqlite = tmp_path / "lci.db"
        _seed_sqlite(
            sqlite,
            "ecoinvent",
            [
                {
                    "output_database": "ecoinvent",
                    "output_code": "a",
                    "input_database": "ecoinvent",
                    "input_code": "a",
                    "amount": 1.0,
                    "type": "production",
                },
                {
                    "output_database": "ecoinvent",
                    "output_code": "a",
                    "input_database": "ecoinvent",
                    "input_code": "b",
                    "amount": 2.5,
                    "type": "technosphere",
                },
            ],
        )
        out = tmp_path / "snap.parquet"
        report = EcoinventExchangesSnapshotter(
            reader=EcoinventSqliteReader(sqlite_path=sqlite, db_name="ecoinvent"),
            output_path=out,
        ).snapshot()
        assert report["n_rows"] == 2
        assert report["by_type"] == {"production": 1, "technosphere": 1}
        assert out.exists()
        roundtrip = pd.read_parquet(out)
        assert set(roundtrip.columns) >= {"output_database", "output_code", "amount", "type"}


class TestEcoinventExchangesIngester:
    def test_load_long_projects_to_runtime_schema(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                {
                    "output_database": "eco",
                    "output_code": "a",
                    "input_database": "eco",
                    "input_code": "a",
                    "amount": 1.0,
                    "type": "production",
                },
                {
                    "output_database": "eco",
                    "output_code": "a",
                    "input_database": "bio",
                    "input_code": "co2",
                    "amount": 0.5,
                    "type": "biosphere",
                },
            ]
        )
        path = tmp_path / "ei.parquet"
        df.to_parquet(path)

        long = EcoinventExchangesIngester(path=path).load_long()

        # Schema parity with ExchangeFrameBuilder.long_from_sp_data.
        assert list(long.columns) == [
            "output_id",
            "input_id",
            "amount",
            "edge_type",
            "is_biosphere",
            "allocation_factor",
        ]
        # ids match the canonical hash so AGB and ecoinvent rows can be
        # concatenated and produce a single coherent matrix.
        prod_row = long.iloc[0]
        assert prod_row["output_id"] == ExchangeFrameBuilder.flow_id_for(("eco", "a"))
        assert prod_row["input_id"] == ExchangeFrameBuilder.flow_id_for(("eco", "a"))
        bio_row = long.iloc[1]
        assert bio_row["is_biosphere"] is True or bio_row["is_biosphere"] == True  # noqa: E712
        # Single-product ecoinvent → no allocation needed; the Allocator
        # passes these through.
        assert (long["allocation_factor"] == 1.0).all()

    def test_missing_parquet_raises_actionable_error(self, tmp_path: Path):
        ing = EcoinventExchangesIngester(path=tmp_path / "missing.parquet")
        with pytest.raises(FileNotFoundError, match="dds-snapshot-ecoinvent-exchanges"):
            ing.load_long()

    def test_waste_treatment_sign_normalisation(self, tmp_path: Path):
        """Ecoinvent cutoff stores waste-treatment activities with
        ``production amount = -1`` and any consumer of the waste product
        with ``technosphere amount = -X``. The ingester must flip both so
        downstream code sees the standard "production positive, consumption
        positive" form and the matrix balance works when mixed with AGB
        rows that use the conventional positive sign."""
        df = pd.DataFrame(
            [
                # Waste-treatment activity: production = -1.
                {
                    "output_database": "eco",
                    "output_code": "treatment",
                    "input_database": "eco",
                    "input_code": "treatment",
                    "amount": -1.0,
                    "type": "production",
                },
                # Treatment's electricity input — positive, must NOT flip.
                {
                    "output_database": "eco",
                    "output_code": "treatment",
                    "input_database": "eco",
                    "input_code": "electricity",
                    "amount": 0.01,
                    "type": "technosphere",
                },
                # A producer of the waste — negative tech amount on the
                # treatment product, must be flipped to positive.
                {
                    "output_database": "eco",
                    "output_code": "bakery",
                    "input_database": "eco",
                    "input_code": "treatment",
                    "amount": -0.5,
                    "type": "technosphere",
                },
                # Bakery's positive-sign producer of its own bread.
                {
                    "output_database": "eco",
                    "output_code": "bakery",
                    "input_database": "eco",
                    "input_code": "bakery",
                    "amount": 1.0,
                    "type": "production",
                },
                # An unrelated negative-tech edge whose input is NOT a
                # waste-treatment activity — must be left alone (these
                # represent allocation by-products / substitution-shaped
                # edges in the live snapshot).
                {
                    "output_database": "eco",
                    "output_code": "bakery",
                    "input_database": "eco",
                    "input_code": "byproduct",
                    "amount": -2.0,
                    "type": "technosphere",
                },
            ]
        )
        path = tmp_path / "ei.parquet"
        df.to_parquet(path)
        long = EcoinventExchangesIngester(path=path).load_long()
        treatment_id = ExchangeFrameBuilder.flow_id_for(("eco", "treatment"))
        bakery_id = ExchangeFrameBuilder.flow_id_for(("eco", "bakery"))
        byprod_id = ExchangeFrameBuilder.flow_id_for(("eco", "byproduct"))

        # Treatment production: flipped to +1.
        treatment_prod = long[
            (long["output_id"] == treatment_id) & (long["edge_type"] == "production")
        ]
        assert treatment_prod["amount"].iloc[0] == pytest.approx(1.0)
        # Treatment's positive technosphere edge: unchanged (+0.01).
        elec_row = long[(long["output_id"] == treatment_id) & (long["edge_type"] == "technosphere")]
        assert elec_row["amount"].iloc[0] == pytest.approx(0.01)
        # Bakery's biowaste-treatment input: flipped to +0.5.
        baker_to_treat = long[
            (long["output_id"] == bakery_id)
            & (long["input_id"] == treatment_id)
            & (long["edge_type"] == "technosphere")
        ]
        assert baker_to_treat["amount"].iloc[0] == pytest.approx(0.5)
        # Unrelated negative-tech edge (input not a neg-prod activity): unchanged.
        byprod_row = long[
            (long["output_id"] == bakery_id)
            & (long["input_id"] == byprod_id)
            & (long["edge_type"] == "technosphere")
        ]
        assert byprod_row["amount"].iloc[0] == pytest.approx(-2.0)

    def test_empty_parquet_returns_empty_long(self, tmp_path: Path):
        empty = pd.DataFrame(
            columns=[
                "output_database",
                "output_code",
                "input_database",
                "input_code",
                "amount",
                "type",
            ]
        )
        path = tmp_path / "empty.parquet"
        empty.to_parquet(path)
        long = EcoinventExchangesIngester(path=path).load_long()
        assert long.empty
        assert list(long.columns) == [
            "output_id",
            "input_id",
            "amount",
            "edge_type",
            "is_biosphere",
            "allocation_factor",
        ]
