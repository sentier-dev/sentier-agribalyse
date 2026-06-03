"""Unit tests for ``ProductCatalogBuilder``.

L4 will read this parquet to map ``(ciqual_code, lci_name) → product_id``
without going through ``bw2data.Database``. The contract this file pins
down: schema, deterministic sort, atomic write, integer ids that match
``ExchangeFrameBuilder.flow_id_for``, and idempotent re-runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.product_catalog import ProductCatalogBuilder


def _activity(database: str, code: str, **fields) -> dict:
    return {"database": database, "code": code, **fields}


class TestProductCatalogBuilderSchema:
    def test_writes_one_row_per_activity(self, tmp_path: Path):
        sp_data = [
            _activity(
                "agb",
                "p1",
                name="wheat",
                type="process",
                unit="kg",
            ),
            _activity(
                "agb",
                "p2",
                name="rice",
                type="process",
                unit="kg",
            ),
        ]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)

        df = pd.read_parquet(target)
        assert len(df) == 2
        assert list(df.columns) == [
            "database",
            "code",
            "name",
            "type",
            "unit",
            "product_id",
        ]

    def test_columns_have_expected_dtypes(self, tmp_path: Path):
        sp_data = [_activity("agb", "p1", name="wheat", type="process", unit="kg")]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)

        df = pd.read_parquet(target)
        assert df["database"].dtype == "string"
        assert df["code"].dtype == "string"
        assert df["name"].dtype == "string"
        assert df["type"].dtype == "string"
        assert df["unit"].dtype == "string"
        assert df["product_id"].dtype == "int64"


class TestProductCatalogBuilderDeterminism:
    def test_rows_sorted_by_database_then_code(self, tmp_path: Path):
        # Insertion order is purposely jumbled — output must come back
        # alphabetically sorted on (database, code).
        sp_data = [
            _activity("agb", "z-product", name="z", type="process", unit="kg"),
            _activity("ecoinvent", "a-product", name="a", type="process", unit="kg"),
            _activity("agb", "a-product", name="a-agb", type="process", unit="kg"),
        ]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)

        df = pd.read_parquet(target)
        observed = list(zip(df["database"].tolist(), df["code"].tolist(), strict=False))
        assert observed == [
            ("agb", "a-product"),
            ("agb", "z-product"),
            ("ecoinvent", "a-product"),
        ]

    def test_product_id_matches_flow_id_for(self, tmp_path: Path):
        # The whole point of having one canonical hash: rows in the
        # catalog must collide with the ids the technosphere row map
        # uses. If they drift, BacktestPipeline can't find products.
        sp_data = [
            _activity("agb", "wheat", name="wheat", type="process", unit="kg"),
            _activity("ecoinvent", "fertiliser", name="fert", type="process", unit="kg"),
        ]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)

        df = pd.read_parquet(target)
        for _, row in df.iterrows():
            expected = ExchangeFrameBuilder.flow_id_for((row["database"], row["code"]))
            assert int(row["product_id"]) == expected

    def test_product_id_uses_production_input_when_present(self, tmp_path: Path):
        # AGB processes carry a production edge whose ``input`` is a
        # *separate* product node (different code from the process).
        # The technosphere row map is keyed by that input, so the
        # catalog must follow suit — otherwise scorers report
        # "product id not in technosphere" for every AGB process.
        sp_data = [
            {
                "database": "agb",
                "code": "process-1",
                "name": "[Dummy] Algae",
                "type": "process",
                "unit": "kg",
                "exchanges": [
                    {"type": "production", "input": ("agb", "product-1"), "amount": 1.0},
                ],
            },
        ]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)

        df = pd.read_parquet(target)
        row = df.iloc[0]
        # The product_id resolves the production edge's input, not the
        # activity's own (db, code).
        assert int(row["product_id"]) == ExchangeFrameBuilder.flow_id_for(("agb", "product-1"))
        # The catalog still lists the row by the activity's (db, code) —
        # that's how callers look it up.
        assert row["database"] == "agb"
        assert row["code"] == "process-1"

    def test_product_id_falls_back_when_production_unlinked(self, tmp_path: Path):
        # ``[Dummy]`` activities sometimes have an unlinked production
        # edge (input=None). Pre-drop_unlinked the row still exists; the
        # catalog should fall back to the activity's own (db, code) so
        # the lookup at least *resolves* (the entry will simply be
        # absent from the technosphere row map at scoring time, and
        # NativeLciaScorer reports "product id not in technosphere").
        sp_data = [
            {
                "database": "agb",
                "code": "dummy-1",
                "name": "[Dummy]",
                "type": "process",
                "unit": "kg",
                "exchanges": [
                    {"type": "production", "input": None, "amount": 0.0},
                ],
            },
        ]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)
        row = pd.read_parquet(target).iloc[0]
        assert int(row["product_id"]) == ExchangeFrameBuilder.flow_id_for(("agb", "dummy-1"))


class TestProductCatalogBuilderAtomicity:
    def test_no_partial_file_left_behind_on_success(self, tmp_path: Path):
        sp_data = [_activity("agb", "p1", name="wheat", type="process", unit="kg")]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)

        assert target.exists()
        # ``ParquetAtomicWriter`` writes to ``<path>.partial`` and renames;
        # nothing must survive at the partial path on a successful run.
        assert not target.with_suffix(target.suffix + ".partial").exists()

    def test_idempotent_rebuild(self, tmp_path: Path):
        # Re-running the builder over the same data must not raise and
        # must produce the same bytes.
        sp_data = [_activity("agb", "p1", name="wheat", type="process", unit="kg")]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)
        bytes_a = target.read_bytes()
        ProductCatalogBuilder().build(sp_data, target)
        bytes_b = target.read_bytes()
        assert bytes_a == bytes_b


class TestProductCatalogBuilderEdgeCases:
    def test_handles_missing_optional_fields(self, tmp_path: Path):
        # An activity without ``type``/``unit``/``name`` still writes a
        # row — we coerce missing strings to ``""`` rather than failing.
        sp_data = [{"database": "agb", "code": "minimal"}]
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build(sp_data, target)

        df = pd.read_parquet(target)
        row = df.iloc[0]
        assert row["database"] == "agb"
        assert row["code"] == "minimal"
        assert row["name"] == ""
        assert row["type"] == ""
        assert row["unit"] == ""
        assert int(row["product_id"]) == ExchangeFrameBuilder.flow_id_for(("agb", "minimal"))

    def test_empty_input_writes_empty_parquet(self, tmp_path: Path):
        target = tmp_path / "product_catalog.parquet"
        ProductCatalogBuilder().build([], target)
        df = pd.read_parquet(target)
        assert df.empty
        assert list(df.columns) == [
            "database",
            "code",
            "name",
            "type",
            "unit",
            "product_id",
        ]

    def test_duplicate_activities_deduped(self, tmp_path: Path):
        # Two activities with same (database, code) → catalog has one row.
        sp_data = [
            {"database": "agb", "code": "abc", "name": "act1", "type": "process", "unit": "kg"},
            {"database": "agb", "code": "abc", "name": "act1_dup", "type": "process", "unit": "kg"},
        ]
        path = tmp_path / "catalog.parquet"
        ProductCatalogBuilder().build(sp_data, path)
        df = pd.read_parquet(path)
        assert len(df) == 1
