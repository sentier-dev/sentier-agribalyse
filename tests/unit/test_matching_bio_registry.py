"""Unit tests for ``BiosphereRegistryBuilder`` and ``BiosphereCatalog``.

After REFACTOR_FINAL F6 the builder reads JSON snapshots from
``source/`` instead of bw2data. These tests cover both sides of the
round-trip plus the on-disk schema contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from domain import Bucket
from matching.bio_catalog import BiosphereCatalog
from matching.bio_registry import BiosphereRegistryBuilder


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_ef_flows_parquet(path: Path, rows: list[dict]) -> Path:
    cols = ["code", "name", "categories", "unit", "cas"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(by="code", kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression=None)
    return path


@pytest.fixture
def snapshots(settings):
    """Materialise the JSON snapshots + EF parquet the builder expects."""
    _write_json(
        settings.paths.biosphere3_flows_json,
        {
            "version": "3.9",
            "flows": [
                {
                    "code": "bio-co2",
                    "name": "Carbon dioxide",
                    "categories": ["air"],
                    "unit": "kilogram",
                    "cas": "124-38-9",
                    "synonyms": [],
                },
                {
                    "code": "bio-ch4",
                    "name": "Methane",
                    "categories": ["air", "urban"],
                    "unit": "kilogram",
                    "cas": None,
                    "synonyms": [],
                },
            ],
        },
    )
    _write_json(
        settings.paths.ecoinvent_biosphere_flows_json,
        {
            "version": "3.9.1",
            "db_name": settings.biosphere_db_name,
            "flows": [
                {
                    "code": "ec-no2",
                    "name": "Nitrogen dioxide",
                    "categories": ["air", "urban"],
                    "unit": "kilogram",
                    "cas": None,
                    "synonyms": [],
                }
            ],
        },
    )
    _write_ef_flows_parquet(
        settings.paths.registry_ef_flows,
        [
            {
                "code": "ef-water",
                "name": "Water",
                "categories": ["water"],
                "unit": "cubic meter",
                "cas": None,
            },
        ],
    )


class TestBiosphereRegistryBuilder:
    def test_build_writes_parquet_with_expected_schema(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        assert path == settings.paths.registry_biosphere_catalog
        assert path.exists()

        df = pd.read_parquet(path)
        expected_cols = {"database", "code", "name", "categories", "unit", "cas", "synonyms"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 4  # 2 biosphere3 + 1 ecoinvent-biosphere + 1 ef

    def test_build_sorts_rows_for_determinism(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        df = pd.read_parquet(path)
        keys = list(zip(df["database"].tolist(), df["code"].tolist(), strict=True))
        assert keys == sorted(keys), "rows must be sorted by (database, code)"

    def test_build_is_byte_equal_across_runs(self, settings, snapshots):
        path1 = BiosphereRegistryBuilder(settings=settings).build()
        bytes1 = path1.read_bytes()
        path2 = BiosphereRegistryBuilder(settings=settings).build()
        bytes2 = path2.read_bytes()
        assert bytes1 == bytes2

    def test_build_skips_missing_ecoinvent_snapshot(self, settings):
        """Builder must tolerate the optional ecoinvent-biosphere snapshot
        being absent; biosphere3 + EF still load."""
        _write_json(
            settings.paths.biosphere3_flows_json,
            {
                "flows": [
                    {
                        "code": "bio-co2",
                        "name": "Carbon dioxide",
                        "categories": ["air"],
                        "unit": "kilogram",
                        "cas": None,
                        "synonyms": [],
                    }
                ]
            },
        )
        _write_ef_flows_parquet(settings.paths.registry_ef_flows, [])
        path = BiosphereRegistryBuilder(settings=settings).build()
        df = pd.read_parquet(path)
        assert set(df["database"]) == {"biosphere3"}

    def test_build_raises_when_biosphere3_snapshot_missing(self, settings):
        _write_ef_flows_parquet(settings.paths.registry_ef_flows, [])
        with pytest.raises(FileNotFoundError, match="biosphere snapshot missing"):
            BiosphereRegistryBuilder(settings=settings).build()

    def test_build_raises_when_ef_parquet_missing(self, settings):
        _write_json(
            settings.paths.biosphere3_flows_json,
            {"flows": []},
        )
        with pytest.raises(FileNotFoundError, match="EF flows parquet"):
            BiosphereRegistryBuilder(settings=settings).build()

    def test_ef_rows_come_from_parquet_with_correct_db_tag(self, settings):
        _write_json(settings.paths.biosphere3_flows_json, {"flows": []})
        _write_ef_flows_parquet(
            settings.paths.registry_ef_flows,
            [
                {
                    "code": "ef-uuid-1",
                    "name": "FlowOne",
                    "categories": ["air"],
                    "unit": "kilogram",
                    "cas": None,
                }
            ],
        )
        path = BiosphereRegistryBuilder(settings=settings).build()
        df = pd.read_parquet(path)
        ef_rows = df[df["database"] == settings.ef_db_name]
        assert list(ef_rows["code"]) == ["ef-uuid-1"]
        assert list(ef_rows["name"]) == ["FlowOne"]

    def test_build_atomic_write_leaves_no_partial_file(self, settings, snapshots):
        BiosphereRegistryBuilder(settings=settings).build()
        partial = settings.paths.registry_biosphere_catalog.with_suffix(
            settings.paths.registry_biosphere_catalog.suffix + ".partial"
        )
        assert not partial.exists()

    def test_synonyms_column_is_empty_list_per_row(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        df = pd.read_parquet(path)
        for syn in df["synonyms"]:
            assert isinstance(syn, (list, tuple)) or hasattr(syn, "__len__")
            assert not isinstance(syn, str)
            assert list(syn) == []

    def test_categories_round_trip_as_list(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        df = pd.read_parquet(path)
        ch4 = df[df["code"] == "bio-ch4"].iloc[0]
        assert list(ch4["categories"]) == ["air", "urban"]


class TestBiosphereCatalogRoundTrip:
    def test_load_returns_same_set_of_refs_as_written(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        catalog = BiosphereCatalog.load(path)

        keys_loaded = {(f.db, f.code) for f in catalog.flows}
        keys_expected = {
            ("biosphere3", "bio-co2"),
            ("biosphere3", "bio-ch4"),
            ("ef", "ef-water"),
            ("ecoinvent-3.9.1-biosphere", "ec-no2"),
        }
        assert keys_loaded == keys_expected

    def test_load_preserves_cas_and_bucket(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        catalog = BiosphereCatalog.load(path)

        co2 = catalog.get("biosphere3", "bio-co2")
        assert co2 is not None
        assert co2.cas == "124-38-9"
        assert co2.bucket == Bucket.AIR
        assert co2.unit == "kilogram"

    def test_load_with_db_names_filter(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        catalog = BiosphereCatalog.load(path, db_names=("biosphere3",))
        assert set(catalog.db_names) == {"biosphere3"}
        assert all(f.db == "biosphere3" for f in catalog.flows)
        assert len(catalog.flows) == 2

    def test_load_without_filter_keeps_all_databases(self, settings, snapshots):
        path = BiosphereRegistryBuilder(settings=settings).build()
        catalog = BiosphereCatalog.load(path)
        assert set(catalog.db_names) == {"biosphere3", "ef", "ecoinvent-3.9.1-biosphere"}

    def test_load_missing_path_raises_clearly(self, tmp_path: Path):
        ghost = tmp_path / "ghost.parquet"
        with pytest.raises(FileNotFoundError, match="biosphere catalog parquet"):
            BiosphereCatalog.load(ghost)
