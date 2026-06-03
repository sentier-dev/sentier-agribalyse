"""Unit tests for the ``ef/`` layer.

After REFACTOR_FINAL F6 the builders read from JSON snapshots in
``source/`` instead of bw2data. Coverage:

- ``EfCfTable`` (the parquet-backed source-CSV wrapper).
- ``EfFlowsRegistryBuilder`` (emits ``registry/ef_flows.parquet``).
- ``MethodCfRegistryBuilder`` (emits per-method CF parquets).
- ``MethodCfRegistryLoader`` (rehydrates them).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ef import (
    EfCfTable,
    EfFlowsRegistryBuilder,
    MethodCfRegistryBuilder,
    MethodCfRegistryLoader,
)
from scoring.method_slug import MethodSlug
from tests.fixtures.builders import (
    make_mappings_biosphere_df,
    make_registry,
)

# ============================================================================
# Helpers


def _cf_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "FLOW_uuid": "uuid-1",
                "FLOW_name": "CO2",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "air",
                "FLOW_class2": None,
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": None,
                "LCIAMethod_direction": "input",
                "CF EF3.1": 1.0,
            },
            {
                "FLOW_uuid": "uuid-1",
                "FLOW_name": "CO2",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "air",
                "FLOW_class2": None,
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": "FR",
                "LCIAMethod_direction": "input",
                "CF EF3.1": 2.0,
            },
            {
                "FLOW_uuid": "uuid-2",
                "FLOW_name": "Methane",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "air",
                "FLOW_class2": None,
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": "FR",
                "LCIAMethod_direction": "input",
                "CF EF3.1": 25.0,
            },
            {
                "FLOW_uuid": "uuid-2",
                "FLOW_name": "Methane",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "air",
                "FLOW_class2": None,
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": "DE",
                "LCIAMethod_direction": "input",
                "CF EF3.1": 27.0,
            },
        ]
    )


def _ef_target_index_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["code", "name", "name_lower", "unit", "bucket", "categories", "cas"]
    return pd.DataFrame(
        [
            {
                "code": r["code"],
                "name": r.get("name", "Flow " + r["code"]),
                "name_lower": r.get("name", "Flow " + r["code"]).lower(),
                "unit": r.get("unit", "kilogram"),
                "bucket": r.get("bucket", "air"),
                "categories": r.get("categories", []),
                "cas": r.get("cas"),
            }
            for r in rows
        ],
        columns=cols,
    )


def _write_methods_snapshot(settings, methods: dict[str, dict]) -> Path:
    """Materialise ``source/ef-v31-methods.json`` for the F6 builders."""
    path = settings.paths.ef_methods_snapshot_json
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "EF v3.1",
        "ef_db_name": settings.ef_db_name,
        "methods": methods,
    }
    path.write_text(json.dumps(payload))
    return path


def _ef_method_entry(key: tuple, *, inherited: list[dict] | None = None) -> dict:
    return {
        "key": list(key),
        "inherited_cfs": list(inherited or []),
    }


# ============================================================================
# EfCfTable


class TestEfCfTable:
    def test_global_cfs_prefers_null_location(self, tmp_path):
        path = tmp_path / "ef.parquet"
        _cf_dataframe().to_parquet(path, index=False)
        global_cfs = EfCfTable(path=path).global_cfs
        c1 = global_cfs[global_cfs["FLOW_uuid"] == "uuid-1"]
        assert c1["CF_global"].tolist() == [1.0]

    def test_global_cfs_falls_back_to_regional_mean(self, tmp_path):
        path = tmp_path / "ef.parquet"
        _cf_dataframe().to_parquet(path, index=False)
        global_cfs = EfCfTable(path=path).global_cfs
        c2 = global_cfs[global_cfs["FLOW_uuid"] == "uuid-2"]
        assert c2["CF_global"].tolist() == [pytest.approx(26.0)]


# ============================================================================
# EfFlowsRegistryBuilder


class TestEfFlowsRegistryBuilder:
    def test_no_target_index_writes_empty_parquet(self, settings):
        registry = make_registry(settings)
        path = EfFlowsRegistryBuilder(settings=settings, registry=registry).build()
        assert path == settings.paths.registry_ef_flows
        assert path.exists()
        df = pd.read_parquet(path)
        assert list(df.columns) == ["code", "name", "categories", "unit", "cas"]
        assert len(df) == 0

    def test_filters_to_uuids_referenced_by_registry_mappings(self, settings):
        target_index = _ef_target_index_df(
            [
                {"code": "uuid-keep", "name": "CO2", "categories": ["Emissions", "air"]},
                {"code": "uuid-drop", "name": "Other", "categories": []},
            ]
        )
        bio = make_mappings_biosphere_df(
            [
                {
                    "source_name": "CO2",
                    "target_db": settings.ef_db_name,
                    "target_code": "uuid-keep",
                }
            ]
        )
        registry = make_registry(settings, mappings_biosphere=bio, target_index_ef=target_index)
        path = EfFlowsRegistryBuilder(settings=settings, registry=registry).build()
        df = pd.read_parquet(path)
        assert list(df["code"]) == ["uuid-keep"]
        kept = df.iloc[0]
        assert list(kept["categories"]) == ["Emissions", "air"]
        assert kept["unit"] == "kilogram"

    def test_filter_picks_up_uuids_from_methods_snapshot(self, settings):
        target_index = _ef_target_index_df(
            [
                {"code": "uuid-from-method", "name": "Bonus", "categories": []},
                {"code": "uuid-untouched", "name": "Skipped", "categories": []},
            ]
        )
        registry = make_registry(settings, target_index_ef=target_index)

        m_key = ("EF v3.1 no LT", "EF v3.1", "ozone depletion", "ozone depletion potential (ODP)")
        _write_methods_snapshot(
            settings,
            {
                "ozone-depletion": _ef_method_entry(
                    m_key,
                    inherited=[
                        {"db": settings.ef_db_name, "code": "uuid-from-method", "amount": 0.5}
                    ],
                )
            },
        )

        path = EfFlowsRegistryBuilder(settings=settings, registry=registry).build()
        df = pd.read_parquet(path)
        codes = set(df["code"].tolist())
        assert codes == {"uuid-from-method"}

    def test_rows_sorted_by_code(self, settings):
        target_index = _ef_target_index_df([{"code": "zzz"}, {"code": "aaa"}, {"code": "mmm"}])
        bio = make_mappings_biosphere_df(
            [
                {"source_name": s, "target_db": settings.ef_db_name, "target_code": code}
                for s, code in (("a", "zzz"), ("b", "aaa"), ("c", "mmm"))
            ]
        )
        registry = make_registry(settings, mappings_biosphere=bio, target_index_ef=target_index)
        path = EfFlowsRegistryBuilder(settings=settings, registry=registry).build()
        df = pd.read_parquet(path)
        assert df["code"].tolist() == ["aaa", "mmm", "zzz"]

    def test_atomic_write_leaves_no_partial_file(self, settings):
        registry = make_registry(settings)
        EfFlowsRegistryBuilder(settings=settings, registry=registry).build()
        partial = settings.paths.registry_ef_flows.with_suffix(
            settings.paths.registry_ef_flows.suffix + ".partial"
        )
        assert not partial.exists()

    def test_byte_equal_across_runs(self, settings):
        target_index = _ef_target_index_df([{"code": "u1"}, {"code": "u2"}])
        bio = make_mappings_biosphere_df(
            [
                {"source_name": "a", "target_db": settings.ef_db_name, "target_code": "u1"},
                {"source_name": "b", "target_db": settings.ef_db_name, "target_code": "u2"},
            ]
        )
        registry = make_registry(settings, mappings_biosphere=bio, target_index_ef=target_index)
        b = EfFlowsRegistryBuilder(settings=settings, registry=registry)
        b1 = b.build().read_bytes()
        b2 = b.build().read_bytes()
        assert b1 == b2


# ============================================================================
# MethodCfRegistryBuilder + MethodCfRegistryLoader


class TestMethodCfRegistryBuilder:
    @pytest.fixture
    def cf_table(self, tmp_path):
        path = tmp_path / "cf.parquet"
        _cf_dataframe().to_parquet(path, index=False)
        return EfCfTable(path=path)

    def test_no_methods_writes_empty_index(self, settings, cf_table):
        _write_methods_snapshot(settings, {})
        out_dir = MethodCfRegistryBuilder(settings=settings, cf_table=cf_table).build()
        assert out_dir == settings.paths.registry_method_cfs_dir
        index = json.loads(settings.paths.registry_method_cfs_index.read_text())
        assert index["methods"] == []
        assert index["version"] == 1

    def test_writes_one_parquet_per_mapped_method(self, settings, cf_table):
        mapped_key = (
            "EF v3.1 no LT",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        unmapped_key = ("EF v3.1 no LT", "EF v3.1", "definitely not in the map", "x")
        _write_methods_snapshot(
            settings,
            {
                "climate-change": _ef_method_entry(mapped_key),
                "definitely-not": _ef_method_entry(unmapped_key),
            },
        )

        MethodCfRegistryBuilder(settings=settings, cf_table=cf_table).build()
        index = json.loads(settings.paths.registry_method_cfs_index.read_text())
        slugs = [e["slug"] for e in index["methods"]]
        assert slugs == [MethodSlug.encode(mapped_key)]

        cf_path = settings.paths.registry_method_cfs_dir / slugs[0] / "cfs.parquet"
        assert cf_path.exists()
        df = pd.read_parquet(cf_path)
        assert set(df.columns) == {"database", "code", "amount"}
        assert set(df["code"]) == {"uuid-1", "uuid-2"}
        assert set(df["database"]) == {settings.ef_db_name}

    def test_inherits_non_ef_cfs_from_snapshot(self, settings, cf_table):
        mapped_key = (
            "EF v3.1 no LT",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        _write_methods_snapshot(
            settings,
            {
                "climate-change": _ef_method_entry(
                    mapped_key,
                    inherited=[{"db": "biosphere3", "code": "bio-co2", "amount": 1.5}],
                )
            },
        )

        MethodCfRegistryBuilder(settings=settings, cf_table=cf_table).build()
        slug = MethodSlug.encode(mapped_key)
        df = pd.read_parquet(settings.paths.registry_method_cfs_dir / slug / "cfs.parquet")
        bio_rows = df[df["database"] == "biosphere3"]
        assert list(bio_rows["code"]) == ["bio-co2"]
        assert bio_rows["amount"].tolist() == [pytest.approx(1.5)]

    def test_cfs_are_sorted_by_database_then_code(self, settings, cf_table):
        mapped_key = (
            "EF v3.1 no LT",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        _write_methods_snapshot(
            settings,
            {
                "climate-change": _ef_method_entry(
                    mapped_key,
                    inherited=[{"db": "biosphere3", "code": "z-bio", "amount": 1.0}],
                )
            },
        )

        MethodCfRegistryBuilder(settings=settings, cf_table=cf_table).build()
        slug = MethodSlug.encode(mapped_key)
        df = pd.read_parquet(settings.paths.registry_method_cfs_dir / slug / "cfs.parquet")
        keys = list(zip(df["database"].tolist(), df["code"].tolist(), strict=True))
        assert keys == sorted(keys)

    def test_atomic_write_leaves_no_partial_files(self, settings, cf_table):
        mapped_key = (
            "EF v3.1 no LT",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        _write_methods_snapshot(settings, {"x": _ef_method_entry(mapped_key)})

        MethodCfRegistryBuilder(settings=settings, cf_table=cf_table).build()
        leftovers = list(settings.paths.registry_method_cfs_dir.rglob("*.partial"))
        assert leftovers == []

    def test_water_resource_augmenter_adds_rows_to_water_use_only(self, settings, cf_table):
        """When ``water_resource_augmenter`` is wired in, its emitted CF
        rows land in the water-use parquet (and only that one). Other
        methods are unchanged.
        """
        from ef.water_resource_augmenter import WaterResourceCfAugmenter

        water_key = (
            settings.ef_db_name,
            "EF v3.1",
            "water use",
            "user deprivation potential (deprivation-weighted water consumption)",
        )
        cc_key = (
            settings.ef_db_name,
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        _write_methods_snapshot(
            settings,
            {
                "water-use": _ef_method_entry(water_key),
                "climate-change": _ef_method_entry(cc_key),
            },
        )

        class _FakeAugmenter(WaterResourceCfAugmenter):
            def __init__(self):
                pass

            def cf_rows(self):
                return [
                    {
                        "database": "biosphere3",
                        "code": "river-uuid",
                        "amount": 42.95,
                    },
                ]

        builder = MethodCfRegistryBuilder(
            settings=settings,
            cf_table=cf_table,
            water_resource_augmenter=_FakeAugmenter(),
        )
        builder.build()

        water_slug = MethodSlug.encode(water_key)
        cc_slug = MethodSlug.encode(cc_key)
        water_df = pd.read_parquet(
            settings.paths.registry_method_cfs_dir / water_slug / "cfs.parquet"
        )
        cc_df = pd.read_parquet(settings.paths.registry_method_cfs_dir / cc_slug / "cfs.parquet")

        # Augmented row appears in water-use parquet.
        water_bio = water_df[water_df["database"] == "biosphere3"]
        assert "river-uuid" in water_bio["code"].tolist()
        assert water_bio[water_bio["code"] == "river-uuid"]["amount"].iloc[0] == pytest.approx(
            42.95
        )
        # Climate-change parquet untouched.
        assert "river-uuid" not in cc_df["code"].tolist()

    def test_water_resource_augmenter_does_not_overwrite_inherited(self, settings, cf_table):
        """If the bw2io snapshot already inherits a CF on a (db, code) the
        augmenter would emit, the inherited value wins — augmentation
        only fills GAPS, never replaces existing entries.
        """
        from ef.water_resource_augmenter import WaterResourceCfAugmenter

        water_key = (
            settings.ef_db_name,
            "EF v3.1",
            "water use",
            "user deprivation potential (deprivation-weighted water consumption)",
        )
        _write_methods_snapshot(
            settings,
            {
                "water-use": _ef_method_entry(
                    water_key,
                    inherited=[
                        {"db": "biosphere3", "code": "river-uuid", "amount": 6.98},
                    ],
                ),
            },
        )

        class _FakeAugmenter(WaterResourceCfAugmenter):
            def __init__(self):
                pass

            def cf_rows(self):
                return [
                    {
                        "database": "biosphere3",
                        "code": "river-uuid",
                        "amount": 42.95,
                    },
                ]

        MethodCfRegistryBuilder(
            settings=settings,
            cf_table=cf_table,
            water_resource_augmenter=_FakeAugmenter(),
        ).build()

        slug = MethodSlug.encode(water_key)
        df = pd.read_parquet(settings.paths.registry_method_cfs_dir / slug / "cfs.parquet")
        bio = df[df["database"] == "biosphere3"]
        # Single row — inherited value preserved, augmenter row dropped
        # to avoid double-count.
        assert list(bio["code"]) == ["river-uuid"]
        assert bio["amount"].tolist() == [pytest.approx(6.98)]

    def test_skips_inherited_entry_with_invalid_amount(self, settings, cf_table):
        """Garbage-in-snapshot CF amounts are dropped, not raised."""
        mapped_key = (
            "EF v3.1 no LT",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        _write_methods_snapshot(
            settings,
            {
                "climate-change": _ef_method_entry(
                    mapped_key,
                    inherited=[
                        {"db": "biosphere3", "code": "good", "amount": 1.0},
                        {"db": "biosphere3", "code": "bad", "amount": "not-a-number"},
                    ],
                )
            },
        )
        MethodCfRegistryBuilder(settings=settings, cf_table=cf_table).build()
        slug = MethodSlug.encode(mapped_key)
        df = pd.read_parquet(settings.paths.registry_method_cfs_dir / slug / "cfs.parquet")
        bio = df[df["database"] == "biosphere3"]
        assert list(bio["code"]) == ["good"]


class TestMethodCfRegistryLoader:
    def test_load_all_round_trips_what_builder_wrote(self, settings, tmp_path):
        cf_path = tmp_path / "cf.parquet"
        _cf_dataframe().to_parquet(cf_path, index=False)
        cf_table = EfCfTable(path=cf_path)

        m_a = (
            "EF v3.1 no LT",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        m_b = (
            "EF v3.1 no LT",
            "EF v3.1",
            "ozone depletion",
            "ozone depletion potential (ODP)",
        )
        _write_methods_snapshot(
            settings,
            {
                "climate-change": _ef_method_entry(m_a),
                "ozone-depletion": _ef_method_entry(m_b),
            },
        )

        MethodCfRegistryBuilder(settings=settings, cf_table=cf_table).build()
        loaded = MethodCfRegistryLoader(
            registry_dir=settings.paths.registry_method_cfs_dir
        ).load_all()
        assert m_a in loaded
        assert set(loaded[m_a].columns) == {"database", "code", "amount"}
        assert len(loaded[m_a]) > 0

    def test_load_all_raises_when_index_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="method-CFs index"):
            MethodCfRegistryLoader(registry_dir=tmp_path).load_all()
