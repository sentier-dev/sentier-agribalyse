"""Unit tests for ``registry.MappingRegistry`` — load + lazy index access."""

from __future__ import annotations

import dataclasses
import json

import pytest

from domain import Bucket, Tier
from registry import MappingRegistry
from registry.indexes import (
    CasIndex,
    TieredNameBucketIndex,
    UnitConverter,
    UnmatchableIndex,
)
from tests.fixtures.builders import (
    make_mappings_biosphere_df,
    make_registry,
    make_unit_aliases_df,
    make_unit_conversions_df,
)


class TestMappingRegistryConstruction:
    def test_make_registry_helper_yields_empty_dataframes(self, settings):
        reg = make_registry(settings)
        assert reg.total_biosphere_mappings == 0
        assert reg.total_technosphere_mappings == 0

    def test_total_counts_reflect_dataframe_lengths(self, settings):
        df = make_mappings_biosphere_df(
            [{"source_name": "X"}, {"source_name": "Y"}, {"source_name": "Z"}]
        )
        reg = make_registry(settings, mappings_biosphere=df)
        assert reg.total_biosphere_mappings == 3


class TestLazyIndexes:
    def test_biosphere_index_caches_instance(self, settings):
        reg = make_registry(settings)
        idx1 = reg.biosphere_index
        idx2 = reg.biosphere_index
        assert idx1 is idx2  # cached_property

    def test_each_index_is_typed(self, settings):
        df = make_mappings_biosphere_df([{"source_name": "X"}])
        reg = make_registry(settings, mappings_biosphere=df)
        assert isinstance(reg.biosphere_index, TieredNameBucketIndex)
        assert isinstance(reg.technosphere_index, TieredNameBucketIndex)
        assert isinstance(reg.biosphere_cas_index, CasIndex)
        assert isinstance(reg.unit_converter, UnitConverter)
        assert isinstance(reg.unmatchable_index, UnmatchableIndex)

    def test_unit_converter_uses_both_dataframes(self, settings):
        reg = make_registry(
            settings,
            unit_conversions=make_unit_conversions_df(
                [{"source_unit": "m", "target_unit": "km", "multiplier": 0.001}]
            ),
            unit_aliases=make_unit_aliases_df([{"alias": "a", "canonical": "year"}]),
        )
        assert reg.unit_converter.canonical("a") == "year"
        assert reg.unit_converter.multiplier("m", "km") == 0.001


class TestLoadFromDisk:
    def test_load_returns_empty_dataframes_when_no_files_exist(self, settings):
        reg = MappingRegistry.load(settings)
        assert reg.mappings_biosphere.empty
        assert reg.mappings_technosphere.empty
        assert reg.meta == {}

    def test_load_reads_meta_json_when_present(self, settings):
        settings.paths.registry.mkdir(parents=True, exist_ok=True)
        settings.paths.registry_meta.write_text(
            json.dumps({"built_at": "2026-04-28T00:00:00Z", "row_counts": {"x": 1}})
        )
        reg = MappingRegistry.load(settings)
        assert reg.meta["built_at"].startswith("2026")
        assert reg.meta["row_counts"]["x"] == 1

    def test_load_reads_existing_parquets(self, settings):
        settings.paths.registry.mkdir(parents=True, exist_ok=True)
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "CO2",
                    "source_top_bucket": Bucket.AIR,
                    "priority_tier": Tier.CURATED_TARGETED,
                    "target_db": "biosphere3",
                    "target_code": "abc",
                }
            ]
        )
        df.to_parquet(settings.paths.registry_mappings_biosphere, index=False)

        reg = MappingRegistry.load(settings)
        assert reg.total_biosphere_mappings == 1
        hits = reg.biosphere_index.lookup("agb_flow", "co2", Bucket.AIR)
        assert len(hits) == 1
        assert hits[0].target_code == "abc"

    def test_loaded_indexes_match_passed_dataframes(self, settings):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "CO2",
                    "source_cas": "124-38-9",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "abc",
                }
            ]
        )
        reg = make_registry(settings, mappings_biosphere=df)
        cas_hits = reg.biosphere_cas_index.lookup("124-38-9")
        assert cas_hits[0].target_code == "abc"


class TestRegistryImmutability:
    def test_meta_assignment_is_blocked(self, settings):
        reg = make_registry(settings, meta={"a": 1})
        with pytest.raises(dataclasses.FrozenInstanceError):
            reg.meta = {"b": 2}  # type: ignore[misc]
