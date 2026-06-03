"""Unit tests for ``registry.sources`` — every source ingester is exercised
on tiny synthetic inputs (json, gz-json, xlsx, parquet) so we can flex
schema parsing without depending on the real production artifacts.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from core import ParquetCache
from domain import Bucket, SourceKind, Tier
from readers import XlsxReader
from registry.sources import (
    AGB32FlowmapperBiosphereSource,
    AgbDeleteAggregatedSource,
    AgbEdgeLabelsSource,
    CuratedOverridesSource,
    EfCfTargetIndexSource,
    HarmonisedFlowsSource,
    LlmReviewedSource,
    PlaceholderEcoinventSource,
    PlaceholderEfNativeSource,
    PlaceholderTransitiveSource,
    PlaceholderUnmatchableSource,
    RandonneurAgribalyseBiosphereSource,
    RandonneurSimaproBiosphereSource,
    RandonneurSimaproContextSource,
    RandonneurUnitAliasSource,
    RandonneurUnitConversionSource,
    RandonneurUnitNormalisationSource,
    RandonneurUnlinkedListSource,
    RandonneurWaterSlashM3Source,
)

# ============================================================================
# Helpers


class FakeLoader:
    """Stand-in for ``RandonneurDataLoader`` keyed by label → datapackage dict."""

    def __init__(self, packages: dict | None = None):
        self._packages = packages or {}

    def load(self, label):
        return self._packages.get(label, {})

    def has(self, label):
        return label in self._packages

    def labels(self):
        return list(self._packages)

    def metadata(self, label):
        return {"name": label}


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)


# ============================================================================
# Curated / LLM xlsx


class TestCuratedOverridesSource:
    def test_missing_file_returns_empty_list(self, tmp_path):
        src = CuratedOverridesSource(path=tmp_path / "missing.json")
        assert src.read() == []

    def test_default_tier_is_curated_targeted(self, tmp_path):
        path = tmp_path / "curated.json"
        path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "source_name": "1,2-Dichloropropane",
                            "source_unit": "kg",
                            "source_context": ["Emissions to water", "river"],
                            "target_db": "biosphere3",
                            "target_code": "uuid-1",
                            "target_name": "Propane, 1,2-dichloro-",
                            "target_unit": "kg",
                        }
                    ]
                }
            )
        )
        rows = CuratedOverridesSource(path=path).read()
        assert len(rows) == 1
        assert rows[0].priority_tier == Tier.CURATED_TARGETED
        assert rows[0].source_top_bucket == Bucket.WATER
        assert rows[0].target_code == "uuid-1"
        assert rows[0].source_kind == SourceKind.AGB_FLOW

    def test_unknown_tier_falls_back_to_curated_targeted(self, tmp_path):
        path = tmp_path / "curated.json"
        path.write_text(
            json.dumps({"entries": [{"source_name": "X", "tier": "DEFINITELY_NOT_A_TIER"}]})
        )
        rows = CuratedOverridesSource(path=path).read()
        assert rows[0].priority_tier == Tier.CURATED_TARGETED

    def test_explicit_synonym_tier_is_honored(self, tmp_path):
        path = tmp_path / "curated.json"
        path.write_text(
            json.dumps({"entries": [{"source_name": "X", "tier": "CURATED_SYNONYM_FALLBACK"}]})
        )
        rows = CuratedOverridesSource(path=path).read()
        assert rows[0].priority_tier == Tier.CURATED_SYNONYM_FALLBACK

    def test_select_unmatchable_separates_dead_ends_from_mappings(self, tmp_path):
        """The same JSON is read twice by the builder — with
        ``select_unmatchable=False`` for ``mappings_biosphere`` and with
        ``select_unmatchable=True`` for ``unmatchable``. Each instance must
        emit ONLY the rows matching its filter, never both."""
        path = tmp_path / "curated.json"
        path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "source_name": "Mappable thing",
                            "target_db": "biosphere3",
                            "target_name": "Replacement",
                            "tier": "CURATED_TARGETED",
                        },
                        {
                            "source_name": "Dead-end concept",
                            "is_unmatchable": True,
                            "notes": "no equivalent in any bio DB",
                        },
                    ]
                }
            )
        )
        # Default: only mappings (select_unmatchable=False)
        mapped = CuratedOverridesSource(path=path).read()
        assert [r.source_name for r in mapped] == ["Mappable thing"]
        assert mapped[0].is_unmatchable is False

        # Unmatchable view
        deads = CuratedOverridesSource(path=path, select_unmatchable=True).read()
        assert [r.source_name for r in deads] == ["Dead-end concept"]
        assert deads[0].is_unmatchable is True
        assert deads[0].priority_tier == Tier.UNMATCHABLE


class TestLlmReviewedSource:
    def test_missing_file_returns_empty_list(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        src = LlmReviewedSource(path=tmp_path / "missing.xlsx", cache=cache)
        assert src.read() == []

    def test_only_accepted_rows_are_emitted(self, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        path = tmp_path / "llm.xlsx"
        write_xlsx(
            path,
            {
                "Sheet1": pd.DataFrame(
                    [
                        {
                            "source_name": "Forever",
                            "source_unit": "kg",
                            "decision": "accept",
                            "target_name": "Replacement",
                            "target_top_cat": "air",
                            "target_sub_cat": "rural",
                            "multiplier": 1.0,
                        },
                        {
                            "source_name": "RejectMe",
                            "source_unit": "kg",
                            "decision": "reject",
                            "target_name": "Skipped",
                            "target_top_cat": "air",
                            "target_sub_cat": "rural",
                            "multiplier": 1.0,
                        },
                    ]
                )
            },
        )
        rows = LlmReviewedSource(path=path, cache=cache).read()
        accepted_names = {r.source_name for r in rows}
        # ``decision='reject'`` rows must never reach the registry, regardless of
        # whether their target_name happens to be populated.
        assert "RejectMe" not in accepted_names
        assert "Forever" in accepted_names
        forever = next(r for r in rows if r.source_name == "Forever")
        assert forever.priority_tier == Tier.LLM_OVERRIDES
        assert forever.target_db == "biosphere3"

    def test_source_top_bucket_is_unspecified_so_mapping_applies_to_any_compartment(self, tmp_path):
        """LLM rows are name synonyms — the same source name can appear in
        air, water, soil exchanges across AGB. Storing under UNSPECIFIED lets
        ``TieredNameBucketIndex.lookup`` surface the row for any actual
        bucket; ``BiosphereMatcher._resolve_target`` then locates the target
        in the exchange's actual compartment."""
        from domain import Bucket

        cache = ParquetCache(cache_dir=tmp_path / "cache")
        path = tmp_path / "llm.xlsx"
        write_xlsx(
            path,
            {
                "Sheet1": pd.DataFrame(
                    [
                        {
                            "source_name": "Alkenes, unspecified",
                            "source_unit": "kg",
                            "source_top_cat": "water",  # one observed bucket
                            "decision": "accept",
                            "target_name": "Hydrocarbons, aliphatic, unsaturated",
                            "target_top_cat": "water",  # bucket of the target the LLM picked
                            "multiplier": 1.0,
                        }
                    ]
                )
            },
        )
        rows = LlmReviewedSource(path=path, cache=cache).read()
        assert len(rows) == 1
        # Critical: the row must NOT be pinned to water bucket — otherwise an
        # AGB air-emission of "Alkenes, unspecified" would never find this
        # synonym (regression seen in the wild).
        assert rows[0].source_top_bucket == Bucket.UNSPECIFIED


# ============================================================================
# Placeholder workbook


class TestPlaceholderSources:
    @pytest.fixture
    def placeholder_xlsx(self, tmp_path: Path) -> Path:
        path = tmp_path / "placeholder.xlsx"
        write_xlsx(
            path,
            {
                "match with ecoinvent v3.9.1": pd.DataFrame(
                    [
                        {
                            "name": "Carbon dioxide",
                            "unit": "kg",
                            "categories": "('air',)",
                            "cas": "124-38-9",
                            "matched_ecoinvent_name": "Carbon dioxide, fossil",
                            "ecoinvent_match_type": "exact",
                        }
                    ]
                ),
                "match with EF v3.1": pd.DataFrame(
                    [
                        {
                            "name": "Methane",
                            "unit": "kg",
                            "categories": "('air',)",
                            "matched_ef_name": "Methane, fossil",
                        }
                    ]
                ),
                "Neither in ecoinvent nor EF": pd.DataFrame(
                    [
                        {
                            "name": "Mystery substance",
                            "unit": "kg",
                            "categories": "('air',)",
                        }
                    ]
                ),
                "ecoinvent flows - EF v3.1 map": pd.DataFrame(
                    [
                        {
                            "name": "Bridge",
                            "unit": "kg",
                            "categories": "('water', 'river')",
                            "matched_ef_name": "Bridge target",
                            "matched_ecoinvent_name": "via X",
                        }
                    ]
                ),
            },
        )
        return path

    def test_ecoinvent_source_emits_curated_tier_rows(self, placeholder_xlsx, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        xlsx = XlsxReader(cache=cache)
        rows = PlaceholderEcoinventSource(xlsx=xlsx, xlsx_path=placeholder_xlsx).read()
        assert len(rows) == 1
        assert rows[0].source_name == "Carbon dioxide"
        assert rows[0].priority_tier == Tier.CURATED_TARGETED
        assert rows[0].source_top_bucket == Bucket.AIR
        assert rows[0].source_cas == "124-38-9"

    def test_ef_native_source_emits_ef_placeholder_tier(self, placeholder_xlsx, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        xlsx = XlsxReader(cache=cache)
        rows = PlaceholderEfNativeSource(xlsx=xlsx, xlsx_path=placeholder_xlsx).read()
        assert len(rows) == 1
        assert rows[0].priority_tier == Tier.EF_PLACEHOLDER

    def test_unmatchable_source_marks_is_unmatchable(self, placeholder_xlsx, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        xlsx = XlsxReader(cache=cache)
        rows = PlaceholderUnmatchableSource(xlsx=xlsx, xlsx_path=placeholder_xlsx).read()
        assert len(rows) == 1
        assert rows[0].is_unmatchable is True
        assert rows[0].priority_tier == Tier.UNMATCHABLE

    def test_transitive_source_is_off_by_default(self, placeholder_xlsx, tmp_path):
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        xlsx = XlsxReader(cache=cache)
        # The transitive source still works on its own — the *gate* lives in
        # ``RegistryBuilder._build_sources``, not in the source itself.
        rows = PlaceholderTransitiveSource(xlsx=xlsx, xlsx_path=placeholder_xlsx).read()
        assert len(rows) == 1
        assert "transitive via via X" in rows[0].notes


# ============================================================================
# Harmonised flows


class TestHarmonisedFlowsSource:
    def test_missing_file_returns_empty(self, tmp_path):
        assert HarmonisedFlowsSource(path=tmp_path / "ghost.json.gz").read() == []

    def test_emits_one_row_per_alt_label_dedup_by_uuid(self, tmp_path):
        path = tmp_path / "harmonised.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(
                {
                    "flows": [
                        {
                            "source": "EF 3.1",
                            "identifier": "uuid-1",
                            "context_iri": "http://x.org/envi-air-rural",
                            "prefLabel": "Carbon dioxide",
                            "altLabel": ["CO2", "Carbon dioxide", "C O2"],
                        },
                        {
                            "source": "EF 3.1",
                            "identifier": "uuid-2",
                            "context_iri": "http://x.org/envi-wate-suwa",
                            "prefLabel": "Water",
                            "altLabel": [],
                        },
                        {
                            "source": "OTHER",
                            "identifier": "uuid-other",
                            "prefLabel": "skip me",
                        },
                    ]
                },
                f,
            )
        rows = HarmonisedFlowsSource(path=path).read()
        names = {r.source_name for r in rows}
        # CO2 + Carbon dioxide + C O2 + Water (one entry per altLabel + pref).
        assert "CO2" in names
        assert "Carbon dioxide" in names
        assert "Water" in names
        assert all(r.target_db == "ef" for r in rows)
        assert all(r.priority_tier == Tier.HARMONISED_FLOWS for r in rows)

    def test_buckets_derived_from_context_iri(self, tmp_path):
        path = tmp_path / "h.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(
                {
                    "flows": [
                        {
                            "source": "EF 3.1",
                            "identifier": "u1",
                            "context_iri": "http://x.org/reso-grou",
                            "prefLabel": "Mineral",
                            "altLabel": [],
                        }
                    ]
                },
                f,
            )
        rows = HarmonisedFlowsSource(path=path).read()
        assert rows[0].source_top_bucket == Bucket.RESOURCE

    def test_valid_uuids_filter_drops_flows_without_cf(self, tmp_path):
        path = tmp_path / "h.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(
                {
                    "flows": [
                        {
                            "source": "EF 3.1",
                            "identifier": "uuid-with-cf",
                            "context_iri": "http://x.org/envi-air-rural",
                            "prefLabel": "Carbon dioxide",
                            "altLabel": ["CO2"],
                        },
                        {
                            "source": "EF 3.1",
                            "identifier": "uuid-no-cf",
                            "context_iri": "http://x.org/envi-air-rural",
                            "prefLabel": "Phantom flow",
                            "altLabel": ["Phantom"],
                        },
                    ]
                },
                f,
            )
        rows = HarmonisedFlowsSource(path=path, valid_uuids=frozenset({"uuid-with-cf"})).read()
        assert {r.target_code for r in rows} == {"uuid-with-cf"}

    def test_valid_uuids_none_keeps_all_flows(self, tmp_path):
        path = tmp_path / "h.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(
                {
                    "flows": [
                        {
                            "source": "EF 3.1",
                            "identifier": "u1",
                            "context_iri": "http://x.org/envi-air-rural",
                            "prefLabel": "X",
                            "altLabel": [],
                        }
                    ]
                },
                f,
            )
        assert HarmonisedFlowsSource(path=path, valid_uuids=None).read()


# ============================================================================
# AGB JSON assets


class TestAGB32FlowmapperBiosphereSource:
    def test_missing_file_returns_empty_list(self, tmp_path):
        src = AGB32FlowmapperBiosphereSource(path=tmp_path / "missing.json")
        assert src.read() == []

    def test_reads_update_entries(self, tmp_path):
        path = tmp_path / "bio.json"
        path.write_text(
            json.dumps(
                {
                    "update": [
                        {
                            "source": {
                                "name": "Methane",
                                "unit": "kilogram",
                                "context": ["Emissions to air"],
                            },
                            "target": {
                                "name": "Methane, fossil",
                                "identifier": "abc123",
                                "unit": "kilogram",
                            },
                            "conversion_factor": 1.0,
                            "comment": "direct match",
                        }
                    ]
                }
            )
        )
        rows = AGB32FlowmapperBiosphereSource(path=path).read()
        assert len(rows) == 1
        r = rows[0]
        assert r.source_name == "Methane"
        assert r.source_unit == "kilogram"
        assert r.source_top_bucket == Bucket.AIR
        assert r.target_name == "Methane, fossil"
        assert r.target_code == "abc123"
        assert r.priority_tier == Tier.AGRIBALYSE_EI_BIOSPHERE
        assert r.provenance == "agribalyse.ecoinvent-3.10-biosphere-flowmapper"

    def test_skips_entries_with_no_target_code_or_name(self, tmp_path):
        path = tmp_path / "bio.json"
        path.write_text(
            json.dumps(
                {
                    "update": [
                        {"source": {"name": "X", "unit": "kg"}, "target": {}},
                        {
                            "source": {"name": "Y", "unit": "kg"},
                            "target": {"name": "Target Y"},
                        },
                    ]
                }
            )
        )
        rows = AGB32FlowmapperBiosphereSource(path=path).read()
        assert len(rows) == 1
        assert rows[0].source_name == "Y"

    def test_conversion_factor_stored_as_unit_conversion(self, tmp_path):
        path = tmp_path / "bio.json"
        path.write_text(
            json.dumps(
                {
                    "update": [
                        {
                            "source": {"name": "Cs-134", "unit": "Becquerel"},
                            "target": {"name": "Caesium-134", "identifier": "c1"},
                            "conversion_factor": 0.001,
                        }
                    ]
                }
            )
        )
        rows = AGB32FlowmapperBiosphereSource(path=path).read()
        assert rows[0].unit_conversion == pytest.approx(0.001)


class TestAgbDeleteAggregatedSource:
    def test_missing_files_yield_empty_list(self, tmp_path):
        src = AgbDeleteAggregatedSource(
            processes_path=tmp_path / "missing-1.json",
            products_path=tmp_path / "missing-2.json",
        )
        assert src.read() == []

    def test_processes_and_products_get_merged_with_distinct_kinds(self, tmp_path):
        proc_path = tmp_path / "proc.json"
        prod_path = tmp_path / "prod.json"
        proc_path.write_text(
            json.dumps(
                {
                    "delete": [
                        {"source": {"name": "agg-X", "identifier": "code-X"}},
                        {"source": {"name": "agg-Y"}},
                    ]
                }
            )
        )
        prod_path.write_text(
            json.dumps({"delete": [{"source": {"name": "prod-Z", "identifier": "code-Z"}}]})
        )
        rows = AgbDeleteAggregatedSource(processes_path=proc_path, products_path=prod_path).read()
        kinds = {r.kind for r in rows}
        assert kinds == {"process", "product"}
        assert any(r.code == "code-X" for r in rows)
        assert any(r.name == "prod-Z" for r in rows)


class TestAgbEdgeLabelsSource:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert AgbEdgeLabelsSource(path=tmp_path / "missing.json").read() == []

    def test_only_complete_rows_are_emitted(self, tmp_path):
        path = tmp_path / "edges.json"
        path.write_text(
            json.dumps(
                {
                    "replace": [
                        {
                            "source": {"name": "old", "type": "technosphere"},
                            "target": {"name": "new"},
                        },
                        {"source": {"name": ""}, "target": {"name": "skip"}},
                        {"source": {"name": "x"}, "target": {"name": ""}},
                    ]
                }
            )
        )
        rows = AgbEdgeLabelsSource(path=path).read()
        assert len(rows) == 1
        assert rows[0].source_name == "old"
        assert rows[0].target_name == "new"
        assert rows[0].edge_type == "technosphere"


# ============================================================================
# Randonneur datapackage sources


class TestRandonneurAgribalyseBiosphereSource:
    def test_emits_tier_two_rows_from_replace_block(self):
        loader = FakeLoader(
            {
                "agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches": {
                    "replace": [
                        {
                            "source": {
                                "name": "Foo",
                                "unit": "kg",
                                "context": ["air", "rural"],
                            },
                            "target": {
                                "identifier": "uuid-foo",
                                "name": "Foo target",
                                "unit": "kg",
                            },
                            "conversion_factor": 1.0,
                            "comment": "linker note",
                        }
                    ]
                }
            }
        )
        rows = RandonneurAgribalyseBiosphereSource(loader=loader).read()
        assert len(rows) == 1
        assert rows[0].priority_tier == Tier.RANDONNEUR_AGB_SPECIFIC
        assert rows[0].target_code == "uuid-foo"
        assert rows[0].source_top_bucket == Bucket.AIR
        assert rows[0].notes == "linker note"


class TestRandonneurSimaproBiosphereSource:
    def test_emits_tier_three_rows(self):
        loader = FakeLoader(
            {
                "SimaPro-9-ecoinvent-3.9-biosphere-manual-matches": {
                    "replace": [
                        {
                            "source": {"name": "X", "unit": "kg", "context": ["water"]},
                            "target": {"identifier": "u-x", "name": "X t", "unit": "kg"},
                            "conversion_factor": 0.5,
                        }
                    ]
                }
            }
        )
        rows = RandonneurSimaproBiosphereSource(loader=loader).read()
        assert rows[0].priority_tier == Tier.RANDONNEUR_SIMAPRO_BIO
        assert rows[0].unit_conversion == 0.5
        assert rows[0].source_top_bucket == Bucket.WATER


class TestRandonneurWaterSlashM3Source:
    def test_emits_tier_five_rows(self):
        loader = FakeLoader(
            {
                "simapro-9-ecoinvent-3-water-slash-m3": {
                    "replace": [
                        {
                            "source": {"name": "Water", "unit": "kg"},
                            "target": {
                                "identifier": "u-w",
                                "name": "Water",
                                "unit": "m3",
                            },
                            "conversion_factor": 0.001,
                            "location": "GLO",
                        }
                    ]
                }
            }
        )
        rows = RandonneurWaterSlashM3Source(loader=loader).read()
        assert rows[0].priority_tier == Tier.RANDONNEUR_WATER_M3
        assert rows[0].unit_conversion == 0.001


class TestRandonneurUnitConversionSource:
    def test_only_complete_rows_are_emitted(self):
        loader = FakeLoader(
            {
                "generic-brightway-unit-conversions": {
                    "replace": [
                        {
                            "source": {"unit": "m"},
                            "target": {"unit": "km", "allocation": 0.001},
                        },
                        {
                            "source": {"unit": "kg"},
                            "target": {"unit": "lb"},  # no allocation -> skipped
                        },
                        {
                            "source": {"unit": ""},  # missing src unit -> skipped
                            "target": {"unit": "x", "allocation": 1.0},
                        },
                    ]
                }
            }
        )
        rows = RandonneurUnitConversionSource(loader=loader).read()
        assert len(rows) == 1
        assert rows[0].source_unit == "m"
        assert rows[0].target_unit == "km"
        assert rows[0].multiplier == 0.001


class TestRandonneurUnitAliasSources:
    def test_alias_source_reads_update_block(self):
        loader = FakeLoader(
            {
                "Flowmapper-standard-units-harmonization": {
                    "update": [
                        {"source": {"unit": "a"}, "target": {"unit": "year"}},
                        {"source": {"unit": ""}, "target": {"unit": "year"}},
                    ]
                }
            }
        )
        rows = RandonneurUnitAliasSource(loader=loader).read()
        assert len(rows) == 1
        assert rows[0].alias == "a"
        assert rows[0].canonical == "year"

    def test_normalisation_source_reads_replace_block(self):
        loader = FakeLoader(
            {
                "generic-brightway-units-normalization": {
                    "replace": [{"source": {"unit": "kg/kg"}, "target": {"unit": "ratio"}}]
                }
            }
        )
        rows = RandonneurUnitNormalisationSource(loader=loader).read()
        assert rows[0].alias == "kg/kg"
        assert rows[0].canonical == "ratio"


class TestRandonneurSimaproContextSource:
    def test_emits_one_context_norm_per_replace_entry(self):
        loader = FakeLoader(
            {
                "simapro-9-ecoinvent-3-context": {
                    "replace": [
                        {
                            "source": {"context": ["Emissions to air", "high pop."]},
                            "target": {"context": ["air", "urban"]},
                        }
                    ]
                }
            }
        )
        rows = RandonneurSimaproContextSource(loader=loader).read()
        assert rows[0].source_context == ("Emissions to air", "high pop.")
        assert rows[0].target_context == ("air", "urban")


class TestRandonneurUnlinkedListSource:
    def test_returns_empty_when_label_missing(self):
        loader = FakeLoader({})
        assert RandonneurUnlinkedListSource(loader=loader).read() == []

    def test_marks_rows_as_unmatchable(self):
        loader = FakeLoader(
            {
                "agribalyse-3.1.1-biosphere-ecoinvent-3.8-biosphere": {
                    "update": [{"source": {"name": "X", "unit": "kg", "context": ["air"]}}]
                }
            }
        )
        rows = RandonneurUnlinkedListSource(loader=loader).read()
        assert rows[0].is_unmatchable is True
        assert rows[0].priority_tier == Tier.UNMATCHABLE


# ============================================================================
# EF target index source


class TestEfCfTargetIndexSource:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert EfCfTargetIndexSource(path=tmp_path / "ghost.parquet").read() == []

    def test_emits_one_row_per_uuid(self, tmp_path):
        path = tmp_path / "cf.parquet"
        df = pd.DataFrame(
            [
                {
                    "FLOW_uuid": "uuid-1",
                    "FLOW_name": "Carbon dioxide, fossil",
                    "FLOW_class0": "Emissions",
                    "FLOW_class1": "air",
                    "FLOW_class2": None,
                    "LCIAMethod_name": "Climate change",
                    "LCIAMethod_location": None,
                    "CF EF3.1": 1.0,
                },
                {
                    "FLOW_uuid": "uuid-1",  # duplicate UUID — should collapse
                    "FLOW_name": "Carbon dioxide, fossil",
                    "FLOW_class0": "Emissions",
                    "FLOW_class1": "air",
                    "FLOW_class2": None,
                    "LCIAMethod_name": "Climate change",
                    "LCIAMethod_location": "FR",
                    "CF EF3.1": 1.05,
                },
                {
                    "FLOW_uuid": "uuid-2",
                    "FLOW_name": "Methane, fossil",
                    "FLOW_class0": "Emissions",
                    "FLOW_class1": "air",
                    "FLOW_class2": None,
                    "LCIAMethod_name": "Climate change",
                    "LCIAMethod_location": None,
                    "CF EF3.1": 27.0,
                },
            ]
        )
        df.to_parquet(path, index=False)
        rows = EfCfTargetIndexSource(path=path).read()
        codes = sorted(r.code for r in rows)
        assert codes == ["uuid-1", "uuid-2"]
        assert all(r.unit == "kilogram" for r in rows)
