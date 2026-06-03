"""Unit tests for ``registry.indexes`` — TieredNameBucketIndex, CasIndex,
UnitConverter, UnmatchableIndex.
"""

from __future__ import annotations

import pandas as pd
import pytest

from domain import Bucket, Tier
from registry.indexes import (
    CasIndex,
    TieredNameBucketIndex,
    UnitConverter,
    UnmatchableIndex,
)
from tests.fixtures.builders import (
    make_mappings_biosphere_df,
    make_unit_aliases_df,
    make_unit_conversions_df,
    make_unmatchable_df,
)


class TestTieredNameBucketIndex:
    def test_empty_df_returns_no_hits(self):
        idx = TieredNameBucketIndex(df=pd.DataFrame())
        assert idx.lookup("agb_flow", "co2", Bucket.AIR) == []

    def test_lookup_returns_candidate_rows_sorted_by_tier(self):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "CO2",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "ef",
                    "target_code": "ef-low-tier",
                    "priority_tier": Tier.HARMONISED_FLOWS,
                    "provenance": "p2",
                },
                {
                    "source_name": "CO2",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "bio3-high-tier",
                    "priority_tier": Tier.CURATED_TARGETED,
                    "provenance": "p1",
                },
            ]
        )
        idx = TieredNameBucketIndex(df=df)
        hits = idx.lookup("agb_flow", "co2", Bucket.AIR)

        assert [h.target_code for h in hits] == ["bio3-high-tier", "ef-low-tier"]
        assert hits[0].tier == Tier.CURATED_TARGETED
        assert hits[1].tier == Tier.HARMONISED_FLOWS

    def test_lookup_normalizes_name_lowercase_and_whitespace(self):
        df = make_mappings_biosphere_df(
            [{"source_name": "  CO2  ", "source_top_bucket": Bucket.AIR}]
        )
        idx = TieredNameBucketIndex(df=df)
        # Indexed name is lowered+stripped at build time.
        assert idx.lookup("agb_flow", "co2", Bucket.AIR)
        assert idx.lookup("agb_flow", "CO2", Bucket.AIR) == []  # case-sensitive lookup arg

    def test_lookup_accepts_bucket_enum_or_string(self):
        df = make_mappings_biosphere_df([{"source_name": "X", "source_top_bucket": Bucket.WATER}])
        idx = TieredNameBucketIndex(df=df)
        assert idx.lookup("agb_flow", "x", Bucket.WATER)
        assert idx.lookup("agb_flow", "x", "water")

    def test_any_source_bucket_fallback_when_no_exact_or_unspecified_hit(self):
        """When a source name has rows only under specific other buckets (e.g.
        a placeholder mapping pinned to soil, but the AGB exchange is air),
        the lookup must surface those rows so the matcher's _resolve_target
        can do its target-side compartment work. Without this fallback, the
        air-exchange would silently fail to match Bifenazate even though a
        soil-bucket placeholder mapping exists for it."""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Bifenazate",
                    "source_top_bucket": Bucket.SOIL,  # mapping pinned to soil
                    "target_db": "biosphere3",
                    "target_code": "",
                    "target_name": "Bifenazate",
                    "priority_tier": Tier.CURATED_TARGETED,
                }
            ]
        )
        idx = TieredNameBucketIndex(df=df)
        # An air-bucket exchange must still surface the soil-pinned mapping.
        air_hits = idx.lookup("agb_flow", "bifenazate", Bucket.AIR)
        assert [h.target_name for h in air_hits] == ["Bifenazate"]
        # The soil exchange of course also gets it.
        soil_hits = idx.lookup("agb_flow", "bifenazate", Bucket.SOIL)
        assert [h.target_name for h in soil_hits] == ["Bifenazate"]

    def test_exact_bucket_takes_precedence_over_any_bucket_fallback(self):
        """When BOTH the exchange bucket and another bucket have a row for
        this name, the exact-bucket row wins — the any-bucket fallback only
        fires when neither exact-bucket NOR UNSPECIFIED has any candidate."""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Methane",
                    "source_top_bucket": Bucket.SOIL,
                    "target_db": "biosphere3",
                    "target_code": "soil-pin",
                    "priority_tier": Tier.CURATED_TARGETED,
                },
                {
                    "source_name": "Methane",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "air-pin",
                    "priority_tier": Tier.CURATED_TARGETED,
                },
            ]
        )
        idx = TieredNameBucketIndex(df=df)
        air_hits = idx.lookup("agb_flow", "methane", Bucket.AIR)
        # Exact-bucket only — the soil-bucket row must NOT be surfaced when
        # the exchange already has a precise air-bucket mapping.
        assert [h.target_code for h in air_hits] == ["air-pin"]

    def test_unspecified_bucket_entries_fire_for_any_bucket(self):
        """Curated synonyms / SimaPro biosphere manual matches don't constrain
        compartment — they're stored as bucket=UNSPECIFIED. The lookup must
        surface them for any actual exchange bucket so they can fire."""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Acetylene",
                    "source_top_bucket": Bucket.UNSPECIFIED,
                    "target_db": "biosphere3",
                    "target_code": "ethyne-uuid",
                    "target_name": "Ethyne",
                    "priority_tier": Tier.CURATED_SYNONYM_FALLBACK,
                    "provenance": "curated",
                },
                {
                    "source_name": "Acetylene",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "ef",
                    "target_code": "ef-uuid",
                    "priority_tier": Tier.HARMONISED_FLOWS,
                    "provenance": "harmonised",
                },
            ]
        )
        idx = TieredNameBucketIndex(df=df)

        # Air-bucket exchange: gets BOTH the air-specific harmonised hit AND the
        # bucket-agnostic curated synonym — the matcher's tier sort then prefers
        # whichever is higher priority.
        air_hits = idx.lookup("agb_flow", "acetylene", Bucket.AIR)
        codes = sorted(h.target_code for h in air_hits)
        assert codes == ["ef-uuid", "ethyne-uuid"]

        # Soil-bucket exchange: only the bucket-agnostic synonym fires.
        soil_hits = idx.lookup("agb_flow", "acetylene", Bucket.SOIL)
        assert [h.target_code for h in soil_hits] == ["ethyne-uuid"]

        # Explicit unspecified lookup: must NOT double-count the unspecified entry.
        unspec_hits = idx.lookup("agb_flow", "acetylene", Bucket.UNSPECIFIED)
        assert [h.target_code for h in unspec_hits] == ["ethyne-uuid"]

    def test_unmatchable_flag_propagates(self):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Forever chem",
                    "source_top_bucket": Bucket.AIR,
                    "is_unmatchable": True,
                    "priority_tier": Tier.UNMATCHABLE,
                }
            ]
        )
        idx = TieredNameBucketIndex(df=df)
        hits = idx.lookup("agb_flow", "forever chem", Bucket.AIR)
        assert hits[0].is_unmatchable is True


class TestCasIndex:
    def test_empty_df_returns_empty_list(self):
        assert CasIndex(df=pd.DataFrame()).lookup("12-34-5") == []

    def test_lookup_returns_rows_for_known_cas(self):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "CO2",
                    "source_cas": "124-38-9",
                    "source_top_bucket": Bucket.AIR,
                    "target_code": "code-1",
                }
            ]
        )
        idx = CasIndex(df=df)
        hits = idx.lookup("124-38-9")
        assert len(hits) == 1
        assert hits[0].target_code == "code-1"

    def test_blank_cas_lookup_returns_no_hits(self):
        df = make_mappings_biosphere_df(
            [{"source_name": "X", "source_cas": "124-38-9", "target_code": "c1"}]
        )
        idx = CasIndex(df=df)
        # Empty lookup string never matches a real CAS.
        assert idx.lookup("") == []
        assert idx.lookup("   ") == []

    def test_cas_lookup_strips_whitespace(self):
        df = make_mappings_biosphere_df(
            [{"source_name": "X", "source_cas": "124-38-9", "target_code": "c1"}]
        )
        idx = CasIndex(df=df)
        assert idx.lookup("  124-38-9  ")[0].target_code == "c1"


class TestUnitConverter:
    def test_canonical_returns_alias_target(self):
        aliases = make_unit_aliases_df([{"alias": "a", "canonical": "year"}])
        conv = UnitConverter(conversions_df=pd.DataFrame(), aliases_df=aliases)
        assert conv.canonical("a") == "year"
        assert conv.canonical("A") == "year"  # case-insensitive

    def test_canonical_passthrough_when_no_alias(self):
        conv = UnitConverter(conversions_df=pd.DataFrame(), aliases_df=pd.DataFrame())
        assert conv.canonical("kg") == "kg"

    def test_canonical_empty_string(self):
        conv = UnitConverter(conversions_df=pd.DataFrame(), aliases_df=pd.DataFrame())
        assert conv.canonical("") == ""

    def test_multiplier_returns_one_when_units_equal(self):
        conv = UnitConverter(conversions_df=pd.DataFrame(), aliases_df=pd.DataFrame())
        assert conv.multiplier("kg", "kg") == 1.0
        assert conv.multiplier("KG", "kg") == 1.0  # case-insensitive

    def test_multiplier_via_alias_then_match(self):
        conv = UnitConverter(
            conversions_df=pd.DataFrame(),
            aliases_df=make_unit_aliases_df([{"alias": "a", "canonical": "year"}]),
        )
        assert conv.multiplier("a", "year") == 1.0

    def test_multiplier_uses_conversion_table(self):
        conv = UnitConverter(
            conversions_df=make_unit_conversions_df(
                [{"source_unit": "m", "target_unit": "km", "multiplier": 0.001}]
            ),
            aliases_df=pd.DataFrame(),
        )
        assert conv.multiplier("m", "km") == 0.001

    def test_multiplier_returns_none_when_no_conversion(self):
        conv = UnitConverter(conversions_df=pd.DataFrame(), aliases_df=pd.DataFrame())
        assert conv.multiplier("kg", "m") is None

    def test_multiplier_returns_none_for_blank_units(self):
        conv = UnitConverter(conversions_df=pd.DataFrame(), aliases_df=pd.DataFrame())
        assert conv.multiplier("", "kg") is None
        assert conv.multiplier("kg", "") is None

    def test_alias_does_not_rewrite_known_canonical_unit(self):
        """A flowmapper-style alias that renames `kilo becquerel` to
        `kilobecquerel` (no space) must be ignored — the conversion table
        keys on `kilo becquerel`, so re-aliasing strands every Bq → kBq
        lookup. Regression test for the production unit-converter bug.
        """
        conv = UnitConverter(
            conversions_df=make_unit_conversions_df(
                [
                    {
                        "source_unit": "Becquerel",
                        "target_unit": "kilo Becquerel",
                        "multiplier": 0.001,
                    }
                ]
            ),
            aliases_df=make_unit_aliases_df(
                [
                    # Conflict: flowmapper says rename to snake_case;
                    # conversion table uses space-separated form.
                    {"alias": "kilo becquerel", "canonical": "kilobecquerel"},
                    {"alias": "kbq", "canonical": "kilo Becquerel"},
                ]
            ),
        )
        assert conv.canonical("kilo Becquerel") == "kilo Becquerel"
        assert conv.multiplier("Becquerel", "kilo Becquerel") == 0.001

    def test_alias_canonical_rewritten_when_snake_case_form_unknown(self):
        """If the alias's canonical target is snake_case but the matching
        space-separated form is in the conversion table, the alias should
        be rewritten so it lands on a key the converter can resolve."""
        conv = UnitConverter(
            conversions_df=make_unit_conversions_df(
                [
                    {
                        "source_unit": "kilogram",
                        "target_unit": "cubic meter",
                        "multiplier": 0.001,
                    }
                ]
            ),
            aliases_df=make_unit_aliases_df([{"alias": "m3", "canonical": "cubic_meter"}]),
        )
        assert conv.canonical("m3").lower() == "cubic meter"
        assert conv.multiplier("kilogram", "m3") == 0.001


class TestUnmatchableIndex:
    def test_empty_df_says_nothing_is_unmatchable(self):
        idx = UnmatchableIndex(df=pd.DataFrame())
        assert idx.is_unmatchable("X", Bucket.AIR) is False

    def test_known_unmatchable_returns_true(self):
        df = make_unmatchable_df(
            [
                {
                    "source_name": "Forever chem",
                    "source_top_bucket": Bucket.AIR,
                    "is_unmatchable": True,
                }
            ]
        )
        idx = UnmatchableIndex(df=df)
        assert idx.is_unmatchable("Forever chem", Bucket.AIR) is True
        assert idx.is_unmatchable("forever chem", "air") is True
        assert idx.is_unmatchable("Other", Bucket.AIR) is False

    def test_membership_keyed_on_bucket_too(self):
        df = make_unmatchable_df([{"source_name": "X", "source_top_bucket": Bucket.WATER}])
        idx = UnmatchableIndex(df=df)
        assert idx.is_unmatchable("X", Bucket.WATER) is True
        assert idx.is_unmatchable("X", Bucket.AIR) is False

    @pytest.mark.parametrize("value", ["", None])
    def test_blank_or_none_name_returns_false(self, value):
        df = make_unmatchable_df([{"source_name": "X", "source_top_bucket": Bucket.AIR}])
        idx = UnmatchableIndex(df=df)
        assert idx.is_unmatchable(value, Bucket.AIR) is False
