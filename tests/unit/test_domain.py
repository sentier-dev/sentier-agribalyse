"""Unit tests for ``domain`` — pure data classes and enums."""

from __future__ import annotations

import dataclasses

import pytest

from domain import (
    AUDIT_COLUMNS,
    MAPPING_COLUMNS,
    AuditEntry,
    Bucket,
    DropEvent,
    Mapping,
    MatchOutcome,
    OverrideKind,
    SourceKind,
    SuppressedStrategy,
    Tier,
    UnitConversion,
)


class TestTier:
    def test_tier_ordering_is_priority_ascending(self):
        assert Tier.CURATED_TARGETED < Tier.HARMONISED_FLOWS
        assert Tier.HARMONISED_FLOWS < Tier.UNMATCHABLE

    def test_fill_only_tiers_membership(self):
        fill_only = Tier.fill_only_tiers()
        assert Tier.EF_GENERIC in fill_only
        assert Tier.BIO3_MATCH_DATABASE in fill_only
        assert Tier.LLM_OVERRIDES in fill_only
        assert Tier.CURATED_SYNONYM_FALLBACK in fill_only
        assert Tier.CASE_INSENSITIVE_FALLBACK in fill_only
        assert Tier.CURATED_TARGETED not in fill_only
        assert Tier.HARMONISED_FLOWS not in fill_only

    def test_llm_gated_tiers_membership(self):
        gated = Tier.llm_gated_tiers()
        assert gated == {Tier.LLM_OVERRIDES, Tier.CURATED_SYNONYM_FALLBACK}

    def test_is_fill_only_matches_class_set(self):
        for t in Tier:
            assert t.is_fill_only() == (t in Tier.fill_only_tiers())

    def test_is_llm_gated_matches_class_set(self):
        for t in Tier:
            assert t.is_llm_gated() == (t in Tier.llm_gated_tiers())

    def test_int_values_are_stable(self):
        # Behaviour relies on these positions; freezing them prevents accidental renumbering.
        assert int(Tier.CURATED_TARGETED) == 1
        assert int(Tier.AGRIBALYSE_EI_BIOSPHERE) == 4
        assert int(Tier.UNMATCHABLE) == 13


class TestBucket:
    @pytest.mark.parametrize(
        "cats,expected",
        [
            ((), Bucket.UNSPECIFIED),
            (None, Bucket.UNSPECIFIED),
            (("Emissions to air",), Bucket.AIR),
            (("Emissions to water", "river"), Bucket.WATER),
            (("Emissions to soil",), Bucket.SOIL),
            (("Resources",), Bucket.RESOURCE),
            (("Raw materials",), Bucket.RESOURCE),
            (("air",), Bucket.AIR),
            (("Final waste flows",), Bucket.UNSPECIFIED),
            # EF schema: compartment lives in cat[1], not cat[0].
            (
                ("Emissions", "Emissions to air", "Emissions to urban air close to ground"),
                Bucket.AIR,
            ),
            (
                ("Emissions", "Emissions to water", "Emissions to fresh water"),
                Bucket.WATER,
            ),
            (
                ("Emissions", "Emissions to soil", "Emissions to agricultural soil"),
                Bucket.SOIL,
            ),
            (
                (
                    "Resources",
                    "Resources from ground",
                    "Non-renewable element resources from ground",
                ),
                Bucket.RESOURCE,
            ),
            (
                ("Resources", "Resources from air", "Renewable material resources from air"),
                Bucket.RESOURCE,
            ),
            # Ecoinvent ('natural resource', 'in air') must remain RESOURCE,
            # not be reclassified as AIR by the new multi-element scan.
            (("natural resource", "in air"), Bucket.RESOURCE),
            (("natural resource", "in water"), Bucket.RESOURCE),
        ],
    )
    def test_from_categories(self, cats, expected):
        assert Bucket.from_categories(cats) == expected

    @pytest.mark.parametrize(
        "iri,expected",
        [
            ("envi-air-indr-unkn", Bucket.AIR),
            ("envi-wate-suwa", Bucket.WATER),
            ("envi-grou-soil", Bucket.SOIL),
            ("reso-grou", Bucket.RESOURCE),
            ("laus", Bucket.RESOURCE),
            ("", Bucket.UNSPECIFIED),
            ("unknown-iri", Bucket.UNSPECIFIED),
        ],
    )
    def test_from_harmonised_iri(self, iri, expected):
        assert Bucket.from_harmonised_iri(iri) == expected

    def test_strenum_values_are_lowercase(self):
        assert Bucket.AIR.value == "air"
        assert Bucket.UNSPECIFIED.value == "unspecified"


class TestMapping:
    def test_default_mapping_has_curated_tier(self):
        m = Mapping(source_kind=SourceKind.AGB_FLOW, source_name="X", source_unit="kg")
        assert m.priority_tier == Tier.CURATED_TARGETED
        assert m.target_key == ("", "")
        assert m.unit_conversion == 1.0

    def test_target_key_returns_db_code_tuple(self):
        m = Mapping(
            source_kind=SourceKind.AGB_FLOW,
            source_name="X",
            source_unit="kg",
            target_db="biosphere3",
            target_code="abc",
        )
        assert m.target_key == ("biosphere3", "abc")

    def test_name_lower_strips_and_lowercases(self):
        m = Mapping(source_kind=SourceKind.AGB_FLOW, source_name="  CO2  ", source_unit="kg")
        assert m.name_lower == "co2"

    def test_mapping_columns_constant_includes_required_fields(self):
        for col in (
            "source_kind",
            "source_name",
            "target_db",
            "target_code",
            "priority_tier",
            "is_unmatchable",
        ):
            assert col in MAPPING_COLUMNS

    def test_mapping_is_frozen(self):
        m = Mapping(source_kind=SourceKind.AGB_FLOW, source_name="X", source_unit="kg")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.target_db = "biosphere3"  # type: ignore[misc]


class TestMatchOutcome:
    def test_hit_marks_matched_true(self):
        out = MatchOutcome.hit(
            tier=Tier.CURATED_TARGETED,
            target_db="biosphere3",
            target_code="c",
            target_unit="kg",
        )
        assert out.matched is True
        assert out.tier == Tier.CURATED_TARGETED
        assert out.target_db == "biosphere3"
        assert out.skipped_reason == ""

    def test_miss_marks_matched_false_and_carries_reason(self):
        out = MatchOutcome.miss("no candidates")
        assert out.matched is False
        assert out.skipped_reason == "no candidates"
        assert out.target_db == ""

    def test_hit_default_unit_conversion_is_one(self):
        out = MatchOutcome.hit(
            tier=Tier.CURATED_TARGETED,
            target_db="b",
            target_code="c",
            target_unit="kg",
        )
        assert out.unit_conversion == 1.0


class TestAuditEntryAndOverrideKind:
    def test_audit_columns_include_kind_and_tiers(self):
        for col in ("kind", "new_tier", "prior_tier", "reason", "unit_conversion"):
            assert col in AUDIT_COLUMNS

    def test_audit_entry_default_prior_state(self):
        entry = AuditEntry(
            process_name="p",
            exchange_name="e",
            exchange_unit="kg",
            exchange_bucket="air",
            kind=OverrideKind.NEW_LINK,
            new_tier=Tier.CURATED_TARGETED,
            new_target_db="b",
            new_target_code="c",
            new_provenance="prov",
        )
        assert entry.prior_tier is None
        assert entry.prior_target_db == ""
        assert entry.prior_target_code == ""
        assert entry.unit_conversion == 1.0

    def test_override_kind_values_stable(self):
        # Names are persisted in the audit parquet; freezing them prevents accidental renames.
        assert OverrideKind.NEW_LINK.value == "new_link"
        assert OverrideKind.OVERRIDE.value == "override"
        assert OverrideKind.SAME_TIER_OVERRIDE.value == "same_tier_override"
        assert OverrideKind.REJECTED_UNIT_MISMATCH.value == "rejected_unit_mismatch"


class TestValueObjects:
    def test_drop_event_carries_count(self):
        e = DropEvent(strategy="x", n_dropped=3)
        assert e.n_dropped == 3
        assert e.process_name == ""

    def test_suppressed_strategy_holds_error_metadata(self):
        s = SuppressedStrategy(strategy="s", error_type="ValueError", error_message="boom")
        assert s.error_type == "ValueError"
        assert s.error_message == "boom"
        assert s.occurred_at_step == ""

    def test_unit_conversion_default_provenance(self):
        u = UnitConversion(source_unit="m", target_unit="km", multiplier=0.001)
        assert u.multiplier == 0.001
        assert u.provenance == ""
