"""Unit tests for ``BiosphereMatcher``.

The matcher is the most behaviour-dense class in the codebase. The tests
cover:

* tier ordering — a tier-1 row beats a tier-4 row on the same key;
* fill-only protection — a tier-7+ candidate cannot displace a prior link;
* LLM gating — tier-10 candidates skipped under ``--no-llm`` symmetric;
* unit equality — units must match (or have a registered conversion);
* unit conversion — amount + unit are rescaled when conversion is registered;
* unmatchable index — known-unmatchable flows skip silently;
* CAS fallback — name miss + CAS hit still resolves;
* deterministic tie-break — when multiple candidates exist at the same tier,
  the lexicographically smallest provenance/code wins;
* zero-amount and final-waste drops are counted in the tracker.
"""

from __future__ import annotations

import pytest

from domain import Bucket, Tier
from matching.audit import AuditLog, DropTallyTracker
from matching.bio_catalog import BioFlowRef
from matching.biosphere import BiosphereMatcher
from tests.fixtures.builders import (
    make_bio_catalog,
    make_dataset,
    make_exchange,
    make_mappings_biosphere_df,
    make_registry,
    make_unit_aliases_df,
    make_unit_conversions_df,
    make_unmatchable_df,
)


def _matcher(settings, tmp_path, *, registry, catalog=None):
    catalog = catalog or make_bio_catalog([])
    return BiosphereMatcher(
        settings=settings,
        registry=registry,
        catalog=catalog,
        audit=AuditLog(output_path=tmp_path / "audit.parquet"),
        drops=DropTallyTracker(),
    )


class TestSimpleHappyPath:
    def test_links_a_known_flow_via_registry(self, settings, tmp_path):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Carbon dioxide",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "co2-uuid",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="co2-uuid",
                    name="Carbon dioxide",
                    unit="kg",
                    bucket=Bucket.AIR,
                )
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "Process A",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Carbon dioxide",
                        unit="kg",
                        amount=2.0,
                        categories=("air",),
                    )
                ],
            )
        ]
        stats = m.match(sp_data)
        exc = sp_data[0]["exchanges"][0]
        assert exc["input"] == ("biosphere3", "co2-uuid")
        assert exc["_match_tier"] == int(Tier.CURATED_TARGETED)
        assert stats.n_total == 1
        assert stats.n_linked == 1


class TestTierOrdering:
    def test_higher_priority_tier_wins(self, settings, tmp_path):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Methane",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "low-tier-target",
                    "priority_tier": Tier.HARMONISED_FLOWS,
                    "target_unit": "kg",
                    "provenance": "low",
                },
                {
                    "source_name": "Methane",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "high-tier-target",
                    "priority_tier": Tier.CURATED_TARGETED,
                    "target_unit": "kg",
                    "provenance": "high",
                },
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="high-tier-target",
                    name="Methane",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
                BioFlowRef(
                    db="biosphere3",
                    code="low-tier-target",
                    name="Methane low",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="Methane", unit="kg", categories=("air",))
                ],
            )
        ]
        m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == ("biosphere3", "high-tier-target")


class TestFillOnlyProtection:
    def test_fill_only_tier_does_not_displace_prior_link(self, settings, tmp_path):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "X",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "ef-generic-target",
                    "target_unit": "kg",
                    "priority_tier": Tier.EF_GENERIC,
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="ef-generic-target",
                    name="X",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
                BioFlowRef(
                    db="biosphere3", code="prior-target", name="X", unit="kg", bucket=Bucket.AIR
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="X",
                        unit="kg",
                        categories=("air",),
                        input=("biosphere3", "prior-target"),
                    )
                ],
            )
        ]
        m.match(sp_data)
        # Prior link must survive the EF_GENERIC fill-only attempt.
        assert sp_data[0]["exchanges"][0]["input"] == ("biosphere3", "prior-target")


class TestLlmGate:
    def test_llm_tier_skipped_when_overrides_disabled(self, settings, tmp_path):
        s = settings.with_no_llm()
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Y",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "llm-target",
                    "target_unit": "kg",
                    "priority_tier": Tier.LLM_OVERRIDES,
                }
            ]
        )
        catalog = make_bio_catalog(
            [BioFlowRef(db="biosphere3", code="llm-target", name="Y", unit="kg", bucket=Bucket.AIR)]
        )
        m = _matcher(s, tmp_path, registry=make_registry(s, mappings_biosphere=df), catalog=catalog)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="Y", unit="kg", categories=("air",))
                ],
            )
        ]
        m.match(sp_data)
        assert "input" not in sp_data[0]["exchanges"][0]


class TestUnitEnforcement:
    def test_unit_mismatch_with_conversion_rescales_amount_and_unit(self, settings, tmp_path):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Water",
                    "source_top_bucket": Bucket.WATER,
                    "target_db": "biosphere3",
                    "target_code": "water-m3-target",
                    "target_unit": "m3",
                    "priority_tier": Tier.CURATED_TARGETED,
                }
            ]
        )
        conv = make_unit_conversions_df(
            [{"source_unit": "kg", "target_unit": "m3", "multiplier": 0.001}]
        )
        reg = make_registry(settings, mappings_biosphere=df, unit_conversions=conv)
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="water-m3-target",
                    name="Water",
                    unit="m3",
                    bucket=Bucket.WATER,
                )
            ]
        )
        m = _matcher(settings, tmp_path, registry=reg, catalog=catalog)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Water",
                        unit="kg",
                        amount=1000.0,
                        categories=("water",),
                    )
                ],
            )
        ]
        m.match(sp_data)
        exc = sp_data[0]["exchanges"][0]
        assert exc["input"] == ("biosphere3", "water-m3-target")
        assert exc["amount"] == pytest.approx(1.0)
        assert exc["unit"] == "m3"

    def test_per_row_unit_conversion_is_authoritative_when_non_default(self, settings, tmp_path):
        """A registry row carrying an explicit non-default ``unit_conversion``
        bypasses the global converter. Encodes substance-specific factors
        (water density 0.001, wood bulk density, gas calorific content,
        radionuclide specific activity) the global table can't express."""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Cesium-137",
                    "source_top_bucket": Bucket.SOIL,
                    "source_unit": "Becquerel",
                    "target_db": "biosphere3",
                    "target_code": "cs137-kbq",
                    "target_unit": "kilo Becquerel",
                    "unit_conversion": 0.001,  # explicit Bq → kBq factor
                    "priority_tier": Tier.CURATED_TARGETED,
                }
            ]
        )
        # Deliberately NO global conversion — the row is authoritative.
        reg = make_registry(settings, mappings_biosphere=df)
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="cs137-kbq",
                    name="Cesium-137",
                    unit="kilo Becquerel",
                    bucket=Bucket.SOIL,
                )
            ]
        )
        m = _matcher(settings, tmp_path, registry=reg, catalog=catalog)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Cesium-137",
                        unit="Becquerel",
                        amount=1000.0,
                        categories=("soil",),
                    )
                ],
            )
        ]
        m.match(sp_data)
        exc = sp_data[0]["exchanges"][0]
        assert exc["input"] == ("biosphere3", "cs137-kbq")
        assert exc["amount"] == pytest.approx(1.0)
        assert exc["unit"] == "kilo Becquerel"

    def test_per_row_conversion_does_not_double_apply_when_exchange_already_converted(
        self, settings, tmp_path
    ):
        """If a prior transform (e.g. ``BiosphereFlowmapApplier``) already
        rescaled the exchange to the target unit, the matcher must not
        re-apply the registry row's ``unit_conversion``.

        Regression for the radioactivity 30–60× under-score (FIX_DATA.md
        § 2): the flowmap converted Bq→kBq once on every AGB radioactivity
        edge; the matcher then overrode the link at tier 4 with a
        ``unit_conversion=0.001`` row authored for ``Bq`` source units —
        and re-multiplied the already-kBq amount by 0.001, producing a
        1000× under-count.
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Radon-222",
                    "source_top_bucket": Bucket.AIR,
                    "source_unit": "Bq",
                    "target_db": "biosphere3",
                    "target_code": "rn222-kbq",
                    "target_unit": "kilo Becquerel",
                    "unit_conversion": 0.001,
                    "priority_tier": Tier.CURATED_TARGETED,
                }
            ]
        )
        # Global converter knows Bq → kBq. With my fix that path wins
        # over the cand-row override, so the kBq-already exchange stays
        # at 1.0 multiplier.
        conv = make_unit_conversions_df(
            [{"source_unit": "Becquerel", "target_unit": "kilo Becquerel", "multiplier": 0.001}]
        )
        aliases = make_unit_aliases_df([{"alias": "Bq", "canonical": "Becquerel"}])
        reg = make_registry(
            settings,
            mappings_biosphere=df,
            unit_conversions=conv,
            unit_aliases=aliases,
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="rn222-kbq",
                    name="Radon-222",
                    unit="kilo Becquerel",
                    bucket=Bucket.AIR,
                )
            ]
        )
        m = _matcher(settings, tmp_path, registry=reg, catalog=catalog)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    # Exchange is *already* in kBq — a prior flowmap step
                    # converted it. Amount must stay at 13.44, not be
                    # divided by 1000 again.
                    make_exchange(
                        type="biosphere",
                        name="Radon-222",
                        unit="kilo Becquerel",
                        amount=13.44,
                        categories=("air", "non-urban air or from high stacks"),
                    )
                ],
            )
        ]
        m.match(sp_data)
        exc = sp_data[0]["exchanges"][0]
        assert exc["input"] == ("biosphere3", "rn222-kbq")
        assert exc["amount"] == pytest.approx(13.44)
        assert exc["unit"] == "kilo Becquerel"

    def test_sub_compartment_tie_break_prefers_matching_target(self, settings, tmp_path):
        """When N candidates share (tier, provenance) and differ only in
        target sub-compartment, prefer the one whose target's sub-comp
        matches the exchange's sub-compartment.

        Regression for FIX_DATA.md § 3 — eutrophication freshwater
        159× outlier on sea-cage fish products. The harmonised-flows-simple
        registry produces 3 tier-5 candidates for ``Phosphorus, Total``
        water (one each for ``Emissions to sea water``, ``Emissions to
        fresh water``, ``Emissions to water, unspecified``). Without
        sub-compartment-aware tie-break, the lex-smallest target code
        (``Emissions to water, unspecified``) won, applying a CF=1.0 to
        ocean P emissions that should have CF=0 in the freshwater
        eutrophication method, inflating sea-cage fish products by ~160×.
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Phosphorus, Total",
                    "source_top_bucket": Bucket.WATER,
                    "target_db": "ef",
                    "target_code": "ef-water-unspec",
                    "target_unit": "kg",
                    "priority_tier": Tier.HARMONISED_FLOWS,
                },
                {
                    "source_name": "Phosphorus, Total",
                    "source_top_bucket": Bucket.WATER,
                    "target_db": "ef",
                    "target_code": "ef-water-fresh",
                    "target_unit": "kg",
                    "priority_tier": Tier.HARMONISED_FLOWS,
                },
                {
                    "source_name": "Phosphorus, Total",
                    "source_top_bucket": Bucket.WATER,
                    "target_db": "ef",
                    "target_code": "ef-water-sea",
                    "target_unit": "kg",
                    "priority_tier": Tier.HARMONISED_FLOWS,
                },
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ef",
                    code="ef-water-unspec",
                    name="Phosphorus, Total",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=(
                        "Emissions",
                        "Emissions to water",
                        "Emissions to water, unspecified",
                    ),
                ),
                BioFlowRef(
                    db="ef",
                    code="ef-water-fresh",
                    name="Phosphorus, Total",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=("Emissions", "Emissions to water", "Emissions to fresh water"),
                ),
                BioFlowRef(
                    db="ef",
                    code="ef-water-sea",
                    name="Phosphorus, Total",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=("Emissions", "Emissions to water", "Emissions to sea water"),
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "Sea bream cage",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Phosphorus, Total",
                        unit="kg",
                        amount=12000.0,
                        # Normalised AGB sub-compartment for ocean is "ocean"
                        # which maps to the "sea" group same as
                        # "Emissions to sea water" — the matcher should pick
                        # ef-water-sea, not the lex-smallest ef-water-fresh.
                        categories=("water", "ocean"),
                    )
                ],
            )
        ]
        m.match(sp_data)
        exc = sp_data[0]["exchanges"][0]
        assert exc["input"] == ("ef", "ef-water-sea")

    def test_long_term_subcompartment_wins_over_short_term_sibling(self, settings, tmp_path):
        """When two same-tier candidates share the coarse compartment group
        (e.g. both 'ground') but differ in their long-term suffix, the
        candidate whose target sub-compartment exactly matches the
        exchange's must win — long-term emissions characterise at CF=0
        in EF v3.1 ecotox while non-long-term ground- characterises at
        CF=301.44 (same as fresh water).

        Regression for Sardine canned ecotox: AGB
        ``market for potassium sulfate RER`` emits 1.66 kg of
        ``Chloride [groundwater, long-term]`` per kg of K2SO4. Without
        exact-match preference in ``_sub_rank``, both ``ground-`` and
        ``ground-, long-term`` map to the same 'ground' group; the
        lex-smallest target UUID (``85bec19f`` = ``ground-``) wins,
        characterising the long-term emission at CF=301 instead of CF=0
        and inflating Sardine ecotox by ~31 CTUe/kg.
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Chloride",
                    "source_top_bucket": Bucket.WATER,
                    "source_context": ["Emissions to water", "groundwater"],
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": "uuid-ground-short-term",
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                },
                {
                    "source_name": "Chloride",
                    "source_top_bucket": Bucket.WATER,
                    "source_context": ["Emissions to water", "groundwater, long-term"],
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": "uuid-ground-long-term",
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                },
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="uuid-ground-short-term",
                    name="Chloride",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=("water", "ground-"),
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="uuid-ground-long-term",
                    name="Chloride",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=("water", "ground-, long-term"),
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "market for potassium sulfate RER",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Chloride",
                        unit="kg",
                        amount=1.6634938,
                        categories=("water", "ground-, long-term"),
                    ),
                    make_exchange(
                        type="biosphere",
                        name="Chloride",
                        unit="kg",
                        amount=0.0014624,
                        categories=("water", "ground-"),
                    ),
                ],
            )
        ]
        m.match(sp_data)
        long_term_exc, short_term_exc = sp_data[0]["exchanges"]
        assert long_term_exc["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "uuid-ground-long-term",
        )
        assert short_term_exc["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "uuid-ground-short-term",
        )

    def test_uuid_pinned_target_with_empty_target_db_resolves_per_compartment(
        self, settings, tmp_path
    ):
        """``AGB32FlowmapperBiosphereSource`` ingests ~4 039 rows with
        ``target_db=""`` but ``target_code=<uuid>``. The matcher must
        resolve those by UUID across the configured biosphere DBs, not
        fall through to name-based lookup that lex-tie-breaks across
        sub-compartments.

        Regression: Sardine canned ecotox 17× outlier. AGB
        ``market for potassium sulfate RER`` emits chloride to five
        distinct water sub-compartments (river 0.74 kg, ground-, long-term
        1.66 kg, ocean, ground-, unspecified). With the bug, all five
        collapsed onto ``[water, surface water]`` — the lex-smallest
        chloride code in bio3 — because ``_resolve_target`` saw an empty
        ``target_db`` and discarded the explicit ``target_code`` UUID.
        Result: 1.66 kg of long-term groundwater chloride (CF=0 in EF v3.1
        ecotox) was characterised at CF=301.44, inflating Sardine's score
        ~17× over ADEME reference.

        Two same-tier candidates differ ONLY in which UUID their
        ``target_code`` points at — the one matching the source
        sub-compartment must win.
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Chloride",
                    "source_top_bucket": Bucket.WATER,
                    "source_context": ["Emissions to water", "river"],
                    "target_db": "",
                    "target_code": "uuid-surface-water",
                    "target_name": "Chloride",
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                    "provenance": "agribalyse.ecoinvent-3.10-biosphere-flowmapper",
                },
                {
                    "source_name": "Chloride",
                    "source_top_bucket": Bucket.WATER,
                    "source_context": ["Emissions to water", "groundwater, long-term"],
                    "target_db": "",
                    "target_code": "uuid-ground-long-term",
                    "target_name": "Chloride",
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                    "provenance": "agribalyse.ecoinvent-3.10-biosphere-flowmapper",
                },
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="uuid-surface-water",
                    name="Chloride",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=("water", "surface water"),
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="uuid-ground-long-term",
                    name="Chloride",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=("water", "ground-, long-term"),
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "market for potassium sulfate RER",
                exchanges=[
                    # Long-term groundwater chloride emission — must NOT
                    # collapse onto surface water (lex-smallest target_code).
                    make_exchange(
                        type="biosphere",
                        name="Chloride",
                        unit="kg",
                        amount=1.6634938,
                        categories=("water", "ground-, long-term"),
                    ),
                    # River (surface water) chloride — must land on
                    # surface water target.
                    make_exchange(
                        type="biosphere",
                        name="Chloride",
                        unit="kg",
                        amount=0.74067843,
                        categories=("water", "surface water"),
                    ),
                ],
            )
        ]
        m.match(sp_data)
        long_term_exc = sp_data[0]["exchanges"][0]
        river_exc = sp_data[0]["exchanges"][1]
        assert long_term_exc["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "uuid-ground-long-term",
        )
        assert river_exc["input"] == ("ecoinvent-3.9.1-biosphere", "uuid-surface-water")

    def test_unit_mismatch_without_conversion_skips_and_logs(self, settings, tmp_path):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Z",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "z-target",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                }
            ]
        )
        catalog = make_bio_catalog(
            [BioFlowRef(db="biosphere3", code="z-target", name="Z", unit="kg", bucket=Bucket.AIR)]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="Z", unit="m3", categories=("air",))
                ],
            )
        ]
        m.match(sp_data)
        exc = sp_data[0]["exchanges"][0]
        assert "input" not in exc
        # The unit-mismatch event made it into the audit log.
        assert any(e.kind.value == "rejected_unit_mismatch" for e in m.audit.entries)


class TestUnmatchableSkip:
    def test_unmatchable_flow_skips_silently(self, settings, tmp_path):
        unmatchable = make_unmatchable_df(
            [
                {
                    "source_name": "Forever",
                    "source_top_bucket": Bucket.AIR,
                    "is_unmatchable": True,
                }
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, unmatchable=unmatchable),
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="Forever", unit="kg", categories=("air",))
                ],
            )
        ]
        stats = m.match(sp_data)
        assert "input" not in sp_data[0]["exchanges"][0]
        assert stats.n_unmatchable_recognised == 1

    def test_real_mapping_beats_unmatchable_entry(self, settings, tmp_path):
        """The unmatchable list (placeholder Neither + randonneur unlinked) must NOT
        preempt matching when a real mapping exists at any tier — otherwise
        every flow rescued by LLM/harmonised/curated layers would be silently
        dropped just because the upstream placeholder couldn't classify it."""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Rescued",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "rescued-uuid",
                    "target_unit": "kg",
                    "priority_tier": Tier.LLM_OVERRIDES,
                }
            ]
        )
        unmatchable = make_unmatchable_df(
            [
                {
                    "source_name": "Rescued",
                    "source_top_bucket": Bucket.AIR,
                    "is_unmatchable": True,
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="rescued-uuid",
                    name="Rescued",
                    unit="kg",
                    bucket=Bucket.AIR,
                )
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df, unmatchable=unmatchable),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="Rescued", unit="kg", categories=("air",))
                ],
            )
        ]
        stats = m.match(sp_data)
        exc = sp_data[0]["exchanges"][0]
        assert exc["input"] == ("biosphere3", "rescued-uuid")
        assert stats.n_linked == 1
        assert stats.n_unmatchable_recognised == 0


class TestTargetBucketFallbacks:
    """``_resolve_target`` has TWO lookup steps for name-based mappings:

    1. Exact-bucket match in each preferred DB
    2. UNSPECIFIED-bucket fallback (UNSPECIFIED targets are bucket-agnostic)

    Cross-compartment fallback (air ↔ water ↔ soil) is NOT performed:
    the reviewer policy is that compartment mismatches must be discarded.

    UUID-only rows (``target_db=""`` + ``target_code=<uuid>``, the
    flowmapper randonneur shape) walk the preferred-DB list to honour
    the author's sub-compartment pin, but the resolved flow's bucket
    must still match the exchange's (or be ``UNSPECIFIED``). Without
    that guard, flowmapper rows pinned to bio3's only-bucket Bifenazate
    (``[soil, agricultural]``) would silently absorb AGB ``[air,
    non-urban air or from high stacks]`` pesticide emissions.

    Direct-pin rows (``target_db`` + ``target_code`` both set, used by
    ``curated_overrides``) bypass the bucket guard — the curator takes
    responsibility for explicit cross-compartment pins (e.g. AOX air →
    EF v3.1 ``adsorbable organic halogen compounds``, where the EF
    target is itself in the air bucket).
    """

    def _setup(self, settings, tmp_path, *, target_code: str, target_bucket: Bucket):
        """Matcher where the only target lives in ``target_bucket`` of bio3.

        ``target_code`` controls whether the mapping is UUID-pinned (when
        non-empty, ``_resolve_target`` does direct catalog lookup) or
        name-based (empty → triggers the bucket fallback chain).
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Americium-241",
                    "source_top_bucket": Bucket.UNSPECIFIED,
                    "target_db": "biosphere3",
                    "target_code": target_code,
                    "target_name": "Americium-241",
                    "target_unit": "kBq",
                    "priority_tier": Tier.LLM_OVERRIDES,
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="am-target",
                    name="Americium-241",
                    unit="kBq",
                    bucket=target_bucket,
                )
            ]
        )
        return _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )

    def _exchange(self, exchange_bucket: str):
        return [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Americium-241",
                        unit="kBq",
                        categories=(exchange_bucket,),
                    )
                ],
            )
        ]

    def test_unspecified_target_fallback_fires(self, settings, tmp_path):
        """Soil exchange links to an UNSPECIFIED-bucket target — these are
        bucket-agnostic by design (e.g. EF elementary flows without a
        compartment-specific category)."""
        m = self._setup(settings, tmp_path, target_code="", target_bucket=Bucket.UNSPECIFIED)
        sp_data = self._exchange("soil")
        m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == ("biosphere3", "am-target")

    def test_emission_to_emission_cross_compartment_not_linked(self, settings, tmp_path):
        """Pesticide observed as air emission in AGB but only present as a
        soil-emission flow in bio3 — discarded: reviewer policy is that
        cross-compartment matches are not acceptable."""
        m = self._setup(settings, tmp_path, target_code="", target_bucket=Bucket.SOIL)
        sp_data = self._exchange("air")
        m.match(sp_data)
        assert "input" not in sp_data[0]["exchanges"][0]

    def test_emission_does_not_link_to_resource_target(self, settings, tmp_path):
        """REFUSED: an air-emission of "Asbestos" must NOT silently link to a
        resource-bucket "Chrysotile" target. Emission and resource are
        different LCA categories (substance released vs. raw material
        extracted) — that conflation would corrupt the inventory."""
        m = self._setup(settings, tmp_path, target_code="", target_bucket=Bucket.RESOURCE)
        sp_data = self._exchange("air")
        m.match(sp_data)
        assert "input" not in sp_data[0]["exchanges"][0]

    def test_resource_does_not_link_to_emission_target(self, settings, tmp_path):
        """REFUSED in the other direction too: a resource-bucket exchange
        (e.g. natural-resource Crude oil) must NOT link to a water-emission
        Crude oil flow. Same category-mismatch principle."""
        m = self._setup(settings, tmp_path, target_code="", target_bucket=Bucket.WATER)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Americium-241",
                        unit="kBq",
                        categories=("natural resource", "in ground"),
                    )
                ],
            )
        ]
        m.match(sp_data)
        assert "input" not in sp_data[0]["exchanges"][0]

    def test_uuid_pinned_mapping_does_not_fall_back(self, settings, tmp_path):
        """UUID-pinned mapping (``target_db`` + ``target_code`` BOTH set) is the
        direct-hit branch — used by ``curated_overrides`` to express explicit
        cross-DB pins. The author chose that exact flow, including its
        compartment, so the matcher does not second-guess the bucket."""
        m = self._setup(settings, tmp_path, target_code="am-target", target_bucket=Bucket.WATER)
        sp_data = self._exchange("soil")
        m.match(sp_data)
        # Direct catalog hit by code — the bucket discrepancy is the author's
        # explicit choice, not silent drift.
        assert sp_data[0]["exchanges"][0]["input"] == ("biosphere3", "am-target")

    def test_uuid_only_flowmapper_refuses_cross_bucket(self, settings, tmp_path):
        """REGRESSION: ``AGB32FlowmapperBiosphereSource`` ingests rows with
        ``target_db=""`` + ``target_code=<uuid>``. Bio3 has many pesticides
        (Bifenazate, Boscalid, Kaolin, Pesticides unspecified, etc.) only
        in ``[soil, agricultural]``, but AGB also emits them as ``[air,
        non-urban air or from high stacks]``. The flowmapper UUID points
        at the soil flow because that's the only one available.

        Without a bucket guard the air-bucket exchange silently links to the
        soil-bucket flow and gets characterised under soil-emission CFs —
        materially wrong for an air emission. The matcher must refuse.

        For substances that genuinely need an air-target, the cross-compartment
        pin has to come through ``curated_overrides`` (target_db + target_code
        both set, e.g. AOX air → EF ``adsorbable organic halogen compounds``).
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Bifenazate",
                    "source_top_bucket": Bucket.SOIL,
                    "source_context": ["Emissions to soil", "agricultural"],
                    "target_db": "",
                    "target_code": "uuid-bifenazate-soil",
                    "target_name": "Bifenazate",
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                    "provenance": "agribalyse.ecoinvent-3.10-biosphere-flowmapper",
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="uuid-bifenazate-soil",
                    name="Bifenazate",
                    unit="kg",
                    bucket=Bucket.SOIL,
                    categories=("soil", "agricultural"),
                )
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Bifenazate",
                        unit="kg",
                        categories=("air", "non-urban air or from high stacks"),
                    )
                ],
            )
        ]
        m.match(sp_data)
        # Air → soil cross-compartment via UUID-only pin: refused.
        assert "input" not in sp_data[0]["exchanges"][0]

    def test_uuid_only_flowmapper_links_within_same_bucket(self, settings, tmp_path):
        """Counterpart to the cross-bucket guard: when the UUID-only flowmapper
        target IS in the exchange's bucket, the link must still happen.
        Protects the original ``a48b0d0`` chloride sub-compartment fix
        (water → water with the right sub-compartment UUID)."""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Chloride",
                    "source_top_bucket": Bucket.WATER,
                    "source_context": ["Emissions to water", "groundwater, long-term"],
                    "target_db": "",
                    "target_code": "uuid-ground-long-term",
                    "target_name": "Chloride",
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                    "provenance": "agribalyse.ecoinvent-3.10-biosphere-flowmapper",
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="uuid-ground-long-term",
                    name="Chloride",
                    unit="kg",
                    bucket=Bucket.WATER,
                    categories=("water", "ground-, long-term"),
                )
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Chloride",
                        unit="kg",
                        categories=("water", "ground-, long-term"),
                    )
                ],
            )
        ]
        m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "uuid-ground-long-term",
        )

    def test_uuid_only_accepts_unspecified_target(self, settings, tmp_path):
        """UNSPECIFIED-bucket targets are bucket-agnostic by design — a
        UUID-only pin to an UNSPECIFIED flow must still resolve regardless
        of the exchange bucket. (Generic EF flows like ``Phosphate`` with
        no compartment-specific category live here.)"""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Phosphate",
                    "source_top_bucket": Bucket.WATER,
                    "target_db": "",
                    "target_code": "uuid-phosphate-unspec",
                    "target_name": "Phosphate",
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                    "provenance": "agribalyse.ecoinvent-3.10-biosphere-flowmapper",
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ef",
                    code="uuid-phosphate-unspec",
                    name="Phosphate",
                    unit="kg",
                    bucket=Bucket.UNSPECIFIED,
                )
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Phosphate",
                        unit="kg",
                        categories=("water", "river"),
                    )
                ],
            )
        ]
        m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == ("ef", "uuid-phosphate-unspec")

    def test_exact_bucket_wins_over_unspecified(self, settings, tmp_path):
        """When the target exists in the exchange's bucket, prefer it over the
        UNSPECIFIED-bucket fallback (deterministic — no surprise routing)."""
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "X",
                    "source_top_bucket": Bucket.UNSPECIFIED,
                    "target_db": "biosphere3",
                    "target_code": "",
                    "target_name": "X",
                    "target_unit": "kg",
                    "priority_tier": Tier.LLM_OVERRIDES,
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3", code="exact-air", name="X", unit="kg", bucket=Bucket.AIR
                ),
                BioFlowRef(
                    db="biosphere3",
                    code="elsewhere-water",
                    name="X",
                    unit="kg",
                    bucket=Bucket.WATER,
                ),
                BioFlowRef(
                    db="biosphere3", code="unspec", name="X", unit="kg", bucket=Bucket.UNSPECIFIED
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="X", unit="kg", categories=("air",))
                ],
            )
        ]
        m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == ("biosphere3", "exact-air")

    def test_air_bucket_pinned_target_wins_when_unspecified_target_is_water_only(
        self, settings, tmp_path
    ):
        """Regression: AOX air-emission (curated_overrides AOX air → EF UUID).

        Two same-tier candidates compete for the same AGB flow:
        * a bucket-agnostic (UNSPECIFIED) name-based mapping to a target DB
          that holds the substance in WATER only;
        * an AIR-bucket-pinned mapping with explicit ``target_db + target_code``
          pointing at an AIR-bucket flow.

        For an AGB air-bucket exchange, the bucket-agnostic candidate fails to
        resolve (target name lookup in air bucket misses, UNSPECIFIED-bucket
        target fallback misses), so the matcher must fall through to the
        UUID-pinned air-bucket candidate. Without the air-bucket entry the
        exchange would stay unlinked because the placeholder.neither row
        (tier 13) short-circuits.
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "AOX, Adsorbable Organic Halogen",
                    "source_top_bucket": Bucket.UNSPECIFIED,
                    "target_db": "biosphere3",
                    "target_code": "",
                    "target_name": "AOX, Adsorbable Organic Halogen",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                    "provenance": "curated_overrides",
                },
                {
                    "source_name": "AOX, Adsorbable Organic Halogen",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "ef",
                    "target_code": "ef-aox-air",
                    "target_name": "adsorbable organic halogen compounds",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                    "provenance": "curated_overrides",
                },
            ]
        )
        unmatchable = make_unmatchable_df(
            [
                {
                    "source_name": "AOX, Adsorbable Organic Halogen",
                    "source_top_bucket": Bucket.AIR,
                }
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="biosphere3",
                    code="bio3-aox-water",
                    name="AOX, Adsorbable Organic Halogen",
                    unit="kg",
                    bucket=Bucket.WATER,
                ),
                BioFlowRef(
                    db="ef",
                    code="ef-aox-air",
                    name="adsorbable organic halogen compounds",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df, unmatchable=unmatchable),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="AOX, Adsorbable Organic Halogen",
                        unit="kg",
                        categories=("Emissions to air",),
                    )
                ],
            )
        ]
        m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == ("ef", "ef-aox-air")

    def test_empty_subcompartment_prefers_unspecified_target(self, settings, tmp_path):
        """When an AGB exchange has an empty sub-compartment (the AGB CSV
        emits emissions under a bare ``Emissions to air`` section header
        with no second token), the matcher must pick the target whose
        ``categories`` is a single-element tuple — the bio3 / ecoinvent
        unspecified variant — rather than lex-tie-breaking by UUID across
        all sub-compartment-scoped siblings.

        Regression for the fish-PM −51% cluster (72 wild-marine-fish SKUs).
        AGB authors ``Diesel combustion in marine engines {FR}`` with
        78.5 g of ``Nitrogen oxides`` per kg diesel under a section header
        ``Emissions to air`` and no sub-compartment. Five tier-4
        ``agb_flow → ecoinvent-3.9.1-biosphere`` candidates exist, one per
        sub-compartment variant. The exchange's empty ``exc_sub`` made
        every candidate score ``_sub_rank=3`` (unknown), so the final
        tie-break was ``target_code ASC``. For NOx the lex-smallest UUID
        is ``4841a0fe…`` = ``[air, lower stratosphere + upper troposphere]``
        whose EF v3.1 PM CF is 2.1E-7 — 7.6× lower than the ``[air]``
        unspecified variant (UUID ``c1b91234…``, CF=1.6E-6). All 79 g of
        NOx per kg diesel routed to the stratosphere, undercounting PM
        by ~half across every wild-fish SKU in the catalog.
        """
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Nitrogen oxides",
                    "source_top_bucket": Bucket.AIR,
                    "source_context": ["Emissions to air", ""],
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": uuid_lex_smallest,
                    "target_unit": "kg",
                    "priority_tier": Tier.AGRIBALYSE_EI_BIOSPHERE,
                }
                for uuid_lex_smallest in (
                    "4841a0fe-stratosphere",
                    "77357947-low-pop",
                    "9115356e-low-pop-long",
                    "c1b91234-unspec",
                    "d068f3e2-urban",
                )
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="4841a0fe-stratosphere",
                    name="Nitrogen oxides",
                    unit="kg",
                    bucket=Bucket.AIR,
                    categories=("air", "lower stratosphere + upper troposphere"),
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="77357947-low-pop",
                    name="Nitrogen oxides",
                    unit="kg",
                    bucket=Bucket.AIR,
                    categories=("air", "non-urban air or from high stacks"),
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="9115356e-low-pop-long",
                    name="Nitrogen oxides",
                    unit="kg",
                    bucket=Bucket.AIR,
                    categories=("air", "low population density, long-term"),
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="c1b91234-unspec",
                    name="Nitrogen oxides",
                    unit="kg",
                    bucket=Bucket.AIR,
                    categories=("air",),
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="d068f3e2-urban",
                    name="Nitrogen oxides",
                    unit="kg",
                    bucket=Bucket.AIR,
                    categories=("air", "urban air close to ground"),
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "Diesel combustion in marine engines",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Nitrogen oxides",
                        unit="kg",
                        amount=0.0785,
                        categories=("air",),
                    )
                ],
            )
        ]
        m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "c1b91234-unspec",
        )


class TestCleanupDrops:
    def test_zero_amount_unlinked_biosphere_flows_are_dropped(self, settings, tmp_path):
        m = _matcher(settings, tmp_path, registry=make_registry(settings))
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Empty",
                        unit="kg",
                        amount=0,
                        categories=("air",),
                    ),
                    make_exchange(
                        type="biosphere",
                        name="Real",
                        unit="kg",
                        amount=2.0,
                        categories=("air",),
                    ),
                ],
            )
        ]
        m.match(sp_data)
        names = [e["name"] for e in sp_data[0]["exchanges"]]
        assert "Empty" not in names
        assert "Real" in names
        assert m.drops.totals_by_strategy().get("drop_zero_amount_unlinked_biosphere", 0) == 1

    def test_final_waste_flows_unlinked_are_dropped(self, settings, tmp_path):
        m = _matcher(settings, tmp_path, registry=make_registry(settings))
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="WasteX",
                        unit="kg",
                        amount=1.0,
                        categories=("Final waste flows",),
                    )
                ],
            )
        ]
        m.match(sp_data)
        assert sp_data[0]["exchanges"] == []
        assert m.drops.totals_by_strategy().get("drop_final_waste_flows", 0) == 1


class TestRegionalSuffixPreservation:
    """``BiosphereMatcher`` must preserve the country-of-extraction
    suffix on every linked biosphere code that targets a regionalisable
    DB. Without this, the matrix collapses Water-well-CN, Water-well-US,
    Water-well-FR onto a single (db, base_uuid) row and the per-region
    AWARE CFs from SimaPro's adapted method become inaccessible.

    See [docs/superpowers/specs/2026-05-22-regional-flow-mapping-design.md](docs/superpowers/specs/2026-05-22-regional-flow-mapping-design.md).
    """

    def _matcher_with_water_well_mapping(self, settings, tmp_path):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Water, well, CN",
                    "source_top_bucket": Bucket.RESOURCE,
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": "water-well-uuid",
                    "target_unit": "m3",
                    "priority_tier": Tier.CURATED_TARGETED,
                },
                {
                    "source_name": "Water, well, FR",
                    "source_top_bucket": Bucket.RESOURCE,
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": "water-well-uuid",
                    "target_unit": "m3",
                    "priority_tier": Tier.CURATED_TARGETED,
                },
                {
                    "source_name": "Methane, dichloro-, HCC-30",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": "ch2cl2-uuid",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                },
                {
                    "source_name": "Carbon dioxide",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": "co2-uuid",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                },
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="water-well-uuid",
                    name="Water, well",
                    unit="m3",
                    bucket=Bucket.RESOURCE,
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="ch2cl2-uuid",
                    name="Methane, dichloro-",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="co2-uuid",
                    name="Carbon dioxide",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
            ]
        )
        return _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )

    def test_regional_suffix_appended_to_target_code(self, settings, tmp_path):
        """``Water, well, CN`` matches base code ``water-well-uuid`` and the
        exchange ``input`` records the synthetic ``water-well-uuid@CN`` so
        downstream B / Q matrices index the CN-specific row separately."""
        m = self._matcher_with_water_well_mapping(settings, tmp_path)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Water, well, CN",
                        unit="m3",
                        amount=2.0,
                        categories=("natural resource", "in water"),
                    )
                ],
            )
        ]
        stats = m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "water-well-uuid@CN",
        )
        assert stats.n_regional_suffixed == 1

    def test_three_regional_variants_get_distinct_codes(self, settings, tmp_path):
        """Three Water-well exchanges (CN, FR, no-region) must end up with
        three distinct ``(db, code)`` matrix keys."""
        m = self._matcher_with_water_well_mapping(settings, tmp_path)
        # Add a non-regional Water, well mapping so this exchange links.
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Water, well, CN",
                        unit="m3",
                        amount=2.0,
                        categories=("natural resource", "in water"),
                    ),
                    make_exchange(
                        type="biosphere",
                        name="Water, well, FR",
                        unit="m3",
                        amount=3.0,
                        categories=("natural resource", "in water"),
                    ),
                ],
            )
        ]
        stats = m.match(sp_data)
        codes = {tuple(e["input"]) for e in sp_data[0]["exchanges"]}
        assert codes == {
            ("ecoinvent-3.9.1-biosphere", "water-well-uuid@CN"),
            ("ecoinvent-3.9.1-biosphere", "water-well-uuid@FR"),
        }
        assert stats.n_regional_suffixed == 2

    def test_chemical_formula_suffix_does_not_get_treated_as_region(self, settings, tmp_path):
        """``Methane, dichloro-, HCC-30`` is a chemical formula token, not
        a region. The matcher must leave the code untouched."""
        m = self._matcher_with_water_well_mapping(settings, tmp_path)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Methane, dichloro-, HCC-30",
                        unit="kg",
                        amount=1.0,
                        categories=("air",),
                    )
                ],
            )
        ]
        stats = m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "ch2cl2-uuid",
        )
        assert stats.n_regional_suffixed == 0

    def test_no_regional_suffix_leaves_code_unchanged(self, settings, tmp_path):
        """Flows without regional suffix are linked as before."""
        m = self._matcher_with_water_well_mapping(settings, tmp_path)
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Carbon dioxide",
                        unit="kg",
                        amount=1.0,
                        categories=("air",),
                    )
                ],
            )
        ]
        stats = m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "co2-uuid",
        )
        assert stats.n_regional_suffixed == 0

    def test_simapro_name_falls_back_when_in_region_prefix_was_stripped(self, settings, tmp_path):
        """``RemoveBiosphereLocationPrefixIfFlowInSameLocation`` strips
        ``, FR`` from ``exc.name`` for a FR activity, but it preserves
        the original under ``exc["simapro name"]``. The matcher must
        re-read that so the regional suffix isn't silently lost for
        in-region flows."""
        # Add a mapping for the stripped name "Water, well" so the
        # matcher's normal candidate resolution succeeds.
        m = self._matcher_with_water_well_mapping(settings, tmp_path)
        # Need a registry row for the stripped name. Reuse the FR row
        # by also adding a base-name entry — easier to inject via a
        # fresh matcher.
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "Water, well",
                    "source_top_bucket": Bucket.RESOURCE,
                    "target_db": "ecoinvent-3.9.1-biosphere",
                    "target_code": "water-well-uuid",
                    "target_unit": "m3",
                    "priority_tier": Tier.CURATED_TARGETED,
                },
            ]
        )
        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="ecoinvent-3.9.1-biosphere",
                    code="water-well-uuid",
                    name="Water, well",
                    unit="m3",
                    bucket=Bucket.RESOURCE,
                ),
            ]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    {
                        "type": "biosphere",
                        "name": "Water, well",  # already stripped
                        "simapro name": "Water, well, FR",  # original
                        "unit": "m3",
                        "amount": 2.0,
                        "categories": ("natural resource", "in water"),
                    }
                ],
            )
        ]
        stats = m.match(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "water-well-uuid@FR",
        )
        assert stats.n_regional_suffixed == 1


class TestStatsCarryTierBuckets:
    def test_per_tier_count_records_winning_tier(self, settings, tmp_path):
        df = make_mappings_biosphere_df(
            [
                {
                    "source_name": "X",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "xc",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                }
            ]
        )
        catalog = make_bio_catalog(
            [BioFlowRef(db="biosphere3", code="xc", name="X", unit="kg", bucket=Bucket.AIR)]
        )
        m = _matcher(
            settings,
            tmp_path,
            registry=make_registry(settings, mappings_biosphere=df),
            catalog=catalog,
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="X", unit="kg", categories=("air",))
                ],
            )
        ]
        stats = m.match(sp_data)
        assert stats.by_tier[Tier.CURATED_TARGETED] == 1
