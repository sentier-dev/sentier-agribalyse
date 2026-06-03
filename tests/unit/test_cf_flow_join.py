"""Unit tests for ``ef.cf_flow_join`` (Stage 2 flow-level CF join)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ef.cf_flow_join import (
    BiosphereCatalogLoader,
    ContextNormaliser,
    FlowLevelCfJoiner,
    JoinedFlowFrame,
    JoinedFlowParquetWriter,
    SimaProCfIndex,
    SimaProCfUnitHarmoniser,
)

# ---------------------------------------------------------------------------
# Fixtures.


def _rules_fixture() -> pd.DataFrame:
    """Minimal context_normalisation.parquet stand-in: 4 rules.

    Forward direction (SP → ecoinvent): ``[Air, high. pop.]`` →
    ``[air, urban air close to ground]`` etc. The normaliser inverts these.
    """
    rows = [
        {
            "source_context": np.array(["Air", "high. pop."], dtype=object),
            "target_context": np.array(["air", "urban air close to ground"], dtype=object),
        },
        {
            "source_context": np.array(["Air", "(unspecified)"], dtype=object),
            "target_context": np.array(["air", "(unspecified)"], dtype=object),
        },
        {
            "source_context": np.array(["Water", "ocean"], dtype=object),
            "target_context": np.array(["water", "ocean"], dtype=object),
        },
        # A rule whose source compartment is not in the SP-adapted namespace
        # (must be ignored when building the inverse map).
        {
            "source_context": np.array(["Raw materials", "in ground"], dtype=object),
            "target_context": np.array(["natural resource", "in ground"], dtype=object),
        },
    ]
    return pd.DataFrame(rows)


def _sp_df_fixture() -> pd.DataFrame:
    """Minimal simapro-EF31-adapted-cfs.parquet stand-in for Ozone depletion.

    Three substances; one with an ``indoor`` zero variant to exercise the
    CAS preference for ``(unspecified)``.
    """
    return pd.DataFrame(
        [
            # CFC-11 — primary by name + exact context.
            {
                "simapro_method": "Ozone depletion",
                "simapro_method_unit": "kg CFC-11 eq",
                "compartment": "Air",
                "sub_compartment": "(unspecified)",
                "name": "Methane, trichlorofluoro-, CFC-11",
                "cas": "000075-69-4",
                "cf": 1.0,
                "flow_unit": "kg",
                "cf_unit": "kg CFC-11 eq / kg",
            },
            # Bromomethane — has both indoor (cf=0) and (unspecified) (cf=0.57)
            # entries; CAS lookup must prefer the (unspecified) one.
            {
                "simapro_method": "Ozone depletion",
                "simapro_method_unit": "kg CFC-11 eq",
                "compartment": "Air",
                "sub_compartment": "indoor",
                "name": "Bromomethane",
                "cas": "000074-83-9",
                "cf": 0.0,
                "flow_unit": "kg",
                "cf_unit": "kg CFC-11 eq / kg",
            },
            {
                "simapro_method": "Ozone depletion",
                "simapro_method_unit": "kg CFC-11 eq",
                "compartment": "Air",
                "sub_compartment": "(unspecified)",
                "name": "Bromomethane",
                "cas": "000074-83-9",
                "cf": 0.57,
                "flow_unit": "kg",
                "cf_unit": "kg CFC-11 eq / kg",
            },
            # CFC-115 — gives the short-name fallback a target. SP's long
            # name "Ethane, chloropentafluoro-, CFC-115" should match a
            # JRC short name "CFC-115" via the short-name index.
            {
                "simapro_method": "Ozone depletion",
                "simapro_method_unit": "kg CFC-11 eq",
                "compartment": "Air",
                "sub_compartment": "(unspecified)",
                "name": "Ethane, chloropentafluoro-, CFC-115",
                "cas": "000076-15-3",
                "cf": 0.26,
                "flow_unit": "kg",
                "cf_unit": "kg CFC-11 eq / kg",
            },
        ]
    )


def _catalog_fixture() -> pd.DataFrame:
    """Minimal biosphere_catalog.parquet stand-in covering the SP fixture."""
    return pd.DataFrame(
        [
            {
                "database": "ecoinvent-3.9.1-biosphere",
                "code": "cfc11-air-unspec",
                "name": "Methane, trichlorofluoro-, CFC-11",
                "categories": np.array(["air", "(unspecified)"], dtype=object),
                "unit": "kilogram",
                "cas": "000075-69-4",
                "synonyms": np.array([], dtype=object),
            },
            {
                "database": "ecoinvent-3.9.1-biosphere",
                "code": "bromomethane-urban",
                "name": "Bromomethane",
                "categories": np.array(["air", "urban air close to ground"], dtype=object),
                "unit": "kilogram",
                "cas": "000074-83-9",
                "synonyms": np.array(["Methyl bromide"], dtype=object),
            },
            {
                # JRC-style entry with NO cas — must resolve via short_name
                # cross-reference inside FlowLevelCfJoiner.
                "database": "ef",
                "code": "cfc115-jrc-air",
                "name": "CFC-115",
                "categories": np.array(
                    [
                        "Emissions",
                        "Emissions to air",
                        "Emissions to urban air close to ground",
                    ],
                    dtype=object,
                ),
                "unit": "kilogram",
                "cas": None,
                "synonyms": np.array([], dtype=object),
            },
        ]
    )


# ---------------------------------------------------------------------------
# ContextNormaliser.


class TestContextNormaliser:
    @pytest.fixture
    def normaliser(self) -> ContextNormaliser:
        return ContextNormaliser(rules=_rules_fixture())

    def test_ecoinvent_categories_use_inverse_rule(self, normaliser: ContextNormaliser) -> None:
        # urban air close to ground → high. pop.
        assert normaliser.normalise(["air", "urban air close to ground"]) == (
            "Air",
            "high. pop.",
        )

    def test_unspecified_compartment_maps_through_inverse(
        self, normaliser: ContextNormaliser
    ) -> None:
        assert normaliser.normalise(["air", "(unspecified)"]) == ("Air", "(unspecified)")

    def test_natural_resource_uses_built_in_compartment_mapping(
        self, normaliser: ContextNormaliser
    ) -> None:
        # The "Raw materials, in ground" rule is excluded from the inverse
        # (its source compartment is outside the SP namespace) — so the
        # built-in dict maps "natural resource" → "Raw" and the
        # sub-compartment falls back to the ecoinvent string.
        assert normaliser.normalise(["natural resource", "in ground"]) == (
            "Raw",
            "in ground",
        )

    def test_jrc_ef_format_is_handled(self, normaliser: ContextNormaliser) -> None:
        # The 3-element JRC EF categories list goes through the dedicated
        # _EF_CATEGORIES_TO_SP_CONTEXT dictionary.
        assert normaliser.normalise(
            ["Emissions", "Emissions to air", "Emissions to urban air close to ground"]
        ) == ("Air", "high. pop.")

    def test_jrc_ef_long_term_air_maps_to_low_pop_long_term(
        self, normaliser: ContextNormaliser
    ) -> None:
        assert normaliser.normalise(
            ["Emissions", "Emissions to air", "Emissions to air, unspecified (long-term)"]
        ) == ("Air", "low. pop., long-term")

    def test_empty_categories_returns_defaults(self, normaliser: ContextNormaliser) -> None:
        assert normaliser.normalise([]) == ("", "(unspecified)")

    def test_unknown_eco_top_bucket_title_cases_compartment(
        self, normaliser: ContextNormaliser
    ) -> None:
        # Falls outside the SP namespace; we don't have a CFs there but the
        # normaliser shouldn't crash.
        comp, sub = normaliser.normalise(["something_weird", "subbucket"])
        assert comp == "Something_Weird"
        assert sub == "subbucket"


# ---------------------------------------------------------------------------
# SimaProCfIndex.


class TestSimaProCfIndex:
    OZONE_KEY: tuple[str, str, str, str] = (
        "ecoinvent-3.9.1",
        "EF v3.1",
        "ozone depletion",
        "ozone depletion potential (ODP)",
    )

    @pytest.fixture
    def sp_index(self) -> SimaProCfIndex:
        return SimaProCfIndex.from_dataframe(_sp_df_fixture())

    def test_exact_name_lookup_hits(self, sp_index: SimaProCfIndex) -> None:
        cf, prov = sp_index.lookup(
            registry_method_key=self.OZONE_KEY,
            name="Methane, trichlorofluoro-, CFC-11",
            context=("Air", "(unspecified)"),
            synonyms=None,
            cas=None,
        )
        assert cf == pytest.approx(1.0)
        assert prov == SimaProCfIndex.PROV_EXACT

    def test_synonym_fallback(self, sp_index: SimaProCfIndex) -> None:
        # Bromomethane is the SP name; a synonym "Methyl bromide" should
        # also hit (with the same key).
        cf, prov = sp_index.lookup(
            registry_method_key=self.OZONE_KEY,
            name="Methyl bromide",  # not present as primary
            context=("Air", "(unspecified)"),
            synonyms=["Bromomethane"],
            cas=None,
        )
        assert cf == pytest.approx(0.57)
        assert prov == SimaProCfIndex.PROV_SYNONYM

    def test_cas_lookup_prefers_unspecified_over_indoor_zero(
        self, sp_index: SimaProCfIndex
    ) -> None:
        # Bromomethane appears with (unspecified)→0.57 and indoor→0.0.
        # CAS lookup for a context the SP table doesn't have must fall
        # through to (unspecified)=0.57, not the indoor zero — even when
        # the indoor row is processed first during index construction
        # (the priority tuple (is_unspec, -name_len) ensures unspec wins).
        cf, prov = sp_index.lookup(
            registry_method_key=self.OZONE_KEY,
            name="Methane, bromo-",  # not in SP primary
            context=("Air", "low. pop., long-term"),
            synonyms=None,
            cas="000074-83-9",
        )
        assert cf == pytest.approx(0.57)
        assert prov == SimaProCfIndex.PROV_CAS

    def test_cas_lookup_breaks_ties_by_shortest_name(self) -> None:
        # SP has three rows sharing one CAS (e.g. Uranium variants of
        # different energy densities). The canonical (shortest-named)
        # row wins regardless of iteration order.
        df = pd.DataFrame(
            [
                {
                    "simapro_method": "Resource use, fossils",
                    "simapro_method_unit": "MJ",
                    "compartment": "Raw",
                    "sub_compartment": "(unspecified)",
                    "name": "Uranium, 2291 GJ per kg",
                    "cas": "007440-61-1",
                    "cf": 2_291_000.0,
                    "flow_unit": "kg",
                    "cf_unit": "MJ / kg",
                },
                {
                    "simapro_method": "Resource use, fossils",
                    "simapro_method_unit": "MJ",
                    "compartment": "Raw",
                    "sub_compartment": "(unspecified)",
                    "name": "Uranium, 451 GJ per kg",
                    "cas": "007440-61-1",
                    "cf": 451_000.0,
                    "flow_unit": "kg",
                    "cf_unit": "MJ / kg",
                },
                {
                    "simapro_method": "Resource use, fossils",
                    "simapro_method_unit": "MJ",
                    "compartment": "Raw",
                    "sub_compartment": "(unspecified)",
                    "name": "Uranium",
                    "cas": "007440-61-1",
                    "cf": 560_000.0,
                    "flow_unit": "kg",
                    "cf_unit": "MJ / kg",
                },
            ]
        )
        index = SimaProCfIndex.from_dataframe(df)
        cf, prov = index.lookup(
            registry_method_key=(
                "ecoinvent-3.9.1",
                "EF v3.1",
                "energy resources: non-renewable",
                "abiotic depletion potential (ADP): fossil fuels",
            ),
            # ecoinvent's flow name for uranium is different ("Uranium ore"
            # etc.) so the primary lookup misses and the CAS tier handles
            # the match. The canonical SP "Uranium" row must win the
            # CAS-side tie against the GJ-per-kg variants.
            name="Uranium ore",
            context=("Raw", "in ground"),
            synonyms=None,
            cas="007440-61-1",
        )
        # Canonical "Uranium" (shortest name) wins, not the 451 GJ variant
        # or 2291 GJ variant.
        assert cf == pytest.approx(560_000.0)
        assert prov == SimaProCfIndex.PROV_CAS

    def test_short_name_skipped_for_non_chemical_identifiers(self) -> None:
        # SP has "Energy, unspecified" (cf=1.00) — short_name "unspecified".
        # An ecoinvent flow "Coal, hard, unspecified" must NOT match it via
        # the short-name tier, because "unspecified" carries no digit and
        # is therefore not a chemical identifier.
        df = pd.DataFrame(
            [
                {
                    "simapro_method": "Resource use, fossils",
                    "simapro_method_unit": "MJ",
                    "compartment": "Raw",
                    "sub_compartment": "(unspecified)",
                    "name": "Energy, unspecified",
                    "cas": "",
                    "cf": 1.0,
                    "flow_unit": "MJ",
                    "cf_unit": "MJ / MJ",
                },
            ]
        )
        index = SimaProCfIndex.from_dataframe(df)
        cf, prov = index.lookup(
            registry_method_key=(
                "ecoinvent-3.9.1",
                "EF v3.1",
                "energy resources: non-renewable",
                "abiotic depletion potential (ADP): fossil fuels",
            ),
            name="Coal, hard, unspecified",
            context=("Raw", "in ground"),
            synonyms=None,
            cas=None,
        )
        assert cf is None
        assert prov == SimaProCfIndex.PROV_UNMATCHED

    def test_water_use_kg_cf_is_scaled_by_1000_to_match_registry(self) -> None:
        # SP's "Water" emission CF is stored as ``-0.042955`` per kg (SP
        # converted JRC's per-m3 CF). The registry stores JRC's raw
        # per-m3 value (``-42.95``). The harmoniser must scale SP's
        # per-kg CF by 1000 at index-build time so the dashboard
        # comparison reads a near-zero delta.
        df = pd.DataFrame(
            [
                {
                    "simapro_method": "Water use",
                    "simapro_method_unit": "m3 depriv.",
                    "compartment": "Water",
                    "sub_compartment": "(unspecified)",
                    "name": "Water",
                    "cas": "007732-18-5",
                    "cf": -0.042955,
                    "flow_unit": "kg",
                    "cf_unit": "m3 depriv. / kg",
                },
            ]
        )
        index = SimaProCfIndex.from_dataframe(df)
        cf, _ = index.lookup(
            registry_method_key=(
                "ecoinvent-3.9.1",
                "EF v3.1",
                "water use",
                "user deprivation potential (deprivation-weighted water consumption)",
            ),
            name="Water",
            context=("Water", "(unspecified)"),
            synonyms=None,
            cas=None,
        )
        assert cf == pytest.approx(-42.955)

    def test_water_use_m3_cf_is_not_scaled(self) -> None:
        # SP's "Water, fresh" resource CF is already per m3 (matches
        # JRC's frame); the harmoniser must leave it untouched.
        df = pd.DataFrame(
            [
                {
                    "simapro_method": "Water use",
                    "simapro_method_unit": "m3 depriv.",
                    "compartment": "Raw",
                    "sub_compartment": "(unspecified)",
                    "name": "Water, fresh",
                    "cas": "007732-18-5",
                    "cf": 42.95,
                    "flow_unit": "m3",
                    "cf_unit": "m3 depriv. / m3",
                },
            ]
        )
        index = SimaProCfIndex.from_dataframe(df)
        cf, _ = index.lookup(
            registry_method_key=(
                "ecoinvent-3.9.1",
                "EF v3.1",
                "water use",
                "user deprivation potential (deprivation-weighted water consumption)",
            ),
            name="Water, fresh",
            context=("Raw", "(unspecified)"),
            synonyms=None,
            cas=None,
        )
        assert cf == pytest.approx(42.95)

    def test_primary_falls_back_to_unspecified_sub(self) -> None:
        # SP only has one row for "Coal, brown" at (Raw, (unspecified)).
        # An ecoinvent flow "Coal, brown" at (Raw, "in ground") must still
        # match via the (unspecified) sub-compartment fallback.
        df = pd.DataFrame(
            [
                {
                    "simapro_method": "Resource use, fossils",
                    "simapro_method_unit": "MJ",
                    "compartment": "Raw",
                    "sub_compartment": "(unspecified)",
                    "name": "Coal, brown",
                    "cas": "",
                    "cf": 9.41,
                    "flow_unit": "kg",
                    "cf_unit": "MJ / kg",
                },
            ]
        )
        index = SimaProCfIndex.from_dataframe(df)
        cf, prov = index.lookup(
            registry_method_key=(
                "ecoinvent-3.9.1",
                "EF v3.1",
                "energy resources: non-renewable",
                "abiotic depletion potential (ADP): fossil fuels",
            ),
            name="Coal, brown",
            context=("Raw", "in ground"),
            synonyms=None,
            cas=None,
        )
        assert cf == pytest.approx(9.41)
        assert prov == SimaProCfIndex.PROV_EXACT

    def test_short_name_fallback(self, sp_index: SimaProCfIndex) -> None:
        # JRC short name "CFC-115" must match SP's long name
        # "Ethane, chloropentafluoro-, CFC-115".
        cf, prov = sp_index.lookup(
            registry_method_key=self.OZONE_KEY,
            name="CFC-115",
            context=("Air", "(unspecified)"),
            synonyms=None,
            cas=None,
        )
        assert cf == pytest.approx(0.26)
        assert prov == SimaProCfIndex.PROV_SHORT_NAME

    def test_unmatched_when_no_strategy_hits(self, sp_index: SimaProCfIndex) -> None:
        cf, prov = sp_index.lookup(
            registry_method_key=self.OZONE_KEY,
            name="Definitely not a substance",
            context=("Air", "(unspecified)"),
            synonyms=None,
            cas="invalid-cas",
        )
        assert cf is None
        assert prov == SimaProCfIndex.PROV_UNMATCHED


# ---------------------------------------------------------------------------
# SimaProCfUnitHarmoniser.


class TestSimaProCfUnitHarmoniser:
    def test_default_scale_is_one(self) -> None:
        assert SimaProCfUnitHarmoniser.scale_for("climate change", "kg") == 1.0
        assert SimaProCfUnitHarmoniser.scale_for("acidification", "kg") == 1.0

    def test_water_use_kg_scales_to_m3_frame(self) -> None:
        assert SimaProCfUnitHarmoniser.scale_for("water use", "kg") == 1000.0

    def test_water_use_m3_is_identity(self) -> None:
        assert SimaProCfUnitHarmoniser.scale_for("water use", "m3") == 1.0

    def test_unknown_method_unit_combo_is_identity(self) -> None:
        assert SimaProCfUnitHarmoniser.scale_for("water use", "MJ") == 1.0


# ---------------------------------------------------------------------------
# FlowLevelCfJoiner.


class TestFlowLevelCfJoiner:
    OZONE_KEY: tuple[str, str, str, str] = (
        "ecoinvent-3.9.1",
        "EF v3.1",
        "ozone depletion",
        "ozone depletion potential (ODP)",
    )

    @pytest.fixture
    def joiner(self) -> FlowLevelCfJoiner:
        return FlowLevelCfJoiner(
            catalog=_catalog_fixture(),
            normaliser=ContextNormaliser(rules=_rules_fixture()),
            sp_index=SimaProCfIndex.from_dataframe(_sp_df_fixture()),
        )

    def test_exact_match_when_name_and_context_align(self, joiner: FlowLevelCfJoiner) -> None:
        ef = pd.DataFrame(
            [{"database": "ecoinvent-3.9.1-biosphere", "code": "cfc11-air-unspec", "amount": 1.0}]
        )
        out = joiner.join_method(self.OZONE_KEY, ef)
        assert len(out.df) == 1
        row = out.df.iloc[0]
        assert row["sp_cf"] == pytest.approx(1.0)
        assert row["ef_cf"] == pytest.approx(1.0)
        assert row["sp_match_provenance"] == SimaProCfIndex.PROV_EXACT

    def test_primary_unspec_fallback_for_specific_sub(self, joiner: FlowLevelCfJoiner) -> None:
        # Bromomethane catalog row has context (air, urban air close to ground)
        # → normalised to (Air, high. pop.). SP has no (Air, high. pop.) row
        # for Bromomethane but does have an exact-name (Air, "(unspecified)")
        # entry, so the primary lookup's (unspecified) fallback finds 0.57.
        ef = pd.DataFrame(
            [
                {
                    "database": "ecoinvent-3.9.1-biosphere",
                    "code": "bromomethane-urban",
                    "amount": 0.57,
                }
            ]
        )
        out = joiner.join_method(self.OZONE_KEY, ef)
        row = out.df.iloc[0]
        assert row["sp_cf"] == pytest.approx(0.57)
        assert row["sp_match_provenance"] == SimaProCfIndex.PROV_EXACT

    def test_jrc_short_name_match_for_catalog_row_without_cas(
        self, joiner: FlowLevelCfJoiner
    ) -> None:
        # The JRC EF-style catalog entry "CFC-115" has cas=None. The
        # joiner's cas_by_short_name cross-reference (built from the rest
        # of the catalog) supplies a CAS via the matching short-name
        # entry — but our fixture catalog has no CFC-115 ecoinvent row, so
        # the short-name path inside the SP index is the working tier.
        ef = pd.DataFrame([{"database": "ef", "code": "cfc115-jrc-air", "amount": 0.26}])
        out = joiner.join_method(self.OZONE_KEY, ef)
        row = out.df.iloc[0]
        assert row["sp_cf"] == pytest.approx(0.26)
        assert row["sp_match_provenance"] == SimaProCfIndex.PROV_SHORT_NAME

    def test_unknown_code_produces_nan_with_unmatched_provenance(
        self, joiner: FlowLevelCfJoiner
    ) -> None:
        ef = pd.DataFrame(
            [{"database": "ecoinvent-3.9.1-biosphere", "code": "nonexistent", "amount": 0.5}]
        )
        out = joiner.join_method(self.OZONE_KEY, ef)
        row = out.df.iloc[0]
        assert pd.isna(row["sp_cf"])
        assert row["ef_cf"] == pytest.approx(0.5)
        assert row["sp_match_provenance"] == SimaProCfIndex.PROV_UNMATCHED

    def test_empty_ef_input_returns_empty_frame(self, joiner: FlowLevelCfJoiner) -> None:
        ef = pd.DataFrame(columns=["database", "code", "amount"])
        out = joiner.join_method(self.OZONE_KEY, ef)
        assert out.df.empty
        assert list(out.df.columns) == list(JoinedFlowFrame.COLUMNS)


# ---------------------------------------------------------------------------
# JoinedFlowParquetWriter.


class TestJoinedFlowParquetWriter:
    OZONE_KEY: tuple[str, str, str, str] = (
        "ecoinvent-3.9.1",
        "EF v3.1",
        "ozone depletion",
        "ozone depletion potential (ODP)",
    )

    def test_writes_frames_with_method_key_columns(self, tmp_path: Path) -> None:
        frame = JoinedFlowFrame(
            method_key=self.OZONE_KEY,
            df=pd.DataFrame(
                [
                    {
                        "code": "abc",
                        "name": "X",
                        "categories": ["air", "(unspecified)"],
                        "sp_cf": 1.0,
                        "ef_cf": 1.0,
                        "sp_match_provenance": SimaProCfIndex.PROV_EXACT,
                    }
                ]
            ),
        )
        out_path = tmp_path / "cf_per_flow_joined.parquet"
        written = JoinedFlowParquetWriter(out_path=out_path).write([frame])
        assert written == out_path
        df = pd.read_parquet(written)
        assert len(df) == 1
        # Method key fans out across 4 cols
        assert df.loc[0, "method_database"] == self.OZONE_KEY[0]
        assert df.loc[0, "method_ef_version"] == self.OZONE_KEY[1]
        assert df.loc[0, "method_category"] == self.OZONE_KEY[2]
        assert df.loc[0, "method_indicator"] == self.OZONE_KEY[3]
        assert df.loc[0, "code"] == "abc"
        assert df.loc[0, "sp_cf"] == pytest.approx(1.0)

    def test_empty_input_writes_empty_parquet_with_schema(self, tmp_path: Path) -> None:
        out_path = tmp_path / "cf_per_flow_joined.parquet"
        JoinedFlowParquetWriter(out_path=out_path).write([])
        df = pd.read_parquet(out_path)
        assert df.empty
        expected = {
            "method_database",
            "method_ef_version",
            "method_category",
            "method_indicator",
            "code",
            "name",
            "categories",
            "sp_cf",
            "ef_cf",
            "sp_match_provenance",
        }
        assert set(df.columns) == expected


# ---------------------------------------------------------------------------
# BiosphereCatalogLoader.


class TestBiosphereCatalogLoader:
    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        loader = BiosphereCatalogLoader(path=tmp_path / "missing.parquet")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_returns_documented_columns(self, tmp_path: Path) -> None:
        cat = _catalog_fixture()
        path = tmp_path / "biosphere_catalog.parquet"
        cat.to_parquet(path)
        loader = BiosphereCatalogLoader(path=path)
        df = loader.load()
        assert set(df.columns) >= {
            "database",
            "code",
            "name",
            "categories",
            "unit",
            "cas",
            "synonyms",
        }
