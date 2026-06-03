"""Unit tests for ``RegionalSuffixParser``."""

from __future__ import annotations

import pytest

from matching.regional_suffix import RegionalSuffixParser


@pytest.fixture
def parser() -> RegionalSuffixParser:
    return RegionalSuffixParser()


class TestRegionalSuffixParser:
    def test_parses_iso2_suffix(self, parser):
        assert parser.parse("Water, well, CN") == ("Water, well", "CN")

    def test_parses_subregional_suffix(self, parser):
        assert parser.parse("Water, lake, CA-QC") == ("Water, lake", "CA-QC")

    def test_parses_row_suffix(self, parser):
        assert parser.parse("Water, unspecified natural origin, RoW") == (
            "Water, unspecified natural origin",
            "RoW",
        )

    def test_no_suffix_returns_empty(self, parser):
        assert parser.parse("Water, lake") == ("Water, lake", "")

    def test_single_letter_suffix_does_not_match(self, parser):
        assert parser.parse("Sodium, R") == ("Sodium, R", "")

    def test_aggregate_region_glo(self, parser):
        assert parser.parse("Water, well, GLO") == ("Water, well", "GLO")

    def test_aggregate_region_rer(self, parser):
        assert parser.parse("Water, well, RER") == ("Water, well", "RER")

    def test_aggregate_region_oecd(self, parser):
        assert parser.parse("Water, well, OECD") == ("Water, well", "OECD")

    def test_chemical_formula_suffix_not_matched(self, parser):
        """``HCC-30`` is a chemical formula code, not a region. The
        parser must NOT misclassify it as a region."""
        assert parser.parse("Methane, dichloro-, HCC-30") == (
            "Methane, dichloro-, HCC-30",
            "",
        )

    def test_cfc_code_not_matched(self, parser):
        assert parser.parse("Ethane, 1,1,2-trichloro-1,2,2-trifluoro-, CFC-113") == (
            "Ethane, 1,1,2-trichloro-1,2,2-trifluoro-, CFC-113",
            "",
        )

    def test_halon_code_not_matched(self, parser):
        assert parser.parse("Methane, bromotrifluoro-, Halon 1301") == (
            "Methane, bromotrifluoro-, Halon 1301",
            "",
        )

    def test_subregion_invalid_iso_not_matched(self, parser):
        """``ZZ-QC`` is not a valid ISO-2 prefix → not a region."""
        assert parser.parse("Water, lake, ZZ-QC") == ("Water, lake, ZZ-QC", "")

    def test_empty_string(self, parser):
        assert parser.parse("") == ("", "")

    def test_only_water_name(self, parser):
        """Bare ``Water`` has no suffix at all."""
        assert parser.parse("Water") == ("Water", "")

    def test_water_in_release(self, parser):
        """AGB water release ``Water, IN`` (India) — must parse as regional."""
        assert parser.parse("Water, IN") == ("Water", "IN")

    def test_water_us_release(self, parser):
        assert parser.parse("Water, US") == ("Water", "US")

    def test_trailing_whitespace_is_handled(self, parser):
        assert parser.parse("Water, lake, FR  ") == ("Water, lake", "FR")

    def test_no_comma_means_no_region(self, parser):
        assert parser.parse("Sodium chloride") == ("Sodium chloride", "")
