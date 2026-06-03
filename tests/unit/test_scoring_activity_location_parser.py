"""Tests for ``ActivityLocationParser`` — name → ISO/region extraction.

Pin the priority (brace before trailing-token), the trailing-token
length floor (so chemical symbols like ``N`` / ``P`` / ``K`` don't
match), sub-regional prefix collapse, and aggregate-region rejection.
"""

from __future__ import annotations

import pytest

from scoring.activity_location_parser import ActivityLocationParser


@pytest.fixture
def parser() -> ActivityLocationParser:
    return ActivityLocationParser()


# ----------------------------------------------------------------------------
# Brace form (priority)


def test_brace_form_simple(parser):
    assert parser.location_for("Frozen mango puree, at processing {BR} U") == "BR"


def test_brace_form_iso_then_trailing_decoration(parser):
    name = "Mango, pulp, raw, ... at consumer {FR} [Ciqual code: 13025] U"
    assert parser.location_for(name) == "FR"


def test_brace_form_picks_last_when_multiple(parser):
    """If multiple ``{XX}`` appear, the trailing one wins — that's the
    activity's own location, not an embedded country reference earlier
    in the name."""
    name = "Mango, pulp, raw (Brazil by plane), processed in FR | ... | at supermarket {FR} U"
    assert parser.location_for(name) == "FR"


# ----------------------------------------------------------------------------
# Trailing-token fallback


def test_trailing_two_letter_iso(parser):
    assert parser.location_for("mango production BR") == "BR"


def test_trailing_three_letter_aggregate_returns_none(parser):
    """``GLO`` / ``RoW`` etc. are aggregates with no national fallback
    — the parser drops them so the scorer keeps the global CF."""
    assert parser.location_for("market for sugar, from sugar beet GLO") is None
    assert parser.location_for("market for heat, ... RoW") is None
    assert parser.location_for("market group for electricity ... RER") is None


def test_trailing_after_chemical_symbol_uses_last_word(parser):
    """``"... as N BR"`` should resolve to ``BR``, NOT ``N`` — the
    length-2 floor on the regex captures the ISO and skips the symbol.
    """
    assert parser.location_for("market for inorganic nitrogen fertiliser, as N BR") == "BR"


def test_trailing_with_u_suffix_strips_first(parser):
    """``"... GLO U"`` resolves to ``GLO`` (then to None as aggregate)."""
    assert parser.location_for("market for cobalt GLO U") is None


# ----------------------------------------------------------------------------
# Subregional collapse


def test_subregional_collapses_to_country(parser):
    assert parser.location_for("market for electricity, CA-QC") == "CA"
    assert parser.location_for("hard coal {US-WECC} U") == "US"
    assert parser.location_for("dairy farm BR-MG") == "BR"


# ----------------------------------------------------------------------------
# No location


def test_returns_none_when_no_recognisable_suffix(parser):
    # No trailing uppercase token of 2+ chars. ``transport`` is
    # lowercase, ``EU mix`` isn't at the end. Truly nothing.
    assert parser.location_for("transport, freight, lorry, with cooled compartment") is None
    assert parser.location_for("a single word") is None


def test_returns_none_for_empty_or_none(parser):
    assert parser.location_for(None) is None
    assert parser.location_for("") is None


def test_returns_none_for_single_letter_token(parser):
    """A trailing single letter (e.g. ``"… as N"``) must not resolve to
    a location."""
    assert parser.location_for("market for inorganic nitrogen, as N") is None


def test_complex_processing_chain_picks_processing_location(parser):
    """For names like ``"... | at packaging {FR} U"`` we want the
    last bracketed location — that's the activity's own location."""
    name = "Mango juice, fresh, processed in FR | Ambient (long) | Pack proxy | at packaging {FR} U"
    assert parser.location_for(name) == "FR"
