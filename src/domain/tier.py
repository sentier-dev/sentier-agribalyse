"""Priority tiers for mapping resolution. Lower number = higher priority.

Mirrors REFACTOR.md §0.3. The matcher walks tiers in ascending order and
selects the highest-priority row that satisfies type/unit/context
constraints. Tiers >= ``EF_GENERIC`` are *fill-only*: they may only place
links onto exchanges that have no prior link.
"""

from __future__ import annotations

from enum import IntEnum


class Tier(IntEnum):
    """Mapping priority tier (lower = higher priority)."""

    CURATED_TARGETED = 1
    """Curated AGB-specific overrides — placeholder 'match with ecoinvent v3.9.1'
    sheet (806 rows) + the small dated ``curated_overrides.json``."""

    RANDONNEUR_AGB_SPECIFIC = 2
    """``agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches.json`` (96 rows).
    Replaces the residual hardcoded ``BIOSPHERE_SYNONYMS``."""

    RANDONNEUR_SIMAPRO_BIO = 3
    """``simapro-9-ecoinvent-3.9-biosphere-manual-matches.json`` (580 rows)."""

    AGRIBALYSE_EI_BIOSPHERE = 4
    """``agribalyse-3.2-ecoinvent-3.10-biosphere.json`` — Flowmapper-generated
    SimaPro-9 → ecoinvent-3.10-biosphere mappings (~4 039 rows). UUID-pinned."""

    HARMONISED_FLOWS = 5
    """The brightway-style harmonised flow registry."""

    RANDONNEUR_WATER_M3 = 6
    """``simapro-9-ecoinvent-3-water-slash-m3.gz`` (~39 675 rows). Also the home
    of CAS-derived index entries used for disambiguation (fix 1.o) — same
    confidence band, distinguished by ``provenance``."""

    EF_PLACEHOLDER = 7
    """Placeholder 'match with EF v3.1' sheet (643 rows). Override-capable."""

    EF_GENERIC = 8
    """``(name, bucket, unit)`` lookup against the full EF parquet. Fill-only."""

    BIO3_MATCH_DATABASE = 9
    """bw2io ``match_database`` chain against biosphere3. Fill-only."""

    CASE_INSENSITIVE_FALLBACK = 10
    """``(name_lower, unit, bucket)`` fallback. Fill-only, deterministic
    tie-breaker (fix 1.f)."""

    LLM_OVERRIDES = 11
    """LLM-reviewed accept rows. Fill-only, gated by ``--no-llm`` (fix 1.j)."""

    CURATED_SYNONYM_FALLBACK = 12
    """The legacy ``BIOSPHERE_SYNONYMS`` residue, now living in
    ``curated_overrides.json`` rows tagged as ``synonym``. Fill-only, same
    ``--no-llm`` gate as tier 11."""

    UNMATCHABLE = 13
    """Known-unmatchable AGB flows. Never produces a link; suppresses warnings."""

    @classmethod
    def fill_only_tiers(cls) -> set[Tier]:
        """Tiers that may only fill empty links — never override."""
        return {
            cls.EF_GENERIC,
            cls.BIO3_MATCH_DATABASE,
            cls.CASE_INSENSITIVE_FALLBACK,
            cls.LLM_OVERRIDES,
            cls.CURATED_SYNONYM_FALLBACK,
        }

    def is_fill_only(self) -> bool:
        return self in self.fill_only_tiers()

    @classmethod
    def llm_gated_tiers(cls) -> set[Tier]:
        """Tiers gated by ``Settings.apply_llm_overrides`` (fix 1.j)."""
        return {cls.LLM_OVERRIDES, cls.CURATED_SYNONYM_FALLBACK}

    def is_llm_gated(self) -> bool:
        return self in self.llm_gated_tiers()
