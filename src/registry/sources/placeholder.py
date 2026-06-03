"""Source ingesters for ``placeholder_flow_classification.xlsx``.

Four sheets, each its own class:

* ``PlaceholderEcoinventSource`` — sheet ``match with ecoinvent v3.9.1`` (806
  rows). Wired into tier 1 (fix 1.a). Currently unused by the legacy linker.
* ``PlaceholderEfNativeSource`` — sheet ``match with EF v3.1`` (643 rows)
  → tier 6.
* ``PlaceholderUnmatchableSource`` — sheet ``Neither in ecoinvent nor EF``
  (193 rows) → ``is_unmatchable=True``.
* ``PlaceholderTransitiveSource`` — sheet ``ecoinvent flows - EF v3.1 map``
  (806 rows). Off by default; emitted only when
  ``Settings.apply_transitive_layer`` is true.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from domain import Bucket, Mapping, SourceKind, Tier
from readers import XlsxReader


def _parse_cats(value: object) -> tuple[str, ...]:
    """Parse a stringified tuple-of-categories from xlsx into a tuple.

    Hidden helper only — not exposed; lives next to the only callers.
    Lifted into a class would be ceremonial.
    """
    if not isinstance(value, str):
        return ()
    try:
        t = ast.literal_eval(value)
        return tuple(str(x) for x in t) if isinstance(t, tuple) else ()
    except (ValueError, SyntaxError):
        return ()


@dataclass(frozen=True)
class _PlaceholderSheetSource:
    """Base behaviour shared by all four placeholder source classes.

    Not an ABC, not part of the public API — just a tiny shared helper for
    the sheet-reading + DataFrame iteration. Concrete subclasses set the
    sheet name + tier and implement ``_row_to_mapping``.
    """

    xlsx: XlsxReader
    xlsx_path: Path
    sheet_name: str
    tier: Tier
    provenance: str
    is_unmatchable: bool = False

    def read(self) -> list[Mapping]:
        df = self.xlsx.read(self.xlsx_path, sheet_name=self.sheet_name)
        return list(self._iter_mappings(df))

    def _iter_mappings(self, df: pd.DataFrame) -> Iterable[Mapping]:
        raise NotImplementedError


@dataclass(frozen=True)
class PlaceholderEcoinventSource(_PlaceholderSheetSource):
    """Sheet ``match with ecoinvent v3.9.1`` — AGB → ecoinvent biosphere flows.

    Defaults the tier to ``CURATED_TARGETED`` (tier 1) per fix 1.a.
    """

    sheet_name: str = "match with ecoinvent v3.9.1"
    tier: Tier = Tier.CURATED_TARGETED
    provenance: str = "placeholder.match_with_ecoinvent_v3.9.1"

    def _iter_mappings(self, df: pd.DataFrame) -> Iterable[Mapping]:
        for idx, r in df.iterrows():
            src_name = str(r.get("name", "")).strip()
            if not src_name:
                continue
            yield Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name=src_name,
                source_unit=str(r.get("unit", "")).strip(),
                source_context=_parse_cats(r.get("categories")),
                source_top_bucket=Bucket.from_categories(_parse_cats(r.get("categories"))),
                source_cas=_clean_cas(r.get("cas")),
                target_db="",  # resolved at match time against biosphere3 / ei-3.x-biosphere
                target_code="",
                target_name=str(r.get("matched_ecoinvent_name", "")).strip(),
                target_unit="",
                priority_tier=self.tier,
                provenance=self.provenance,
                provenance_row=str(idx),
                notes=str(r.get("ecoinvent_match_type", "")).strip(),
            )


@dataclass(frozen=True)
class PlaceholderEfNativeSource(_PlaceholderSheetSource):
    """Sheet ``match with EF v3.1`` — AGB → EF flow UUIDs (resolved against the EF parquet)."""

    sheet_name: str = "match with EF v3.1"
    tier: Tier = Tier.EF_PLACEHOLDER
    provenance: str = "placeholder.match_with_EF_v3.1"

    def _iter_mappings(self, df: pd.DataFrame) -> Iterable[Mapping]:
        for idx, r in df.iterrows():
            src_name = str(r.get("name", "")).strip()
            if not src_name:
                continue
            yield Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name=src_name,
                source_unit=str(r.get("unit", "")).strip(),
                source_context=_parse_cats(r.get("categories")),
                source_top_bucket=Bucket.from_categories(_parse_cats(r.get("categories"))),
                source_cas=_clean_cas(r.get("cas")),
                target_db="",  # resolved against EF parquet at registry-build time by enrichment pass
                target_code="",
                target_name=str(r.get("matched_ef_name", "")).strip(),
                target_unit="",
                priority_tier=self.tier,
                provenance=self.provenance,
                provenance_row=str(idx),
                notes=str(r.get("ef_match_type", "")).strip(),
            )


@dataclass(frozen=True)
class PlaceholderUnmatchableSource(_PlaceholderSheetSource):
    """Sheet ``Neither in ecoinvent nor EF`` — known-unmatchable AGB flows."""

    sheet_name: str = "Neither in ecoinvent nor EF"
    tier: Tier = Tier.UNMATCHABLE
    provenance: str = "placeholder.neither"
    is_unmatchable: bool = True

    def _iter_mappings(self, df: pd.DataFrame) -> Iterable[Mapping]:
        for idx, r in df.iterrows():
            src_name = str(r.get("name", "")).strip()
            if not src_name:
                continue
            yield Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name=src_name,
                source_unit=str(r.get("unit", "")).strip(),
                source_context=_parse_cats(r.get("categories")),
                source_top_bucket=Bucket.from_categories(_parse_cats(r.get("categories"))),
                source_cas=_clean_cas(r.get("cas")),
                priority_tier=self.tier,
                provenance=self.provenance,
                provenance_row=str(idx),
                is_unmatchable=True,
            )


@dataclass(frozen=True)
class PlaceholderTransitiveSource(_PlaceholderSheetSource):
    """Sheet ``ecoinvent flows - EF v3.1 map`` — transitive ecoinvent → EF map.

    Off by default; emitted only when ``Settings.apply_transitive_layer`` is true.
    """

    sheet_name: str = "ecoinvent flows - EF v3.1 map"
    tier: Tier = Tier.CURATED_TARGETED
    provenance: str = "placeholder.transitive_ecoinvent_ef"

    def _iter_mappings(self, df: pd.DataFrame) -> Iterable[Mapping]:
        for idx, r in df.iterrows():
            src_name = str(r.get("name", "")).strip()
            if not src_name:
                continue
            yield Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name=src_name,
                source_unit=str(r.get("unit", "")).strip(),
                source_context=_parse_cats(r.get("categories")),
                source_top_bucket=Bucket.from_categories(_parse_cats(r.get("categories"))),
                source_cas=_clean_cas(r.get("cas")),
                target_db="",
                target_code="",
                target_name=str(r.get("matched_ef_name", "")).strip(),
                priority_tier=self.tier,
                provenance=self.provenance,
                provenance_row=str(idx),
                notes=f"transitive via {str(r.get('matched_ecoinvent_name', '')).strip()}",
            )


def _clean_cas(value: object) -> str | None:
    """Return a canonical CAS string or None.

    Strips whitespace, NaN sentinels, and string ``'nan'``. CAS comparisons
    later are exact so we keep the original digits including leading zeros.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s
