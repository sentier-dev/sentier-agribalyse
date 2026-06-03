"""Lookup indexes used by the matchers.

Built once per run from the loaded ``MappingRegistry``. Each index is a
small class with a ``.lookup(...)`` method, so swapping one out (e.g.
adding a CAS-based variant) is a single class addition rather than a
restructure of the matcher.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from domain import Bucket, Tier


@dataclass(frozen=True)
class _IndexEntry:
    """One candidate target for a lookup hit."""

    target_db: str
    target_code: str
    target_name: str
    target_unit: str
    unit_conversion: float
    tier: Tier
    provenance: str
    is_unmatchable: bool
    source_unit: str = ""
    source_name: str = ""

    @staticmethod
    def _str(value) -> str:
        """NaN/None-safe string coercion for parquet-roundtripped cells."""
        if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
            return ""
        return str(value)

    @classmethod
    def _conv(cls, value) -> float:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
            return 1.0
        return float(value)

    @classmethod
    def from_row(cls, r) -> _IndexEntry:
        return cls(
            target_db=cls._str(r.target_db),
            target_code=cls._str(r.target_code),
            target_name=cls._str(r.target_name),
            target_unit=cls._str(r.target_unit),
            unit_conversion=cls._conv(r.unit_conversion),
            tier=Tier(int(r.priority_tier)),
            provenance=cls._str(r.provenance),
            is_unmatchable=bool(r.is_unmatchable),
            source_unit=cls._str(getattr(r, "source_unit", "")),
            source_name=cls._str(getattr(r, "source_name", "")),
        )


@dataclass(frozen=True)
class TieredNameBucketIndex:
    """``(source_kind, name_lower, bucket) → [entry, …]`` ordered by tier ascending.

    The biosphere matcher walks tiers ascending and grabs the first entry
    whose constraints are satisfied. Same key ⇒ candidates list ordered
    by tier, then by provenance for deterministic tie-breaks.
    """

    df: pd.DataFrame

    @cached_property
    def _by_key(self) -> dict[tuple[str, str, str], list[_IndexEntry]]:
        out: dict[tuple[str, str, str], list[_IndexEntry]] = defaultdict(list)
        if self.df.empty:
            return dict(out)
        for r in self.df.itertuples(index=False):
            name = _IndexEntry._str(r.source_name).strip().lower()
            key = (
                _IndexEntry._str(r.source_kind),
                name,
                _IndexEntry._str(r.source_top_bucket),
            )
            out[key].append(_IndexEntry.from_row(r))
        # Stable ordering: tier ASC, then provenance ASC, then target_code ASC.
        for _k, v in out.items():
            v.sort(key=lambda e: (int(e.tier), e.provenance, e.target_code, e.target_name))
        return dict(out)

    @cached_property
    def _by_kind_name(self) -> dict[tuple[str, str], list[_IndexEntry]]:
        """``(source_kind, name_lower) → all entries regardless of bucket``.

        Used as the last-resort fallback for ``lookup`` when neither the
        exchange bucket nor UNSPECIFIED bucket has a candidate. Surfaces
        rows whose source bucket differs from the exchange's — e.g. a
        placeholder mapping pinned to soil bucket can still apply to an
        air-bucket exchange of the same source name (the matcher's
        ``_resolve_target`` does the cross-compartment target work).
        """
        out: dict[tuple[str, str], list[_IndexEntry]] = defaultdict(list)
        for (kind, name, _bucket), entries in self._by_key.items():
            out[(kind, name)].extend(entries)
        for _k, v in out.items():
            v.sort(key=lambda e: (int(e.tier), e.provenance, e.target_code, e.target_name))
        return dict(out)

    def lookup(
        self,
        source_kind: str,
        name_lower: str,
        bucket: Bucket | str,
    ) -> list[_IndexEntry]:
        b = bucket.value if isinstance(bucket, Bucket) else str(bucket)
        direct = self._by_key.get((source_kind, name_lower, b), [])
        if b == Bucket.UNSPECIFIED.value:
            return direct
        # Bucket-agnostic mappings (curated synonym fallback, randonneur
        # SimaPro biosphere manual matches) are stored under bucket=UNSPECIFIED.
        agnostic = self._by_key.get((source_kind, name_lower, Bucket.UNSPECIFIED.value), [])
        if direct or agnostic:
            return direct + agnostic
        # Last-resort fallback: surface candidates with the same name regardless
        # of source bucket. This covers placeholder/curated mappings that were
        # filed under one specific compartment but whose source flow appears in
        # several. The matcher's ``_resolve_target`` then handles target-side
        # compartment resolution (and refuses cross-drift for UUID-pinned rows).
        return self._by_kind_name.get((source_kind, name_lower), [])


@dataclass(frozen=True)
class CasIndex:
    """``cas → [entry, …]`` for CAS-disambiguation lookups (fix 1.o)."""

    df: pd.DataFrame

    @cached_property
    def _by_cas(self) -> dict[str, list[_IndexEntry]]:
        out: dict[str, list[_IndexEntry]] = defaultdict(list)
        if self.df.empty:
            return dict(out)
        for r in self.df.itertuples(index=False):
            cas = _IndexEntry._str(r.source_cas).strip()
            if not cas:
                continue
            out[cas].append(_IndexEntry.from_row(r))
        for _k, v in out.items():
            v.sort(key=lambda e: (int(e.tier), e.provenance, e.target_code))
        return dict(out)

    def lookup(self, cas: str) -> list[_IndexEntry]:
        return self._by_cas.get(cas.strip(), [])


@dataclass(frozen=True)
class UnitConverter:
    """Resolve a source unit → target unit multiplier from the registry.

    Aliases (case-insensitive) are applied first so ``a`` → ``year`` →
    canonical conversion happens transparently.

    The two randonneur sources we ingest disagree on canonical naming:
    ``flowmapper-standard-units-harmonization`` uses snake_case
    (``cubic_meter``, ``kilobecquerel``) while
    ``generic-brightway-unit-conversions`` (and Brightway itself) uses
    space-separated forms (``cubic meter``, ``kilo Becquerel``). The
    conversion table is keyed in the brightway convention; an alias that
    rewrites a brightway-canonical unit to flowmapper's snake_case form
    silently breaks every downstream lookup. ``_alias_to_canonical``
    therefore (a) skips aliases whose alias is already a known canonical,
    and (b) rewrites snake_case alias targets to their space-separated
    brightway equivalent when that form is in the conversion table.
    """

    conversions_df: pd.DataFrame
    aliases_df: pd.DataFrame

    @cached_property
    def _known_canonicals(self) -> frozenset[str]:
        """Lowercased units that appear as a source or target in the conversion table."""
        if self.conversions_df.empty:
            return frozenset()
        return frozenset(self.conversions_df["source_unit"].astype(str).str.lower()) | frozenset(
            self.conversions_df["target_unit"].astype(str).str.lower()
        )

    @cached_property
    def _alias_to_canonical(self) -> dict[str, str]:
        if self.aliases_df.empty:
            return {}
        known = self._known_canonicals
        out: dict[str, str] = {}
        for row in self.aliases_df.itertuples(index=False):
            alias = row.alias_lower
            target = str(row.canonical)
            target_l = target.strip().lower()
            # (a) Don't rename a unit that's already a known canonical form;
            # any rewrite would just strand it (e.g. ``kilo becquerel`` →
            # ``kilobecquerel``).
            if alias in known:
                continue
            # (b) If the target is snake_case but the space-separated form is
            # known, prefer the space-separated form so downstream conversion
            # lookups hit the brightway-keyed table.
            if target_l not in known and "_" in target_l:
                spaced = target_l.replace("_", " ")
                if spaced in known:
                    target = spaced
            out[alias] = target
        return out

    @cached_property
    def _conversion(self) -> dict[tuple[str, str], float]:
        if self.conversions_df.empty:
            return {}
        return {
            (row.source_unit.lower(), row.target_unit.lower()): float(row.multiplier)
            for row in self.conversions_df.itertuples(index=False)
        }

    def canonical(self, unit: str) -> str:
        """Apply alias normalisation: ``a`` → ``year`` etc."""
        if not unit:
            return ""
        return self._alias_to_canonical.get(unit.strip().lower(), unit)

    def multiplier(self, source_unit: str, target_unit: str) -> float | None:
        """Return ``multiplier`` such that ``target = source * multiplier``.

        Returns ``1.0`` if the canonical units already match, ``None`` if
        no conversion is registered.
        """
        src = self.canonical(source_unit).strip().lower()
        tgt = self.canonical(target_unit).strip().lower()
        if not src or not tgt:
            return None
        if src == tgt:
            return 1.0
        return self._conversion.get((src, tgt))


@dataclass(frozen=True)
class UnmatchableIndex:
    """``(name_lower, bucket) → True`` for known-unmatchable AGB flows."""

    df: pd.DataFrame

    @cached_property
    def _set(self) -> set[tuple[str, str]]:
        if self.df.empty:
            return set()
        return {
            (
                _IndexEntry._str(r.source_name).strip().lower(),
                _IndexEntry._str(r.source_top_bucket) or Bucket.UNSPECIFIED.value,
            )
            for r in self.df.itertuples(index=False)
        }

    def is_unmatchable(self, name: str, bucket: Bucket | str) -> bool:
        b = bucket.value if isinstance(bucket, Bucket) else str(bucket)
        return ((name or "").strip().lower(), b) in self._set
