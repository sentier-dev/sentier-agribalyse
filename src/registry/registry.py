"""Loaded ``MappingRegistry`` — the read API consumed by every matcher.

Construction::

    reg = MappingRegistry.load(settings)

The registry holds DataFrames in memory and lazily builds indexes on
first access. Indexes are ``cached_property`` on the index classes, so a
second matcher reuses the same index instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import pandas as pd

from config import Settings
from registry.indexes import (
    CasIndex,
    TieredNameBucketIndex,
    UnitConverter,
    UnmatchableIndex,
)


@dataclass(frozen=True)
class MappingRegistry:
    """Loaded view of the ``registry/*.parquet`` files."""

    settings: Settings
    mappings_biosphere: pd.DataFrame
    mappings_technosphere: pd.DataFrame
    unmatchable: pd.DataFrame
    unit_conversions: pd.DataFrame
    unit_aliases: pd.DataFrame
    context_normalisation: pd.DataFrame
    deletions: pd.DataFrame
    edge_label_corrections: pd.DataFrame
    target_index_ef: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction.

    @classmethod
    def load(cls, settings: Settings) -> MappingRegistry:
        p = settings.paths

        def _read_or_empty(path) -> pd.DataFrame:
            return pd.read_parquet(path) if path.exists() else pd.DataFrame()

        meta: dict[str, Any] = {}
        if p.registry_meta.exists():
            meta = json.loads(p.registry_meta.read_text())

        return cls(
            settings=settings,
            mappings_biosphere=_read_or_empty(p.registry_mappings_biosphere),
            mappings_technosphere=_read_or_empty(p.registry_mappings_technosphere),
            unmatchable=_read_or_empty(p.registry_unmatchable),
            unit_conversions=_read_or_empty(p.registry_unit_conversions),
            unit_aliases=_read_or_empty(p.registry_unit_aliases),
            context_normalisation=_read_or_empty(p.registry_context_normalisation),
            deletions=_read_or_empty(p.registry_deletions),
            edge_label_corrections=_read_or_empty(p.registry_edge_label_corrections),
            target_index_ef=_read_or_empty(p.registry_target_index_ef),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Indexes (lazy).

    @cached_property
    def biosphere_index(self) -> TieredNameBucketIndex:
        return TieredNameBucketIndex(df=self.mappings_biosphere)

    @cached_property
    def technosphere_index(self) -> TieredNameBucketIndex:
        return TieredNameBucketIndex(df=self.mappings_technosphere)

    @cached_property
    def biosphere_cas_index(self) -> CasIndex:
        return CasIndex(df=self.mappings_biosphere)

    @cached_property
    def unit_converter(self) -> UnitConverter:
        return UnitConverter(
            conversions_df=self.unit_conversions,
            aliases_df=self.unit_aliases,
        )

    @cached_property
    def unmatchable_index(self) -> UnmatchableIndex:
        return UnmatchableIndex(df=self.unmatchable)

    # ------------------------------------------------------------------
    # Convenience accessors used by exporters.

    @property
    def total_biosphere_mappings(self) -> int:
        return len(self.mappings_biosphere)

    @property
    def total_technosphere_mappings(self) -> int:
        return len(self.mappings_technosphere)
