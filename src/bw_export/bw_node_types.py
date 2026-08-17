"""Node key + resolved-label value types shared by the bw2 export path.

The label value types are produced by
:class:`bw_export.catalog_key_resolver.CatalogKeyResolver` and consumed by
:class:`bw_export.metadata_emitter.MetadataEmitter`, so they keep a small
dependency-free home here.
"""

from __future__ import annotations

from dataclasses import dataclass

# A node key: (database, code). Kept as a tuple in memory; serialized as a
# 2-element list wherever it crosses a JSON/parquet boundary.
NodeKey = tuple[str, str]


@dataclass(frozen=True)
class ActivityMeta:
    """Resolved label metadata for one technosphere column."""

    key: NodeKey
    name: str
    unit: str
    location: str
    reference_product: str


@dataclass(frozen=True)
class BioMeta:
    """Resolved label metadata for one biosphere row."""

    key: NodeKey
    name: str
    categories: tuple[str, ...]
    unit: str
    is_synthetic_correction: bool
