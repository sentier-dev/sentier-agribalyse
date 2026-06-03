"""Registry row schema. One ``Mapping`` = one row of a ``mappings_*`` parquet."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from domain.bucket import Bucket
from domain.tier import Tier


class SourceKind(StrEnum):
    AGB_FLOW = "agb_flow"
    """Biosphere exchange (substance, compartment, unit)."""

    AGB_NODE = "agb_node"
    """Technosphere process or product."""


@dataclass(frozen=True)
class Mapping:
    """One row of the mapping registry.

    Schema mirrors REFACTOR.md §0.1. Fields shared across biosphere /
    technosphere; the ``source_kind`` discriminator tells the matcher
    which side the row applies to.
    """

    source_kind: SourceKind
    source_name: str
    source_unit: str
    source_context: tuple[str, ...] = ()
    source_top_bucket: Bucket = Bucket.UNSPECIFIED
    source_cas: str | None = None
    source_formula: str | None = None

    target_db: str = ""
    target_code: str = ""
    target_name: str = ""
    target_unit: str = ""

    unit_conversion: float = 1.0
    """Scalar multiplier ``source → target``. ``1.0`` means units already match."""

    priority_tier: Tier = Tier.CURATED_TARGETED
    provenance: str = ""
    provenance_row: str = ""

    is_unmatchable: bool = False
    notes: str = ""

    # ------------------------------------------------------------------
    # Equality helpers used when one row would override another.

    @property
    def target_key(self) -> tuple[str, str]:
        """``(db, code)`` — what bw2data uses to identify the target."""
        return (self.target_db, self.target_code)

    @property
    def name_lower(self) -> str:
        return self.source_name.strip().lower()


# DataFrame schema — column order used when materialising parquets.
MAPPING_COLUMNS: tuple[str, ...] = (
    "source_kind",
    "source_name",
    "source_unit",
    "source_context",
    "source_top_bucket",
    "source_cas",
    "source_formula",
    "target_db",
    "target_code",
    "target_name",
    "target_unit",
    "unit_conversion",
    "priority_tier",
    "provenance",
    "provenance_row",
    "is_unmatchable",
    "notes",
)


@dataclass(frozen=True)
class UnitConversion:
    """One row of ``unit_conversions.parquet``."""

    source_unit: str
    target_unit: str
    multiplier: float
    provenance: str = ""


@dataclass(frozen=True)
class UnitAlias:
    """One row of ``unit_aliases.parquet`` — case-insensitive."""

    alias: str
    canonical: str
    provenance: str = ""


@dataclass(frozen=True)
class ContextNorm:
    """One row of ``context_normalisation.parquet``."""

    source_context: tuple[str, ...]
    target_context: tuple[str, ...]
    provenance: str = ""


@dataclass(frozen=True)
class Deletion:
    """One row of ``deletions.parquet`` — process or product to remove."""

    name: str
    code: str | None = None
    kind: str = "process"
    """``process`` or ``product`` — which AGB graph element to delete."""

    provenance: str = ""


@dataclass(frozen=True)
class EdgeLabelCorrection:
    """One row of ``edge_label_corrections.parquet``."""

    source_name: str
    target_name: str
    edge_type: str = ""
    categories: tuple[str, ...] = ()
    provenance: str = ""


@dataclass(frozen=True)
class EfTargetIndexRow:
    """One row of ``target_index_ef.parquet`` — EF flow universe."""

    code: str
    name: str
    unit: str
    bucket: Bucket
    categories: tuple[str, ...] = ()
    cas: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
