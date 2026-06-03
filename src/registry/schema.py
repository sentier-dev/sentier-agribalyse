"""Schema helpers: turn lists of frozen-dataclass rows into pandas DataFrames.

Class-based: ``MappingTable`` / ``UnitConversionTable`` etc. wrap the
column ordering and type coercion for one parquet file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from domain import (
    AUDIT_COLUMNS,
    MAPPING_COLUMNS,
    AuditEntry,
    ContextNorm,
    Deletion,
    EdgeLabelCorrection,
    EfTargetIndexRow,
    Mapping,
    UnitAlias,
    UnitConversion,
)


@dataclass(frozen=True)
class MappingTable:
    """Materialise ``Mapping`` rows into a DataFrame with stable column order."""

    columns: tuple[str, ...] = MAPPING_COLUMNS

    def to_dataframe(self, rows: Sequence[Mapping]) -> pd.DataFrame:
        records = []
        for m in rows:
            records.append(
                {
                    "source_kind": m.source_kind.value,
                    "source_name": m.source_name,
                    "source_unit": m.source_unit,
                    "source_context": list(m.source_context),
                    "source_top_bucket": m.source_top_bucket.value,
                    "source_cas": m.source_cas,
                    "source_formula": m.source_formula,
                    "target_db": m.target_db,
                    "target_code": m.target_code,
                    "target_name": m.target_name,
                    "target_unit": m.target_unit,
                    "unit_conversion": m.unit_conversion,
                    "priority_tier": int(m.priority_tier),
                    "provenance": m.provenance,
                    "provenance_row": m.provenance_row,
                    "is_unmatchable": m.is_unmatchable,
                    "notes": m.notes,
                }
            )
        if not records:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(records, columns=list(self.columns))


@dataclass(frozen=True)
class UnitConversionTable:
    columns: tuple[str, ...] = ("source_unit", "target_unit", "multiplier", "provenance")

    def to_dataframe(self, rows: Sequence[UnitConversion]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(
            [{c: getattr(r, c) for c in self.columns} for r in rows],
            columns=list(self.columns),
        )


@dataclass(frozen=True)
class UnitAliasTable:
    columns: tuple[str, ...] = ("alias", "alias_lower", "canonical", "provenance")

    def to_dataframe(self, rows: Sequence[UnitAlias]) -> pd.DataFrame:
        records = [
            {
                "alias": r.alias,
                "alias_lower": r.alias.lower(),
                "canonical": r.canonical,
                "provenance": r.provenance,
            }
            for r in rows
        ]
        if not records:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(records, columns=list(self.columns))


@dataclass(frozen=True)
class ContextNormTable:
    columns: tuple[str, ...] = ("source_context", "target_context", "provenance")

    def to_dataframe(self, rows: Sequence[ContextNorm]) -> pd.DataFrame:
        records = [
            {
                "source_context": list(r.source_context),
                "target_context": list(r.target_context),
                "provenance": r.provenance,
            }
            for r in rows
        ]
        if not records:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(records, columns=list(self.columns))


@dataclass(frozen=True)
class DeletionTable:
    columns: tuple[str, ...] = ("name", "code", "kind", "provenance")

    def to_dataframe(self, rows: Sequence[Deletion]) -> pd.DataFrame:
        records = [
            {"name": r.name, "code": r.code, "kind": r.kind, "provenance": r.provenance}
            for r in rows
        ]
        if not records:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(records, columns=list(self.columns))


@dataclass(frozen=True)
class EdgeLabelTable:
    columns: tuple[str, ...] = (
        "source_name",
        "target_name",
        "edge_type",
        "categories",
        "provenance",
    )

    def to_dataframe(self, rows: Sequence[EdgeLabelCorrection]) -> pd.DataFrame:
        records = [
            {
                "source_name": r.source_name,
                "target_name": r.target_name,
                "edge_type": r.edge_type,
                "categories": list(r.categories),
                "provenance": r.provenance,
            }
            for r in rows
        ]
        if not records:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(records, columns=list(self.columns))


@dataclass(frozen=True)
class EfTargetIndexTable:
    columns: tuple[str, ...] = ("code", "name", "name_lower", "unit", "bucket", "categories", "cas")

    def to_dataframe(self, rows: Sequence[EfTargetIndexRow]) -> pd.DataFrame:
        records = [
            {
                "code": r.code,
                "name": r.name,
                "name_lower": r.name.strip().lower(),
                "unit": r.unit,
                "bucket": r.bucket.value,
                "categories": list(r.categories),
                "cas": r.cas,
            }
            for r in rows
        ]
        if not records:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(records, columns=list(self.columns))


@dataclass(frozen=True)
class AuditTable:
    columns: tuple[str, ...] = AUDIT_COLUMNS

    def to_dataframe(self, rows: Sequence[AuditEntry]) -> pd.DataFrame:
        records = [
            {
                "process_name": e.process_name,
                "exchange_name": e.exchange_name,
                "exchange_unit": e.exchange_unit,
                "exchange_bucket": e.exchange_bucket,
                "kind": e.kind.value,
                "new_tier": int(e.new_tier) if e.new_tier is not None else None,
                "new_target_db": e.new_target_db,
                "new_target_code": e.new_target_code,
                "new_provenance": e.new_provenance,
                "prior_tier": int(e.prior_tier) if e.prior_tier is not None else None,
                "prior_target_db": e.prior_target_db,
                "prior_target_code": e.prior_target_code,
                "reason": e.reason,
                "unit_conversion": e.unit_conversion,
            }
            for e in rows
        ]
        if not records:
            return pd.DataFrame(columns=list(self.columns))
        return pd.DataFrame(records, columns=list(self.columns))
