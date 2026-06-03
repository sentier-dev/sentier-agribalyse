"""Pure data classes — frozen dataclasses, IntEnums, value objects.

No behavior here, just shapes. Behavior lives in classes under
``matching/``, ``registry/``, ``transforms/``, etc.
"""

from domain.audit import (
    AUDIT_COLUMNS,
    AuditEntry,
    DropEvent,
    OverrideKind,
    SuppressedStrategy,
)
from domain.bucket import Bucket
from domain.mapping import (
    MAPPING_COLUMNS,
    ContextNorm,
    Deletion,
    EdgeLabelCorrection,
    EfTargetIndexRow,
    Mapping,
    SourceKind,
    UnitAlias,
    UnitConversion,
)
from domain.match import MatchOutcome, TierStats
from domain.tier import Tier

__all__ = [
    "AUDIT_COLUMNS",
    "MAPPING_COLUMNS",
    "AuditEntry",
    "Bucket",
    "ContextNorm",
    "Deletion",
    "DropEvent",
    "EdgeLabelCorrection",
    "EfTargetIndexRow",
    "Mapping",
    "MatchOutcome",
    "OverrideKind",
    "SourceKind",
    "SuppressedStrategy",
    "Tier",
    "TierStats",
    "UnitAlias",
    "UnitConversion",
]
