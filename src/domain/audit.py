"""Audit-trail data classes. One row per override / drop / suppression."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from domain.tier import Tier


class OverrideKind(StrEnum):
    NEW_LINK = "new_link"
    """A previously-unlinked exchange was assigned a target."""

    OVERRIDE = "override"
    """A higher-priority tier displaced an existing link."""

    SAME_TIER_OVERRIDE = "same_tier_override"
    """A row at the same tier displaced another. Always logged for review."""

    REJECTED_LOWER_TIER = "rejected_lower_tier"
    """A row from a lower-priority tier was prevented from claiming a link."""

    REJECTED_UNIT_MISMATCH = "rejected_unit_mismatch"
    """A candidate was skipped because units didn't match and no conversion exists (fix 1.e)."""

    SKIPPED_AMBIGUOUS = "skipped_ambiguous"
    """Multiple equally-valid candidates with no deterministic tie-breaker (fix 1.f)."""


@dataclass(frozen=True)
class AuditEntry:
    """One row of ``override_audit.parquet``."""

    process_name: str
    exchange_name: str
    exchange_unit: str
    exchange_bucket: str
    kind: OverrideKind
    new_tier: Tier | None
    new_target_db: str
    new_target_code: str
    new_provenance: str
    prior_tier: Tier | None = None
    prior_target_db: str = ""
    prior_target_code: str = ""
    reason: str = ""
    unit_conversion: float = 1.0


AUDIT_COLUMNS: tuple[str, ...] = (
    "process_name",
    "exchange_name",
    "exchange_unit",
    "exchange_bucket",
    "kind",
    "new_tier",
    "new_target_db",
    "new_target_code",
    "new_provenance",
    "prior_tier",
    "prior_target_db",
    "prior_target_code",
    "reason",
    "unit_conversion",
)


@dataclass(frozen=True)
class SuppressedStrategy:
    """One row of ``suppressed_strategies.parquet`` — fix 1.g.

    Replaces silent ``contextlib.suppress(Exception)`` over the bw2io
    strategy chain. Captures what threw so the run report can surface it.
    """

    strategy: str
    error_type: str
    error_message: str
    occurred_at_step: str = ""


@dataclass(frozen=True)
class DropEvent:
    """One row of the drop tally — fix 1.i.

    Counts ``drop_final_waste_flows`` and
    ``drop_zero_amount_unlinked_biosphere`` invocations + per-call removed
    counts so the report can report drops per strategy.
    """

    strategy: str
    n_dropped: int
    process_name: str = ""
