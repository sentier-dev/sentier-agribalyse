"""Match-outcome value objects emitted by the matchers."""

from __future__ import annotations

from dataclasses import dataclass

from domain.tier import Tier


@dataclass(frozen=True)
class MatchOutcome:
    """Result of attempting to match one exchange against the registry."""

    matched: bool
    tier: Tier | None = None
    target_db: str = ""
    target_code: str = ""
    target_unit: str = ""
    unit_conversion: float = 1.0
    provenance: str = ""
    skipped_reason: str = ""

    @classmethod
    def hit(
        cls,
        *,
        tier: Tier,
        target_db: str,
        target_code: str,
        target_unit: str,
        unit_conversion: float = 1.0,
        provenance: str = "",
    ) -> MatchOutcome:
        return cls(
            matched=True,
            tier=tier,
            target_db=target_db,
            target_code=target_code,
            target_unit=target_unit,
            unit_conversion=unit_conversion,
            provenance=provenance,
        )

    @classmethod
    def miss(cls, reason: str = "") -> MatchOutcome:
        return cls(matched=False, skipped_reason=reason)


@dataclass(frozen=True)
class TierStats:
    """Per-tier counters summed into the run report."""

    tier: Tier
    n_attempted: int = 0
    n_matched: int = 0
    n_overrode: int = 0
    n_skipped_unit: int = 0
    n_skipped_ambiguous: int = 0
