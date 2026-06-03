"""``AuditLog`` — accumulates audit entries and writes them as a parquet.

Every match decision that overrides a previously-set link goes here
(REFACTOR.md §6 + fix 1.e). Every silently-suppressed exception in the
bw2io strategy chain goes into ``SuppressedStrategyLog`` (fix 1.g).
Every drop from the cleanup strategies goes into ``DropTallyTracker``
(fix 1.i).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from domain import AuditEntry, DropEvent, OverrideKind, SuppressedStrategy, Tier
from registry.schema import AuditTable


@dataclass
class AuditLog:
    """Accumulates audit entries for the run and writes them to parquet."""

    output_path: Path
    entries: list[AuditEntry] = field(default_factory=list)

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def record_new_link(
        self,
        *,
        process_name: str,
        exchange_name: str,
        exchange_unit: str,
        exchange_bucket: str,
        new_tier: Tier,
        new_target_db: str,
        new_target_code: str,
        new_provenance: str,
        unit_conversion: float = 1.0,
    ) -> None:
        self.record(
            AuditEntry(
                process_name=process_name,
                exchange_name=exchange_name,
                exchange_unit=exchange_unit,
                exchange_bucket=exchange_bucket,
                kind=OverrideKind.NEW_LINK,
                new_tier=new_tier,
                new_target_db=new_target_db,
                new_target_code=new_target_code,
                new_provenance=new_provenance,
                unit_conversion=unit_conversion,
            )
        )

    def record_override(
        self,
        *,
        process_name: str,
        exchange_name: str,
        exchange_unit: str,
        exchange_bucket: str,
        new_tier: Tier,
        new_target_db: str,
        new_target_code: str,
        new_provenance: str,
        prior_tier: Tier | None,
        prior_target_db: str,
        prior_target_code: str,
        unit_conversion: float = 1.0,
        reason: str = "",
    ) -> None:
        kind = (
            OverrideKind.SAME_TIER_OVERRIDE
            if prior_tier is not None and prior_tier == new_tier
            else OverrideKind.OVERRIDE
        )
        self.record(
            AuditEntry(
                process_name=process_name,
                exchange_name=exchange_name,
                exchange_unit=exchange_unit,
                exchange_bucket=exchange_bucket,
                kind=kind,
                new_tier=new_tier,
                new_target_db=new_target_db,
                new_target_code=new_target_code,
                new_provenance=new_provenance,
                prior_tier=prior_tier,
                prior_target_db=prior_target_db,
                prior_target_code=prior_target_code,
                unit_conversion=unit_conversion,
                reason=reason,
            )
        )

    def record_unit_mismatch(
        self,
        *,
        process_name: str,
        exchange_name: str,
        exchange_unit: str,
        exchange_bucket: str,
        candidate_tier: Tier,
        candidate_target_db: str,
        candidate_target_code: str,
        candidate_provenance: str,
        candidate_unit: str,
    ) -> None:
        self.record(
            AuditEntry(
                process_name=process_name,
                exchange_name=exchange_name,
                exchange_unit=exchange_unit,
                exchange_bucket=exchange_bucket,
                kind=OverrideKind.REJECTED_UNIT_MISMATCH,
                new_tier=candidate_tier,
                new_target_db=candidate_target_db,
                new_target_code=candidate_target_code,
                new_provenance=candidate_provenance,
                reason=f"source_unit={exchange_unit!r} target_unit={candidate_unit!r}",
            )
        )

    def record_ambiguous(
        self,
        *,
        process_name: str,
        exchange_name: str,
        exchange_unit: str,
        exchange_bucket: str,
        candidate_tier: Tier,
        n_candidates: int,
    ) -> None:
        self.record(
            AuditEntry(
                process_name=process_name,
                exchange_name=exchange_name,
                exchange_unit=exchange_unit,
                exchange_bucket=exchange_bucket,
                kind=OverrideKind.SKIPPED_AMBIGUOUS,
                new_tier=candidate_tier,
                new_target_db="",
                new_target_code="",
                new_provenance="",
                reason=f"{n_candidates} equally-valid candidates",
            )
        )

    def write(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df = AuditTable().to_dataframe(self.entries)
        df.to_parquet(self.output_path, index=False)
        return self.output_path

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class SuppressedStrategyLog:
    """Captures exceptions that the legacy linker silently suppressed (fix 1.g)."""

    output_path: Path
    entries: list[SuppressedStrategy] = field(default_factory=list)

    def record(self, strategy: str, exc: BaseException, *, occurred_at_step: str = "") -> None:
        self.entries.append(
            SuppressedStrategy(
                strategy=strategy,
                error_type=type(exc).__name__,
                error_message=str(exc),
                occurred_at_step=occurred_at_step,
            )
        )

    def counts_by_strategy(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.strategy] = out.get(e.strategy, 0) + 1
        return out

    def write(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "strategy": e.strategy,
                "error_type": e.error_type,
                "error_message": e.error_message,
                "occurred_at_step": e.occurred_at_step,
            }
            for e in self.entries
        ]
        cols = ("strategy", "error_type", "error_message", "occurred_at_step")
        df = (
            pd.DataFrame(records, columns=list(cols))
            if records
            else pd.DataFrame(columns=list(cols))
        )
        df.to_parquet(self.output_path, index=False)
        return self.output_path


@dataclass
class DropTallyTracker:
    """Counts drops from cleanup strategies (fix 1.i)."""

    events: list[DropEvent] = field(default_factory=list)

    def record(self, strategy: str, n_dropped: int, process_name: str = "") -> None:
        if n_dropped <= 0:
            return
        self.events.append(
            DropEvent(strategy=strategy, n_dropped=n_dropped, process_name=process_name)
        )

    def totals_by_strategy(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e.strategy] = out.get(e.strategy, 0) + e.n_dropped
        return out
