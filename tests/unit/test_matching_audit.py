"""Unit tests for ``matching.audit`` — AuditLog, SuppressedStrategyLog, DropTallyTracker."""

from __future__ import annotations

import pandas as pd

from domain import OverrideKind, Tier
from matching.audit import AuditLog, DropTallyTracker, SuppressedStrategyLog


class TestAuditLog:
    def test_record_appends_entry(self, tmp_path):
        log = AuditLog(output_path=tmp_path / "audit.parquet")
        log.record_new_link(
            process_name="p",
            exchange_name="e",
            exchange_unit="kg",
            exchange_bucket="air",
            new_tier=Tier.CURATED_TARGETED,
            new_target_db="biosphere3",
            new_target_code="abc",
            new_provenance="prov",
        )
        assert len(log) == 1
        assert log.entries[0].kind == OverrideKind.NEW_LINK

    def test_override_with_same_tier_logs_same_tier_override(self, tmp_path):
        log = AuditLog(output_path=tmp_path / "audit.parquet")
        log.record_override(
            process_name="p",
            exchange_name="e",
            exchange_unit="kg",
            exchange_bucket="air",
            new_tier=Tier.CURATED_TARGETED,
            new_target_db="b",
            new_target_code="c",
            new_provenance="prov",
            prior_tier=Tier.CURATED_TARGETED,
            prior_target_db="other",
            prior_target_code="d",
        )
        assert log.entries[0].kind == OverrideKind.SAME_TIER_OVERRIDE

    def test_override_with_lower_prior_tier_logs_override(self, tmp_path):
        log = AuditLog(output_path=tmp_path / "audit.parquet")
        log.record_override(
            process_name="p",
            exchange_name="e",
            exchange_unit="kg",
            exchange_bucket="air",
            new_tier=Tier.CURATED_TARGETED,
            new_target_db="b",
            new_target_code="c",
            new_provenance="prov",
            prior_tier=Tier.HARMONISED_FLOWS,
            prior_target_db="other",
            prior_target_code="d",
        )
        assert log.entries[0].kind == OverrideKind.OVERRIDE
        assert log.entries[0].prior_tier == Tier.HARMONISED_FLOWS

    def test_unit_mismatch_records_reason(self, tmp_path):
        log = AuditLog(output_path=tmp_path / "audit.parquet")
        log.record_unit_mismatch(
            process_name="p",
            exchange_name="e",
            exchange_unit="kg",
            exchange_bucket="air",
            candidate_tier=Tier.CURATED_TARGETED,
            candidate_target_db="b",
            candidate_target_code="c",
            candidate_provenance="prov",
            candidate_unit="m3",
        )
        assert log.entries[0].kind == OverrideKind.REJECTED_UNIT_MISMATCH
        assert "kg" in log.entries[0].reason
        assert "m3" in log.entries[0].reason

    def test_ambiguous_records_candidate_count(self, tmp_path):
        log = AuditLog(output_path=tmp_path / "audit.parquet")
        log.record_ambiguous(
            process_name="p",
            exchange_name="e",
            exchange_unit="kg",
            exchange_bucket="air",
            candidate_tier=Tier.HARMONISED_FLOWS,
            n_candidates=4,
        )
        assert "4" in log.entries[0].reason
        assert log.entries[0].kind == OverrideKind.SKIPPED_AMBIGUOUS

    def test_write_emits_parquet_file(self, tmp_path):
        log = AuditLog(output_path=tmp_path / "subdir" / "audit.parquet")
        log.record_new_link(
            process_name="p",
            exchange_name="e",
            exchange_unit="kg",
            exchange_bucket="air",
            new_tier=Tier.CURATED_TARGETED,
            new_target_db="b",
            new_target_code="c",
            new_provenance="prov",
        )
        out = log.write()
        assert out.exists()
        df = pd.read_parquet(out)
        assert df.iloc[0]["kind"] == "new_link"

    def test_write_with_no_entries_creates_empty_parquet(self, tmp_path):
        log = AuditLog(output_path=tmp_path / "audit.parquet")
        log.write()
        df = pd.read_parquet(tmp_path / "audit.parquet")
        assert df.empty


class TestSuppressedStrategyLog:
    def test_record_appends_entry_with_error_metadata(self, tmp_path):
        log = SuppressedStrategyLog(output_path=tmp_path / "x.parquet")
        log.record("strategy_x", ValueError("boom"), occurred_at_step="step.A")
        assert log.entries[0].error_type == "ValueError"
        assert log.entries[0].error_message == "boom"
        assert log.entries[0].occurred_at_step == "step.A"

    def test_counts_by_strategy(self, tmp_path):
        log = SuppressedStrategyLog(output_path=tmp_path / "x.parquet")
        log.record("a", ValueError("x"))
        log.record("a", ValueError("y"))
        log.record("b", RuntimeError("z"))
        assert log.counts_by_strategy() == {"a": 2, "b": 1}

    def test_write_emits_parquet(self, tmp_path):
        log = SuppressedStrategyLog(output_path=tmp_path / "x.parquet")
        log.record("a", ValueError("x"))
        out = log.write()
        df = pd.read_parquet(out)
        assert df.iloc[0]["error_type"] == "ValueError"


class TestDropTallyTracker:
    def test_zero_drops_are_ignored(self):
        tracker = DropTallyTracker()
        tracker.record("strategy", 0)
        assert tracker.events == []

    def test_negative_drops_are_ignored(self):
        tracker = DropTallyTracker()
        tracker.record("strategy", -1)
        assert tracker.events == []

    def test_totals_by_strategy_aggregates(self):
        tracker = DropTallyTracker()
        tracker.record("a", 3)
        tracker.record("a", 5)
        tracker.record("b", 2)
        assert tracker.totals_by_strategy() == {"a": 8, "b": 2}
