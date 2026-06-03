"""Unit tests for ``matching.StrategyRunner`` — exception capture without crash."""

from __future__ import annotations

import pytest

from matching.audit import SuppressedStrategyLog
from matching.strategy_runner import StrategyRunner


@pytest.fixture
def runner(tmp_path):
    return StrategyRunner(suppressed_log=SuppressedStrategyLog(output_path=tmp_path / "x.parquet"))


class TestStrategyRunner:
    def test_successful_strategy_returns_value(self, runner):
        def good(x):
            return x * 2

        result = runner.apply(good, 5)
        assert result == 10
        assert runner.suppressed_log.entries == []

    def test_failing_strategy_returns_none_and_logs(self, runner):
        def bad():
            raise ValueError("boom")

        result = runner.apply(bad)
        assert result is None
        assert len(runner.suppressed_log.entries) == 1
        assert runner.suppressed_log.entries[0].error_type == "ValueError"

    def test_label_kwarg_overrides_function_name(self, runner):
        def bad():
            raise RuntimeError("die")

        runner.apply(bad, label="custom_label")
        assert runner.suppressed_log.entries[0].strategy == "custom_label"

    def test_uses_function_name_when_no_label(self, runner):
        def named_strategy():
            raise RuntimeError("die")

        runner.apply(named_strategy)
        assert runner.suppressed_log.entries[0].strategy == "named_strategy"

    def test_propagates_args_and_kwargs_to_strategy(self, runner):
        seen = {}

        def collect(*a, **k):
            seen["args"] = a
            seen["kwargs"] = k
            return "ok"

        result = runner.apply(collect, 1, 2, key="value")
        assert result == "ok"
        assert seen == {"args": (1, 2), "kwargs": {"key": "value"}}
