"""Unit tests for ``scoring/`` primitives.

``SolverConfigurator`` was removed in REFACTOR_LINKING phase L5 (solver.py
deleted; bw2calc moved to the optional ``build`` dependency group). Tests for
the current native scoring path live in ``test_scoring_native.py``.
``split_chunks`` moved to ``BacktestPipeline._split_chunks`` in the final
cross-phase review (it was a module-level function, which is not OOP-compliant).
"""

from __future__ import annotations

from pipelines.backtest import BacktestPipeline

# Convenience alias so test names read naturally.
_split = BacktestPipeline._split_chunks

# ============================================================================
# BacktestPipeline._split_chunks


class TestSplitChunks:
    def test_empty_input_returns_empty(self):
        assert _split([], 4) == []

    def test_round_robin_balanced(self):
        chunks = _split(list(range(7)), 3)
        # Round-robin: 0,3,6 / 1,4 / 2,5
        assert chunks == [[0, 3, 6], [1, 4], [2, 5]]
        assert sum(len(c) for c in chunks) == 7

    def test_single_worker_returns_one_chunk(self):
        assert _split([1, 2, 3], 1) == [[1, 2, 3]]

    def test_zero_or_negative_workers_treated_as_one(self):
        assert _split([1, 2], 0) == [[1, 2]]
        assert _split([1, 2], -3) == [[1, 2]]

    def test_more_workers_than_items_drops_empty_buckets(self):
        chunks = _split([1, 2], 5)
        assert chunks == [[1], [2]]  # 3 empty buckets removed
