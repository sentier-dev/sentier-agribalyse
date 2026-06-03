"""Unit tests for ``DanglingEdgePruner``.

The pruner replaces the row-removal half of the pre-refactor
``MatrixPurger`` loop: drop technosphere edges whose ``input_id``
isn't produced by anyone and whose ``output_id`` isn't an activity.
The biosphere matrix is rectangular and unaffected.
"""

from __future__ import annotations

import pandas as pd

from scoring.dangling_edge_pruner import DanglingEdgePruner
from scoring.exchange_frame import ExchangeFrame


def _frame(rows: list[tuple]) -> ExchangeFrame:
    df = pd.DataFrame(
        rows,
        columns=["output_id", "input_id", "amount", "edge_type", "is_biosphere"],
    )
    df["output_id"] = df["output_id"].astype("int64")
    df["input_id"] = df["input_id"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["edge_type"] = df["edge_type"].astype("string")
    df["is_biosphere"] = df["is_biosphere"].astype(bool)
    return ExchangeFrame(df=df)


class TestDanglingEdgePruner:
    def test_passthrough_when_every_consumed_input_has_a_producer(self):
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (2, 20, 1.0, "production", False),
                (1, 20, 0.5, "technosphere", False),
            ]
        )
        out, stats = DanglingEdgePruner().prune(frame)
        assert out is frame
        assert stats == {
            "dropped": 0,
            "missing_inputs": 0,
            "missing_outputs": 0,
            "zero_amount_producers": 0,
            "self_loops": 0,
        }

    def test_drops_consumption_with_no_producer(self):
        # Activity 1 consumes product 99, but no activity produces 99.
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 99, 0.5, "technosphere", False),  # dangling input
            ]
        )
        out, stats = DanglingEdgePruner().prune(frame)
        assert stats["dropped"] == 1
        assert stats["missing_inputs"] == 1
        assert stats["missing_outputs"] == 0
        assert (out.df["edge_type"] == "technosphere").sum() == 0

    def test_drops_consumption_owned_by_unknown_activity(self):
        # Activity 99 has a consumption edge but no production — its
        # output isn't in the activity universe.
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (99, 10, 0.5, "technosphere", False),  # dangling output
            ]
        )
        out, stats = DanglingEdgePruner().prune(frame)
        assert stats["dropped"] == 1
        assert stats["missing_outputs"] == 1
        assert 99 not in set(out.df["output_id"])

    def test_biosphere_edges_unaffected(self):
        # Biosphere flow 999 isn't a technosphere product; it shouldn't
        # be flagged as dangling — biosphere lives in matrix B which is
        # rectangular.
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 999, 0.1, "biosphere", True),
            ]
        )
        out, stats = DanglingEdgePruner().prune(frame)
        assert stats["dropped"] == 0
        assert (out.df["edge_type"] == "biosphere").sum() == 1

    def test_substitution_edges_handled_like_technosphere(self):
        # Substitution edges contribute to matrix A and so are subject
        # to the same producer-must-exist invariant.
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 99, -0.2, "substitution", False),  # dangling
            ]
        )
        _out, stats = DanglingEdgePruner().prune(frame)
        assert stats["dropped"] == 1

    def test_self_loop_technosphere_is_dropped(self):
        # Activity 1 produces product 10 (amount 1) AND consumes its own
        # output 10 (amount 1, technosphere). Naive matrix build sums to
        # zero on the diagonal — pypardiso pivots through and amplifies.
        # The pruner drops the consumption row, keeping the production
        # diagonal intact.
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 10, 1.0, "technosphere", False),
            ]
        )
        out, stats = DanglingEdgePruner().prune(frame)
        assert stats["self_loops"] == 1
        # Production row survives, technosphere self-loop is gone.
        assert (out.df["edge_type"] == "production").sum() == 1
        assert (out.df["edge_type"] == "technosphere").sum() == 0

    def test_zero_amount_producer_treated_as_no_producer(self):
        # Activity 99 'produces' product 999 with amount=0 — pypardiso
        # would pivot through this and amplify, so the scoring path
        # treats it the same as no producer at all. The consumer's
        # edge AND the activity's no-op rows must all be dropped.
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),  # real producer
                (99, 999, 0.0, "production", False),  # zero-amount production
                (99, 50, 0.5, "technosphere", False),  # consumption owned by 99
                (1, 999, 0.3, "technosphere", False),  # consumer of the zero-prod product
            ]
        )
        out, stats = DanglingEdgePruner().prune(frame)
        # Activity 99 is gone (zero production), and consumer's reference
        # to product 999 is dropped (no real producer).
        assert 99 not in set(out.df["output_id"])
        assert 999 not in set(out.df["input_id"])
        # Real activity 1's production row stays.
        assert (out.df["edge_type"] == "production").sum() == 1
        assert stats["zero_amount_producers"] >= 2
        assert stats["dropped"] >= 3
