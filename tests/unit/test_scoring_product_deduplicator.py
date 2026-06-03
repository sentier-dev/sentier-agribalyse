"""Unit tests for ``ProductDeduplicator``.

Required because AGB has multiple activities producing the same product
(an orphan-product-relinker artefact). The pre-refactor pipeline
collapsed those via ``MatrixPurger`` running on SQLite; this class
does the same job on a long-form DataFrame.
"""

from __future__ import annotations

import pandas as pd

from scoring.exchange_frame import ExchangeFrame
from scoring.product_deduplicator import ProductDeduplicator


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


class TestProductDeduplicator:
    def test_passthrough_when_each_product_has_one_producer(self):
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (2, 20, 1.0, "production", False),
                (1, 20, 0.3, "technosphere", False),  # 1 consumes from 2
            ]
        )
        out = ProductDeduplicator().deduplicate(frame)
        # No change — frame is already square-ready.
        assert out is frame

    def test_keeps_canonical_producer_per_product(self):
        # Activities 1, 2, 3 all produce product 10. Smallest id wins.
        frame = _frame(
            [
                (3, 10, 1.0, "production", False),
                (1, 10, 1.0, "production", False),
                (2, 10, 1.0, "production", False),
                (5, 99, 1.0, "production", False),  # unrelated
            ]
        )
        out = ProductDeduplicator().deduplicate(frame)
        producers_of_10 = set(
            out.df[(out.df["edge_type"] == "production") & (out.df["input_id"] == 10)]["output_id"]
        )
        assert producers_of_10 == {1}
        # Activity 5 is untouched.
        assert 5 in set(out.df["output_id"])

    def test_dropped_activity_takes_its_consumption_with_it(self):
        # Activity 2 produces product 10 (duplicate of activity 1) and
        # consumes product 20. Both rows must vanish.
        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                (2, 10, 1.0, "production", False),
                (2, 20, 0.5, "technosphere", False),
                (1, 20, 0.5, "technosphere", False),
            ]
        )
        out = ProductDeduplicator().deduplicate(frame)
        assert 2 not in set(out.df["output_id"])
        # Activity 1's rows are still there.
        assert (out.df["output_id"] == 1).sum() == 2

    def test_makes_technosphere_square(self):
        from scoring.matrix_builder import TechnosphereBuilder

        frame = _frame(
            [
                (1, 10, 1.0, "production", False),
                # Activity 2 also claims product 10. Pre-dedup the frame
                # has 1 product but 2 activities → matrix non-square.
                (2, 10, 1.0, "production", False),
                # A consumer to give the matrix a non-trivial off-diagonal.
                (1, 10, 0.1, "technosphere", False),
            ]
        )
        out = ProductDeduplicator().deduplicate(frame)
        TechnosphereBuilder().build(out)  # would raise on non-square

    def test_deterministic_choice_across_runs(self):
        frame = _frame(
            [
                (7, 10, 1.0, "production", False),
                (3, 10, 1.0, "production", False),
                (5, 10, 1.0, "production", False),
            ]
        )
        out_a = ProductDeduplicator().deduplicate(frame)
        out_b = ProductDeduplicator().deduplicate(frame)
        assert sorted(out_a.df["output_id"].tolist()) == sorted(out_b.df["output_id"].tolist())
