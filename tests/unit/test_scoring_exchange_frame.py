"""Unit tests for ``ExchangeFrame`` and the matrix builders.

These are the Phase 2 building blocks for the SQLite-free scoring
pipeline. The tests verify schema enforcement, the technosphere
sign convention, biosphere matrix shape, and the CF row vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scoring.exchange_frame import ExchangeFrame
from scoring.matrix_builder import (
    BiosphereBuilder,
    CharacterizationBuilder,
    TechnosphereBuilder,
)


def _make_frame(rows):
    return ExchangeFrame.from_long(
        pd.DataFrame(
            rows,
            columns=["output_id", "input_id", "amount", "edge_type", "is_biosphere"],
        )
    )


class TestExchangeFrameSchema:
    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"output_id": [1], "input_id": [2], "amount": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            ExchangeFrame(df=df)

    def test_rejects_non_integer_ids(self):
        df = pd.DataFrame(
            {
                "output_id": ["a"],
                "input_id": ["b"],
                "amount": [1.0],
                "edge_type": ["production"],
                "is_biosphere": [False],
            }
        )
        with pytest.raises(ValueError, match="must be integer"):
            ExchangeFrame(df=df)

    def test_from_long_widens_dtypes(self):
        # Ints arriving as numpy int32 are upcast to int64 on construction.
        df = pd.DataFrame(
            {
                "output_id": np.array([1, 2], dtype="int32"),
                "input_id": np.array([10, 20], dtype="int32"),
                "amount": [1.0, 2.0],
                "edge_type": ["production", "production"],
                "is_biosphere": [False, False],
            }
        )
        frame = ExchangeFrame.from_long(df)
        assert frame.df["output_id"].dtype == np.int64
        assert frame.df["input_id"].dtype == np.int64

    def test_slices_partition_the_frame(self):
        frame = _make_frame(
            [
                # activity 1 produces product 10, consumes input 20, emits CO2 (30)
                (1, 10, 1.0, "production", False),
                (1, 20, 0.5, "technosphere", False),
                (1, 30, 0.1, "biosphere", True),
            ]
        )
        assert len(frame.production) == 1
        assert len(frame.technosphere) == 2  # production + technosphere
        assert len(frame.biosphere) == 1


class TestTechnosphereBuilder:
    def test_diagonal_production_edge(self):
        # A 1-activity / 1-product system: A = [[1]].
        frame = _make_frame([(1, 10, 1.0, "production", False)])
        built = TechnosphereBuilder().build(frame)
        assert built.shape == (1, 1)
        assert built.matrix[0, 0] == pytest.approx(1.0)

    def test_consumption_is_negative(self):
        # Two activities, two products. Activity 1 consumes 0.5 of product 20.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False),
                (2, 20, 1.0, "production", False),
                (1, 20, 0.5, "technosphere", False),
            ]
        )
        built = TechnosphereBuilder().build(frame)
        assert built.shape == (2, 2)
        # Product 20 (input) appears in activity 1's column with sign -.
        row_20 = built.row_id_to_idx[20]
        col_1 = built.col_id_to_idx[1]
        assert built.matrix[row_20, col_1] == pytest.approx(-0.5)

    def test_substitution_is_positive_like_production(self):
        # Two activities, two products. Activity 1 produces product 10
        # AND substitutes 0.3 of product 20 (displaces an external
        # producer). Activity 2 produces product 20.
        # bw_processing convention: substitution rows have flip=False,
        # so A[product_20, activity_1] = +0.3 (a credit), not -0.3.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False),
                (2, 20, 1.0, "production", False),
                (1, 20, 0.3, "substitution", False),
            ]
        )
        built = TechnosphereBuilder().build(frame)
        row_20 = built.row_id_to_idx[20]
        col_1 = built.col_id_to_idx[1]
        assert built.matrix[row_20, col_1] == pytest.approx(0.3)

    def test_rejects_nonsquare(self):
        # 2 production rows, 1 activity column → multifunctional, must be allocated first.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 11, 0.5, "production", False),
            ]
        )
        with pytest.raises(ValueError, match="non-square"):
            TechnosphereBuilder().build(frame)

    def test_rejects_empty_technosphere(self):
        # Only a biosphere edge → no technosphere structure to build.
        frame = _make_frame([(1, 30, 0.1, "biosphere", True)])
        with pytest.raises(ValueError, match="empty technosphere"):
            TechnosphereBuilder().build(frame)


class TestBiosphereBuilder:
    def test_one_flow_one_activity(self):
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 30, 0.1, "biosphere", True),
            ]
        )
        built = BiosphereBuilder().build(frame)
        # Rows = 1 flow, cols = 1 activity.
        assert built.shape == (1, 1)
        assert built.matrix[0, 0] == pytest.approx(0.1)

    def test_empty_biosphere_returns_zero_matrix(self):
        # No biosphere edges → zero-row matrix shaped to the activity count.
        frame = _make_frame([(1, 10, 1.0, "production", False)])
        built = BiosphereBuilder().build(frame)
        assert built.shape == (0, 1)


class TestCharacterizationBuilder:
    def test_cf_row_vector_aligns_with_biosphere_rows(self):
        # CO2 (id 30) at 1.0, methane (id 31) at 28.0; biosphere has only CO2.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 30, 0.1, "biosphere", True),
            ]
        )
        biosphere = BiosphereBuilder().build(frame)
        cf_df = pd.DataFrame({"flow_id": [30, 31], "cf": [1.0, 28.0]})
        Q = CharacterizationBuilder().build(cf_df, biosphere)
        # Methane CF (31) gets dropped — not in biosphere.
        assert Q.shape == (1, 1)
        assert Q[0, 0] == pytest.approx(1.0)

    def test_full_score_pipeline_one_unit_demand(self):
        """Q @ B @ supply, where supply solves A @ supply = demand."""
        from scipy.sparse.linalg import spsolve

        # 1-activity / 1-product / 1-flow system with CF=1.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False),
                (1, 30, 0.5, "biosphere", True),
            ]
        )
        A = TechnosphereBuilder().build(frame)
        B = BiosphereBuilder().build(frame)
        Q = CharacterizationBuilder().build(pd.DataFrame({"flow_id": [30], "cf": [1.0]}), B)
        # demand = 1 unit of product 10 → supply = 1 unit of activity 1.
        demand = np.array([1.0])
        supply = spsolve(A.matrix.tocsc(), demand)
        score = float((Q @ B.matrix @ supply)[0])
        # 1 unit of activity emits 0.5 of flow 30, characterized at 1.0.
        assert score == pytest.approx(0.5)
