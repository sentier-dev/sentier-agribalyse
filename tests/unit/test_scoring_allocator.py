"""Unit tests for ``Allocator``.

The allocator splits multifunctional activities (multiple production
edges on one activity) into single-product synthetic activities. This
is the pure-DataFrame replacement for bw2data's allocation strategies
that run on every ``Database.process()`` call.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scoring.allocator import Allocator
from scoring.exchange_frame import ExchangeFrame


def _make_frame(rows, with_factor=True):
    cols = ["output_id", "input_id", "amount", "edge_type", "is_biosphere"]
    if with_factor:
        cols.append("allocation_factor")
    df = pd.DataFrame(rows, columns=cols)
    df["output_id"] = df["output_id"].astype("int64")
    df["input_id"] = df["input_id"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["edge_type"] = df["edge_type"].astype("string")
    df["is_biosphere"] = df["is_biosphere"].astype(bool)
    if with_factor:
        df["allocation_factor"] = df["allocation_factor"].astype("float64")
    return ExchangeFrame(df=df)


class TestAllocatorPassthroughs:
    def test_no_allocation_column_means_passthrough(self):
        # When the frame has no allocation_factor column, the allocator
        # is a no-op (caller hasn't tagged the data yet).
        frame = _make_frame(
            [(1, 10, 1.0, "production", False)],
            with_factor=False,
        )
        out = Allocator().allocate(frame)
        assert out is frame

    def test_single_product_passes_through(self):
        # No multifunctional activities → frame returned unchanged.
        frame = _make_frame([(1, 10, 1.0, "production", False, 1.0)])
        out = Allocator().allocate(frame)
        assert out.n_rows == frame.n_rows


class TestAllocatorSplits:
    def test_two_product_activity_splits(self):
        # Activity 1 produces both product 10 (60%) and 20 (40%),
        # consuming 0.5 of input 30 and emitting 0.1 of biosphere 99.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 0.6),
                (1, 20, 1.0, "production", False, 0.4),
                (1, 30, 0.5, "technosphere", False, 1.0),
                (1, 99, 0.1, "biosphere", True, 1.0),
            ]
        )
        out = Allocator().allocate(frame)
        # Two synthetic activities now, each producing one product.
        assert len(out.activities) == 2
        # Each synthetic activity carries the consumption + biosphere edges
        # scaled by its allocation factor.
        non_prod = out.df[~out.df["edge_type"].isin(("production", "generic production"))]
        # Total consumption preserved? 0.5 * 0.6 + 0.5 * 0.4 = 0.5
        tech = non_prod[non_prod["edge_type"] == "technosphere"]
        assert tech["amount"].sum() == pytest.approx(0.5)
        # Biosphere conservation too.
        bio = non_prod[non_prod["edge_type"] == "biosphere"]
        assert bio["amount"].sum() == pytest.approx(0.1)

    def test_synthetic_ids_are_deterministic(self):
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 0.5),
                (1, 20, 1.0, "production", False, 0.5),
            ]
        )
        out_a = Allocator().allocate(frame)
        out_b = Allocator().allocate(frame)
        assert sorted(out_a.activities) == sorted(out_b.activities)


class TestAllocatorValidation:
    def test_arbitrary_positive_sum_is_normalised(self):
        # Factors in any positive scale (per-cent here) get normalised
        # to fractions before scaling. SimaPro ``manual_allocation`` is
        # per-cent; bw2data did the same divide-by-sum normalisation.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 60.0),
                (1, 20, 1.0, "production", False, 40.0),
                (1, 30, 0.5, "technosphere", False, 1.0),
            ]
        )
        out = Allocator().allocate(frame)
        tech = out.df[out.df["edge_type"] == "technosphere"]
        # Same conservation as the fractional-input case: 0.5*0.6 + 0.5*0.4
        assert tech["amount"].sum() == pytest.approx(0.5)

    def test_zero_sum_factors_rejected(self):
        # Two zero factors → cannot normalise.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 0.0),
                (1, 20, 1.0, "production", False, 0.0),
            ]
        )
        with pytest.raises(ValueError, match="must be > 0"):
            Allocator().allocate(frame)

    def test_negative_factor_rejected(self):
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, -0.5),
                (1, 20, 1.0, "production", False, 1.5),
            ]
        )
        with pytest.raises(ValueError, match="non-negative"):
            Allocator().allocate(frame)

    def test_nan_factor_rejected(self):
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, float("nan")),
                (1, 20, 1.0, "production", False, 0.5),
            ]
        )
        with pytest.raises(ValueError, match="NaN"):
            Allocator().allocate(frame)


class TestAllocatorFiltersProductionlessActivities:
    def test_activity_without_production_is_dropped(self):
        # Activity 99 has only a technosphere consumption row — its
        # production edge was stripped upstream (e.g. by drop_unlinked).
        # The Allocator must remove it so the matrix stays square.
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 1.0),
                (99, 1, 0.5, "technosphere", False, 1.0),  # productionless
                (1, 1, 0.2, "technosphere", False, 1.0),
            ]
        )
        out = Allocator().allocate(frame)
        # Only activity 1 survives.
        assert set(out.activities) == {1}
        assert 99 not in out.df["output_id"].values

    def test_passthrough_when_all_activities_have_production(self):
        # No productionless activities, no multifunctional — frame returned
        # unchanged (identity).
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 1.0),
                (1, 20, 0.3, "technosphere", False, 1.0),
            ]
        )
        out = Allocator().allocate(frame)
        assert out is frame


class TestAllocatorEnablesSquareTechnosphere:
    """End-to-end: allocate a multifunctional system, build the
    technosphere matrix, get a square A. Without the allocator the
    builder rejects with 'non-square'."""

    def test_post_allocation_matrix_is_square(self):
        from scoring.matrix_builder import TechnosphereBuilder

        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 0.6),
                (1, 20, 1.0, "production", False, 0.4),
                # second activity is single-product, untouched.
                (2, 10, 0.3, "technosphere", False, 1.0),
                (2, 30, 1.0, "production", False, 1.0),
            ]
        )
        out = Allocator().allocate(frame)
        # Now 3 activities + 3 products. The synthetic split for activity 1
        # must produce ids that line up with input_ids 10 and 20 in the
        # technosphere edge from activity 2 → product 10.
        # Builder will catch any mismatch.
        # NOTE: this test only checks that the matrix builder doesn't reject
        # the allocator's output for non-squareness — it doesn't validate
        # the supply chain itself, since cross-references between synthetic
        # ids and original product ids would need a separate id-rewrite pass.
        prod_count = len(out.products)
        act_count = len(out.activities)
        assert prod_count == act_count
        # Builder will succeed on a square frame.
        TechnosphereBuilder().build(out)


class TestAllocatorProvenance:
    def test_provenance_maps_synthetic_to_parent_and_product(self):
        # Activity 1 → products 10 (60%) and 20 (40%).
        frame = _make_frame(
            [
                (1, 10, 1.0, "production", False, 0.6),
                (1, 20, 1.0, "production", False, 0.4),
                (1, 30, 0.5, "technosphere", False, 1.0),
            ]
        )
        out, prov = Allocator().allocate_with_provenance(frame)
        # Two synthetic activities, each with a (parent=1, product) entry.
        assert len(prov) == 2
        assert set(prov.values()) == {(1, 10), (1, 20)}
        # The provenance keys are exactly the synthetic column ids emitted.
        assert set(prov) == set(out.activities)
        # Ids match the documented deterministic derivation.
        for syn_id, (parent, product) in prov.items():
            assert syn_id == Allocator._synthetic_id(parent, product)

    def test_no_split_yields_empty_provenance(self):
        frame = _make_frame([(1, 10, 1.0, "production", False, 1.0)])
        out, prov = Allocator().allocate_with_provenance(frame)
        assert prov == {}
        assert out.n_rows == frame.n_rows
