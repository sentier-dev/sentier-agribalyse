"""Unit tests for ``ScoreDecomposer``.

Decomposes ``score = Q @ B @ A^{-1} @ d`` into named contributions:

* ``flow_contributions``  — ``Q[i] * inventory[i]`` per biosphere flow row.
* ``activity_contributions`` — ``(Q @ B)[j] * supply[j]`` per technosphere column.
* ``edge_contributions`` — ``Q[i] * B[i, j] * supply[j]`` per (flow, activity) pair.

All three slices must sum to the same scalar score, modulo float noise.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scoring.decomposer import Decomposition, ScoreDecomposer
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.product_catalog import ProductCatalog
from scoring.scoring_package import ScoringPackageBuilder


def _activity(database, code, name, exchanges, **extra):
    return {
        "database": database,
        "code": code,
        "name": name,
        "exchanges": exchanges,
        **extra,
    }


def _product_catalog(rows: list[dict]) -> ProductCatalog:
    df = pd.DataFrame(
        rows, columns=["database", "code", "name", "type", "unit", "product_id"]
    ).astype(
        {
            "database": "string",
            "code": "string",
            "name": "string",
            "type": "string",
            "unit": "string",
            "product_id": "int64",
        }
    )
    return ProductCatalog(_df=df)


def _bio_catalog(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestScoreDecomposer:
    """Two-activity wheat → steel chain, single biosphere flow.

    Geometry::

        wheat (1 kg) emits 2 kg CO2, consumes 0.5 kg steel
        steel (1 kg) emits 5 kg CO2, no upstream

    Solving for 1 kg wheat demand::

        supply        = [wheat=1.0, steel=0.5]
        inventory_co2 = 2*1 + 5*0.5 = 4.5 kg CO2
        score         = 4.5 (CF=1.0)

    Per-activity contributions: wheat=2.0, steel=2.5.
    """

    @staticmethod
    def _wheat_steel_package():
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {"type": "production", "input": ("agb", "wheat-prod"), "amount": 1.0},
                    {"type": "technosphere", "input": ("agb", "steel-prod"), "amount": 0.5},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
                ],
            ),
            _activity(
                "agb",
                "steel",
                "steel",
                exchanges=[
                    {"type": "production", "input": ("agb", "steel-prod"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 5.0},
                ],
            ),
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        method = ("ef", "climate")
        cfs = {method: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        package = ScoringPackageBuilder().build(frame, cfs)

        product_catalog = _product_catalog(
            [
                {
                    "database": "agb",
                    "code": "wheat",
                    "name": "wheat",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "wheat-prod")),
                },
                {
                    "database": "agb",
                    "code": "steel",
                    "name": "steel",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "steel-prod")),
                },
            ]
        )
        bio = _bio_catalog(
            [{"database": "bio3", "code": "co2", "name": "Carbon dioxide", "categories": ["air"]}]
        )
        return package, product_catalog, bio, method

    def test_score_equals_sum_of_flow_contributions(self):
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        result = d.decompose(("agb", "wheat"), method)

        assert isinstance(result, Decomposition)
        assert result.score == pytest.approx(4.5)
        assert result.flow_contributions["contribution"].sum() == pytest.approx(4.5)

    def test_score_equals_sum_of_activity_contributions(self):
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        result = d.decompose(("agb", "wheat"), method)

        assert result.activity_contributions["contribution"].sum() == pytest.approx(4.5)

    def test_score_equals_sum_of_edge_contributions(self):
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        result = d.decompose(("agb", "wheat"), method)

        assert result.edge_contributions["contribution"].sum() == pytest.approx(4.5)

    def test_per_activity_contributions_split_2_and_2_5(self):
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        result = d.decompose(("agb", "wheat"), method)

        ac = result.activity_contributions.set_index("activity_name")["contribution"]
        assert ac["wheat"] == pytest.approx(2.0)
        assert ac["steel"] == pytest.approx(2.5)

    def test_flow_contribution_carries_name_and_cf(self):
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        result = d.decompose(("agb", "wheat"), method)

        assert len(result.flow_contributions) == 1
        row = result.flow_contributions.iloc[0]
        assert row["flow_name"] == "Carbon dioxide"
        assert row["cf"] == pytest.approx(1.0)
        assert row["inventory_amount"] == pytest.approx(4.5)
        assert row["contribution"] == pytest.approx(4.5)
        # Share is contribution / score.
        assert row["share"] == pytest.approx(1.0)

    def test_edge_contributions_carry_supply_and_edge_amount(self):
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        result = d.decompose(("agb", "wheat"), method)

        # Two edges expected — wheat→co2, steel→co2.
        assert len(result.edge_contributions) == 2
        wheat_edge = result.edge_contributions.set_index("activity_name").loc["wheat"]
        assert wheat_edge["edge_amount"] == pytest.approx(2.0)
        assert wheat_edge["supply"] == pytest.approx(1.0)
        assert wheat_edge["cf"] == pytest.approx(1.0)
        assert wheat_edge["contribution"] == pytest.approx(2.0)
        steel_edge = result.edge_contributions.set_index("activity_name").loc["steel"]
        assert steel_edge["edge_amount"] == pytest.approx(5.0)
        assert steel_edge["supply"] == pytest.approx(0.5)
        assert steel_edge["contribution"] == pytest.approx(2.5)

    def test_unknown_product_raises_value_error(self):
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        with pytest.raises(ValueError, match="not in product catalog"):
            d.decompose(("agb", "ghost"), method)

    def test_unknown_method_raises_value_error(self):
        package, pc, bio, _method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        with pytest.raises(ValueError, match="not registered"):
            d.decompose(("agb", "wheat"), ("ef", "ghost-method"))

    def test_uncharacterised_flow_dropped_from_flow_contributions(self):
        # Add a second biosphere flow (water) that the method does not characterise.
        # Verify it does NOT appear in flow_contributions.
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {"type": "production", "input": ("agb", "wheat-prod"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
                    {"type": "biosphere", "input": ("bio3", "h2o"), "amount": 100.0},
                ],
            ),
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        method = ("ef", "climate")
        cfs = {method: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        package = ScoringPackageBuilder().build(frame, cfs)
        pc = _product_catalog(
            [
                {
                    "database": "agb",
                    "code": "wheat",
                    "name": "wheat",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "wheat-prod")),
                },
            ]
        )
        bio = _bio_catalog(
            [
                {"database": "bio3", "code": "co2", "name": "CO2", "categories": ["air"]},
                {"database": "bio3", "code": "h2o", "name": "Water", "categories": ["water"]},
            ]
        )

        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        result = d.decompose(("agb", "wheat"), method)

        # Score is 2.0 (only CO2 has CF=1).
        assert result.score == pytest.approx(2.0)
        names = set(result.flow_contributions["flow_name"])
        assert names == {"CO2"}

    def test_top_n_truncation_returns_largest_contributions(self):
        # Two-activity case: top_n=1 should return only the larger contributor.
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )

        result = d.decompose(("agb", "wheat"), method, top_n=1)

        # Steel contributes 2.5 (larger than wheat's 2.0); steel should be the only row.
        assert len(result.activity_contributions) == 1
        assert result.activity_contributions.iloc[0]["activity_name"] == "steel"

    def test_negative_contributions_kept_with_sign(self):
        # A *negative* CF (e.g. carbon-uptake credit) should produce a negative
        # contribution, not be filtered out — top-N is by absolute magnitude.
        sp_data = [
            _activity(
                "agb",
                "forest",
                "forest",
                exchanges=[
                    {"type": "production", "input": ("agb", "forest-prod"), "amount": 1.0},
                    # Forest *absorbs* CO2 — biosphere amount negative is canonical.
                    {"type": "biosphere", "input": ("bio3", "co2-uptake"), "amount": -3.0},
                ],
            ),
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2-uptake"))
        method = ("ef", "climate")
        cfs = {method: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        package = ScoringPackageBuilder().build(frame, cfs)
        pc = _product_catalog(
            [
                {
                    "database": "agb",
                    "code": "forest",
                    "name": "forest",
                    "type": "process",
                    "unit": "ha",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "forest-prod")),
                },
            ]
        )
        bio = _bio_catalog(
            [
                {
                    "database": "bio3",
                    "code": "co2-uptake",
                    "name": "CO2 uptake",
                    "categories": ["air"],
                }
            ]
        )

        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        result = d.decompose(("agb", "forest"), method)

        assert result.score == pytest.approx(-3.0)
        assert result.flow_contributions["contribution"].iloc[0] == pytest.approx(-3.0)

    def test_inventory_returns_all_emitted_flows(self):
        # Two biosphere flows; only co2 has a CF. inventory() must
        # surface BOTH (uncharacterised water as well), so the user can
        # see "right amount, wrong CF" diagnostics.
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {"type": "production", "input": ("agb", "wheat-prod"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
                    {"type": "biosphere", "input": ("bio3", "h2o"), "amount": 100.0},
                ],
            ),
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        method = ("ef", "climate")
        cfs = {method: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        package = ScoringPackageBuilder().build(frame, cfs)
        pc = _product_catalog(
            [
                {
                    "database": "agb",
                    "code": "wheat",
                    "name": "wheat",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "wheat-prod")),
                },
            ]
        )
        bio = _bio_catalog(
            [
                {"database": "bio3", "code": "co2", "name": "CO2", "categories": ["air"]},
                {"database": "bio3", "code": "h2o", "name": "Water", "categories": ["water"]},
            ]
        )

        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        inv = d.inventory(("agb", "wheat"), method=method)

        assert set(inv["flow_name"]) == {"CO2", "Water"}
        # CO2 has CF 1, contribution 2.0; Water has CF 0, contribution 0.
        co2_row = inv.set_index("flow_name").loc["CO2"]
        h2o_row = inv.set_index("flow_name").loc["Water"]
        assert co2_row["inventory_amount"] == pytest.approx(2.0)
        assert co2_row["cf"] == pytest.approx(1.0)
        assert co2_row["contribution"] == pytest.approx(2.0)
        assert h2o_row["inventory_amount"] == pytest.approx(100.0)
        assert h2o_row["cf"] == pytest.approx(0.0)
        assert h2o_row["contribution"] == pytest.approx(0.0)

    def test_inventory_without_method_omits_cf_application(self):
        # method=None should still return inventory_amount but cf and
        # contribution columns are zero — the caller is asking for
        # "what does this product emit?", not "scored against what".
        package, pc, bio, _method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        inv = d.inventory(("agb", "wheat"), method=None)
        assert (inv["cf"] == 0).all()
        assert (inv["contribution"] == 0).all()
        # And the inventory_amount is still the correct total.
        assert inv["inventory_amount"].sum() == pytest.approx(4.5)

    def test_regional_correction_folds_into_score_and_activity_table(self):
        # Mirror NativeLciaScorer (src/scoring/native_scorer.py): score
        # = (Q @ inv) + (correction @ supply). A wheat-only chain where
        # the regional CF for FR is 6.98 instead of the global 42.95
        # (matching the AWARE FR vs global numbers we found in the real
        # registry) should produce a score of 6.98 (regional), with the
        # decomposer reporting global_score=42.95 and correction_score
        # =-35.97 so they sum back to the corrected score.
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {"type": "production", "input": ("agb", "wheat-prod"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "water"), "amount": 1.0},
                ],
                location="FR",
            ),
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        water_id = ExchangeFrameBuilder.flow_id_for(("bio3", "water"))
        wheat_aid = ExchangeFrameBuilder.flow_id_for(("agb", "wheat"))
        method = ("ef", "water-use")
        cfs = {method: pd.DataFrame({"flow_id": [water_id], "cf": [42.95]})}
        regional = {method: pd.DataFrame({"flow_id": [water_id], "location": ["FR"], "cf": [6.98]})}
        package = ScoringPackageBuilder().build(
            frame,
            cfs,
            regional_cfs=regional,
            col_id_to_location={wheat_aid: "FR"},
        )
        assert method in package.corrections, (
            "correction row should be built for the regionalised method"
        )
        pc = _product_catalog(
            [
                {
                    "database": "agb",
                    "code": "wheat",
                    "name": "wheat",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "wheat-prod")),
                },
            ]
        )
        bio = _bio_catalog(
            [{"database": "bio3", "code": "water", "name": "Water", "categories": ["air"]}]
        )
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        result = d.decompose(("agb", "wheat"), method)

        assert result.global_score == pytest.approx(42.95)
        assert result.correction_score == pytest.approx(6.98 - 42.95)
        assert result.score == pytest.approx(6.98)
        # Flow contributions reflect the *global* CF (the regional
        # adjustment is per activity, not per flow). They sum back to
        # global_score, not score.
        assert result.flow_contributions["contribution"].sum() == pytest.approx(42.95)
        # Activity contributions DO include the correction, so they sum to score.
        assert result.activity_contributions["contribution"].sum() == pytest.approx(6.98)
        # The activity row exposes the regional delta on a column of its own.
        wheat_row = result.activity_contributions.set_index("activity_name").loc["wheat"]
        assert wheat_row["regional_delta"] == pytest.approx(6.98 - 42.95)

    def test_no_regional_correction_leaves_global_and_score_equal(self):
        # When the method has no correction row, score == global_score
        # and correction_score is exactly zero — no spurious adjustment.
        package, pc, bio, method = self._wheat_steel_package()
        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        result = d.decompose(("agb", "wheat"), method)

        assert result.correction_score == 0.0
        assert result.global_score == pytest.approx(result.score)

    def test_supply_zero_activities_excluded(self):
        # An activity that the demand doesn't reach (no path from the chosen
        # product) has supply=0 and cannot contribute. It must not appear in
        # the activity_contributions table.
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {"type": "production", "input": ("agb", "wheat-prod"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
                ],
            ),
            # Disconnected activity — wheat doesn't consume it.
            _activity(
                "agb",
                "orphan",
                "orphan",
                exchanges=[
                    {"type": "production", "input": ("agb", "orphan-prod"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 999.0},
                ],
            ),
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        method = ("ef", "climate")
        cfs = {method: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        package = ScoringPackageBuilder().build(frame, cfs)
        pc = _product_catalog(
            [
                {
                    "database": "agb",
                    "code": "wheat",
                    "name": "wheat",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "wheat-prod")),
                },
                {
                    "database": "agb",
                    "code": "orphan",
                    "name": "orphan",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "orphan-prod")),
                },
            ]
        )
        bio = _bio_catalog(
            [{"database": "bio3", "code": "co2", "name": "CO2", "categories": ["air"]}]
        )

        d = ScoreDecomposer(
            package=package, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        result = d.decompose(("agb", "wheat"), method)

        names = set(result.activity_contributions["activity_name"])
        assert names == {"wheat"}
        # Total still equal to score (orphan doesn't contribute).
        assert result.activity_contributions["contribution"].sum() == pytest.approx(2.0)
