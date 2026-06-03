"""Tests for ``RegionalCorrectionBuilder`` + ``ScoringPackage`` regional path.

Toy matrix layout used here:

* 2 activities ``A`` (BR) and ``B`` (CY) sharing 1 product each.
* 1 biosphere flow ``water`` with B[water, A]=1.0 and B[water, B]=2.0.
* Global water CF = +42.95; regional CFs: BR=2.43, CY=74.30.
* Expected per-activity correction:
    delta[A] = (2.43 - 42.95) * 1.0 = -40.52
    delta[B] = (74.30 - 42.95) * 2.0 = +62.70
* The scorer adds ``correction @ supply`` to the baseline; one-unit
  demand for product ``A`` gives supply=[1.0, 0.0] so the corrected
  score gets delta[A] added; demand for ``B`` gets delta[B].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from scoring.matrix_builder import BuiltMatrix
from scoring.native_scorer import NativeLciaScorer
from scoring.regional_correction import RegionalCorrectionBuilder
from scoring.scoring_package import ScoringPackage, ScoringPackageStore

# ----------------------------------------------------------------------------
# Fixtures


WATER_METHOD = (
    "ecoinvent-3.9.1",
    "EF v3.1",
    "water use",
    "user deprivation potential (deprivation-weighted water consumption)",
)


@pytest.fixture
def toy_matrices():
    """Build a 2-product / 2-activity / 1-flow toy ScoringPackage skeleton.

    Product/activity ids are arbitrary integers (the matrix builder
    treats them as opaque hashes).
    """
    # Activity ids
    act_a, act_b = 101, 102
    # Product ids — A produces P_A, B produces P_B
    p_a, p_b = 201, 202
    # Flow id for ``water``
    water_id = 301

    technosphere = BuiltMatrix(
        # Identity 2x2 so supply = demand for each activity
        matrix=sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])),
        row_id_to_idx={p_a: 0, p_b: 1},
        col_id_to_idx={act_a: 0, act_b: 1},
    )
    biosphere = BuiltMatrix(
        # B[water, A]=1, B[water, B]=2
        matrix=sp.csr_matrix(np.array([[1.0, 2.0]])),
        row_id_to_idx={water_id: 0},
        col_id_to_idx={act_a: 0, act_b: 1},
    )
    return technosphere, biosphere, water_id, act_a, act_b, p_a, p_b


# ----------------------------------------------------------------------------
# RegionalCorrectionBuilder


def test_regional_correction_subtracts_global_and_scales_by_b(toy_matrices):
    technosphere, biosphere, water_id, act_a, act_b, _, _ = toy_matrices
    global_cf_df = pd.DataFrame({"flow_id": [water_id], "cf": [42.95]})
    regional_cf_df = pd.DataFrame(
        {
            "flow_id": [water_id, water_id],
            "location": ["BR", "CY"],
            "cf": [2.43, 74.30],
        }
    )
    col_id_to_location = {act_a: "BR", act_b: "CY"}

    row = RegionalCorrectionBuilder().build(
        global_cf_df=global_cf_df,
        regional_cf_df=regional_cf_df,
        biosphere=biosphere,
        technosphere=technosphere,
        col_id_to_location=col_id_to_location,
    )
    dense = row.toarray().ravel()
    assert dense.shape == (2,)
    expected = np.array([(2.43 - 42.95) * 1.0, (74.30 - 42.95) * 2.0])
    np.testing.assert_allclose(dense, expected, rtol=1e-9)


def test_regional_correction_skips_unmatched_activities(toy_matrices):
    """Activities at a location with no regional CF (e.g. ``RoW``) get
    zero correction — they keep the global CF via the baseline path.
    """
    technosphere, biosphere, water_id, act_a, act_b, _, _ = toy_matrices
    global_cf_df = pd.DataFrame({"flow_id": [water_id], "cf": [42.95]})
    regional_cf_df = pd.DataFrame({"flow_id": [water_id], "location": ["BR"], "cf": [2.43]})
    col_id_to_location = {act_a: "BR", act_b: "RoW"}

    row = RegionalCorrectionBuilder().build(
        global_cf_df=global_cf_df,
        regional_cf_df=regional_cf_df,
        biosphere=biosphere,
        technosphere=technosphere,
        col_id_to_location=col_id_to_location,
    )
    dense = row.toarray().ravel()
    assert abs(dense[1]) < 1e-12, "RoW activity must not be corrected"
    assert abs(dense[0] - ((2.43 - 42.95) * 1.0)) < 1e-9


def test_regional_correction_handles_missing_global_cf(toy_matrices):
    """If the regional table characterises a flow the global table
    doesn't carry, delta = regional - 0 = regional. This shouldn't
    raise."""
    technosphere, biosphere, water_id, act_a, _, _, _ = toy_matrices
    regional_cf_df = pd.DataFrame({"flow_id": [water_id], "location": ["BR"], "cf": [2.43]})
    row = RegionalCorrectionBuilder().build(
        global_cf_df=pd.DataFrame({"flow_id": [], "cf": []}),
        regional_cf_df=regional_cf_df,
        biosphere=biosphere,
        technosphere=technosphere,
        col_id_to_location={act_a: "BR"},
    )
    dense = row.toarray().ravel()
    assert abs(dense[0] - (2.43 * 1.0)) < 1e-9


def test_regional_correction_empty_inputs_return_zero(toy_matrices):
    """No regional CFs → an all-zero correction row of the right shape."""
    technosphere, biosphere, _, _, _, _, _ = toy_matrices
    row = RegionalCorrectionBuilder().build(
        global_cf_df=pd.DataFrame({"flow_id": [], "cf": []}),
        regional_cf_df=pd.DataFrame({"flow_id": [], "location": [], "cf": []}),
        biosphere=biosphere,
        technosphere=technosphere,
        col_id_to_location={},
    )
    assert row.shape == (1, 2)
    assert row.nnz == 0


# ----------------------------------------------------------------------------
# ScoringPackage round-trip


def test_scoring_package_persists_corrections(tmp_path, toy_matrices):
    technosphere, biosphere, _, _, _, _, _ = toy_matrices
    method_q = sp.csr_matrix(np.array([[42.95]]))
    correction = sp.csr_matrix(np.array([[-40.52, 62.70]]))
    pkg = ScoringPackage(
        technosphere=technosphere,
        biosphere=biosphere,
        methods={WATER_METHOD: method_q},
        corrections={WATER_METHOD: correction},
        content_hash="toy_hash",
    )
    store = ScoringPackageStore(root=tmp_path)
    store.write(pkg)
    loaded = store.read("toy_hash")

    assert WATER_METHOD in loaded.corrections
    np.testing.assert_allclose(
        loaded.corrections[WATER_METHOD].toarray(),
        correction.toarray(),
        rtol=1e-12,
    )


def test_scoring_package_backward_compatible_without_corrections(tmp_path, toy_matrices):
    """A package written without ``corrections`` (older builds) must
    still load cleanly with an empty corrections dict."""
    technosphere, biosphere, _, _, _, _, _ = toy_matrices
    pkg = ScoringPackage(
        technosphere=technosphere,
        biosphere=biosphere,
        methods={WATER_METHOD: sp.csr_matrix(np.array([[42.95]]))},
        content_hash="no_corrections",
    )
    store = ScoringPackageStore(root=tmp_path)
    store.write(pkg)
    loaded = store.read("no_corrections")
    assert loaded.corrections == {}


# ----------------------------------------------------------------------------
# NativeLciaScorer with regional correction


def test_native_scorer_applies_correction_to_score(toy_matrices):
    """The scorer adds ``correction @ supply`` to ``q @ B @ supply`` —
    so an activity in a location with a regional CF different from the
    global gets its score shifted by ``(regional_cf - global_cf) * B``.
    """
    technosphere, biosphere, _, _act_a, _act_b, p_a, p_b = toy_matrices
    q = sp.csr_matrix(np.array([[42.95]]))
    correction = sp.csr_matrix(np.array([[-40.52, 62.70]]))
    pkg = ScoringPackage(
        technosphere=technosphere,
        biosphere=biosphere,
        methods={WATER_METHOD: q},
        corrections={WATER_METHOD: correction},
        content_hash="toy",
    )
    scorer = NativeLciaScorer(package=pkg, use_pardiso=False)

    # Score product A — supply = [1, 0]
    out_a = scorer.score(
        products=[(("agb", "prod-A"), p_a)],
        methods=[WATER_METHOD],
    )
    score_a = out_a[("agb", "prod-A")].scores[WATER_METHOD]
    # baseline = q @ B @ supply = 42.95 * 1.0 = 42.95
    # correction = -40.52 * 1.0 = -40.52
    # total = 2.43
    assert abs(score_a - 2.43) < 1e-9

    # Score product B — supply = [0, 1]
    out_b = scorer.score(
        products=[(("agb", "prod-B"), p_b)],
        methods=[WATER_METHOD],
    )
    score_b = out_b[("agb", "prod-B")].scores[WATER_METHOD]
    # baseline = q @ B @ supply = 42.95 * 2.0 = 85.90
    # correction = 62.70 * 1.0 = 62.70
    # total = 148.60
    assert abs(score_b - 148.60) < 1e-9


def test_native_scorer_without_correction_unchanged(toy_matrices):
    """Methods absent from ``corrections`` keep the legacy
    ``q @ B @ supply`` value exactly."""
    technosphere, biosphere, _, _, _, p_a, _ = toy_matrices
    q = sp.csr_matrix(np.array([[42.95]]))
    pkg = ScoringPackage(
        technosphere=technosphere,
        biosphere=biosphere,
        methods={WATER_METHOD: q},
        content_hash="nocorr",
    )
    scorer = NativeLciaScorer(package=pkg, use_pardiso=False)
    out = scorer.score(
        products=[(("agb", "prod-A"), p_a)],
        methods=[WATER_METHOD],
    )
    score = out[("agb", "prod-A")].scores[WATER_METHOD]
    # Just baseline: 42.95 * 1.0
    assert abs(score - 42.95) < 1e-9
