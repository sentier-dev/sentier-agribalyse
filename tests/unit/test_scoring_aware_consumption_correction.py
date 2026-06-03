"""Tests for ``AwareConsumptionCorrectionBuilder``.

Toy matrix layout:
* 3 activities: ``IRRIGATION`` (BR, res>>ret), ``ELECTRICITY`` (FR,
  closely balanced), ``DEHYDRATION`` (DE, ret>>res — should NOT be
  characterised).
* 2 biosphere flows:
  - ``WATER_RIVER`` row 0 — ``("water, river", ["natural resource", "in water"])``
  - ``WATER_SURFACE`` row 1 — ``("water", ["water", "surface water"])``
* B matrix:
  - IRRIGATION: river=10, surface=0   (100% asymmetric, +10 net)
  - ELECTRICITY: river=5, surface=4.9 (2% asymmetric, balanced — EXCLUDED)
  - DEHYDRATION: river=0, surface=10  (100% asymmetric but res < ret — EXCLUDED)
* Regional CFs: BR=2.43, FR=6.98 (DE not in table → fallback 7.0).
* Expected delta[IRRIGATION] = 2.43 * 10 = 24.3
* Expected delta[ELECTRICITY] = 0 (asymmetric gate fails)
* Expected delta[DEHYDRATION] = 0 (res < ret)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from scoring.aware_consumption_correction import AwareConsumptionCorrectionBuilder
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.matrix_builder import BuiltMatrix


def _make_bio_catalog() -> pd.DataFrame:
    """Two water bio3 rows that match the builder's target lists."""
    return pd.DataFrame(
        {
            "database": ["ecoinvent-3.9.1-biosphere", "ecoinvent-3.9.1-biosphere"],
            "code": ["w-river-uuid", "w-surface-uuid"],
            "name": ["Water, river", "Water"],
            "categories": [
                ["natural resource", "in water"],
                ["water", "surface water"],
            ],
        }
    )


@pytest.fixture
def toy():
    """Build the toy matrices + lookup maps used across tests."""
    bio = _make_bio_catalog()
    # Map each (db, code) to the flow id the matrix builder would produce.
    river_id = ExchangeFrameBuilder.flow_id_for(("ecoinvent-3.9.1-biosphere", "w-river-uuid"))
    surface_id = ExchangeFrameBuilder.flow_id_for(("ecoinvent-3.9.1-biosphere", "w-surface-uuid"))

    # 3 activity columns
    irr, elec, dehy = 1001, 1002, 1003
    technosphere = BuiltMatrix(
        # Identity-like 3x3; values don't matter for this test since we
        # only build the correction row, not the score.
        matrix=sp.csr_matrix(np.eye(3, dtype="float64")),
        row_id_to_idx={2001: 0, 2002: 1, 2003: 2},
        col_id_to_idx={irr: 0, elec: 1, dehy: 2},
    )
    # B rows: 0 = river (resource), 1 = surface water (return)
    biosphere = BuiltMatrix(
        matrix=sp.csr_matrix(
            np.array(
                [
                    [10.0, 5.0, 0.0],  # river
                    [0.0, 4.9, 10.0],  # surface
                ]
            )
        ),
        row_id_to_idx={river_id: 0, surface_id: 1},
        col_id_to_idx={irr: 0, elec: 1, dehy: 2},
    )
    col_id_to_location = {irr: "BR", elec: "FR", dehy: "DE"}
    regional_cf_by_location = {"BR": 2.43, "FR": 6.98}  # DE missing → fallback
    return (
        bio,
        technosphere,
        biosphere,
        col_id_to_location,
        regional_cf_by_location,
        irr,
        elec,
        dehy,
    )


def test_resource_dominant_activity_gets_regional_correction(toy):
    bio, ts, bs, loc_map, reg_cf, *_ = toy
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=bio,
        regional_cf_by_location=reg_cf,
        col_id_to_location=loc_map,
    )
    dense = row.toarray().ravel()
    assert dense.shape == (3,)
    # IRRIGATION: 2.43 * (10 - 0) = 24.3
    np.testing.assert_allclose(dense[0], 2.43 * 10.0, rtol=1e-9)


def test_balanced_activity_excluded(toy):
    bio, ts, bs, loc_map, reg_cf, *_ = toy
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=bio,
        regional_cf_by_location=reg_cf,
        col_id_to_location=loc_map,
    )
    dense = row.toarray().ravel()
    # ELECTRICITY: 2% asymmetric → fails the asymmetry gate.
    assert abs(dense[1]) < 1e-12


def test_return_dominant_activity_excluded(toy):
    """DEHYDRATION has 100% asymmetry BUT res < ret — must be excluded."""
    bio, ts, bs, loc_map, reg_cf, *_ = toy
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=bio,
        regional_cf_by_location=reg_cf,
        col_id_to_location=loc_map,
    )
    dense = row.toarray().ravel()
    assert abs(dense[2]) < 1e-12


def test_missing_location_uses_fallback_cf(toy):
    """Activity at a location not in the regional CF table uses
    FALLBACK_CF (7.0). We swap IRRIGATION's location to one that's
    NOT in the table and confirm the fallback applies."""
    bio, ts, bs, _, reg_cf, irr, elec, dehy = toy
    loc_map = {irr: "ZZ", elec: "FR", dehy: "DE"}  # ZZ not in reg_cf
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=bio,
        regional_cf_by_location=reg_cf,
        col_id_to_location=loc_map,
    )
    dense = row.toarray().ravel()
    assert abs(dense[0] - AwareConsumptionCorrectionBuilder.FALLBACK_CF * 10.0) < 1e-9


def test_unlocated_activity_uses_fallback_cf(toy):
    """An activity with NO entry in col_id_to_location (e.g. GLO) uses
    the fallback CF — emulates the GLO/RoW aggregate-region case."""
    bio, ts, bs, _, reg_cf, _irr, elec, dehy = toy
    # Drop IRRIGATION from the location map entirely.
    loc_map = {elec: "FR", dehy: "DE"}
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=bio,
        regional_cf_by_location=reg_cf,
        col_id_to_location=loc_map,
    )
    dense = row.toarray().ravel()
    assert abs(dense[0] - AwareConsumptionCorrectionBuilder.FALLBACK_CF * 10.0) < 1e-9


def test_min_net_threshold_drops_tiny_contributions():
    """Per-activity net consumption below MIN_NET_M3 (0.05) is ignored
    even when asymmetry is 100%. Protects against numerical noise from
    near-zero flows."""
    bio = _make_bio_catalog()
    river_id = ExchangeFrameBuilder.flow_id_for(("ecoinvent-3.9.1-biosphere", "w-river-uuid"))
    surface_id = ExchangeFrameBuilder.flow_id_for(("ecoinvent-3.9.1-biosphere", "w-surface-uuid"))
    irr = 99
    ts = BuiltMatrix(
        matrix=sp.csr_matrix(np.eye(1, dtype="float64")),
        row_id_to_idx={1: 0},
        col_id_to_idx={irr: 0},
    )
    # 0.01 m³ res, 0 ret → 100% asymmetric but below MIN_NET_M3=0.05
    bs = BuiltMatrix(
        matrix=sp.csr_matrix(np.array([[0.01], [0.0]])),
        row_id_to_idx={river_id: 0, surface_id: 1},
        col_id_to_idx={irr: 0},
    )
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=bio,
        regional_cf_by_location={"BR": 2.43},
        col_id_to_location={irr: "BR"},
    )
    assert row.nnz == 0


def test_empty_biosphere_catalog_returns_zero_row(toy):
    """No catalog rows → no flow resolution → empty correction."""
    _, ts, bs, loc_map, reg_cf, *_ = toy
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=pd.DataFrame(columns=["database", "code", "name", "categories"]),
        regional_cf_by_location=reg_cf,
        col_id_to_location=loc_map,
    )
    assert row.shape == (1, ts.matrix.shape[1])
    assert row.nnz == 0


def test_sub_regional_location_collapses_to_country(toy):
    """``CA-QC`` should look up ``CA`` in the regional table."""
    bio, ts, bs, _, _, irr, elec, dehy = toy
    loc_map = {irr: "CA-QC", elec: "FR", dehy: "DE"}
    reg_cf = {"CA": 1.42, "FR": 6.98}
    row = AwareConsumptionCorrectionBuilder().build(
        biosphere=bs,
        technosphere=ts,
        biosphere_catalog=bio,
        regional_cf_by_location=reg_cf,
        col_id_to_location=loc_map,
    )
    dense = row.toarray().ravel()
    assert abs(dense[0] - 1.42 * 10.0) < 1e-9


def test_thresholds_are_class_constants():
    """Sanity-check the validated tuning values stay in-class."""
    cls = AwareConsumptionCorrectionBuilder
    assert cls.ASYMMETRY_THRESHOLD == 0.95
    assert cls.MIN_NET_M3 == 0.05
    assert cls.FALLBACK_CF == 7.0
