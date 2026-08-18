"""Tiny, fully-known ``ScoringPackage`` for the bw_export / bw_import tests.

The system is small enough to solve by hand, so every test asserts against a
paper-computed truth (see ``tests/integration/bw_export/test_bw2calc_api.py``
for the worked example):

    A = [[1, -0.5], [0, 1]]   rows = products (101, 102), cols = activities (201, 202)
    B = [[2, 3]]              row = biosphere flow 301
    CF(301) = 10.0
    correction row = [0.5, 0.0]

    demand {101: 1} -> supply [1, 0] -> inventory [2] -> plain score 20.0
    corrected score = 20.0 + correction @ supply = 20.5
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from scoring.matrix_builder import BuiltMatrix
from scoring.scoring_package import ScoringPackage

# Product (row) ids and activity (col) ids are deliberately DIFFERENT
# integers, mirroring the real system where row=product, col=activity.
PRODUCT_IDS = [101, 102]
ACTIVITY_IDS = [201, 202]
BIO_FLOW_ID = 301
METHOD = ("ecoinvent-3.9.1", "EF v3.1", "demo", "indicator")
CORRECTED_SCORE = 20.5


def make_synthetic_scoring_package() -> ScoringPackage:
    """One invertible 2x2 system with a single corrected method."""
    a = sp.csr_matrix(np.array([[1.0, -0.5], [0.0, 1.0]]))
    technosphere = BuiltMatrix(
        matrix=a,
        row_id_to_idx={PRODUCT_IDS[0]: 0, PRODUCT_IDS[1]: 1},
        col_id_to_idx={ACTIVITY_IDS[0]: 0, ACTIVITY_IDS[1]: 1},
    )
    b = sp.csr_matrix(np.array([[2.0, 3.0]]))
    biosphere = BuiltMatrix(
        matrix=b,
        row_id_to_idx={BIO_FLOW_ID: 0},
        col_id_to_idx={ACTIVITY_IDS[0]: 0, ACTIVITY_IDS[1]: 1},
    )
    q = sp.csr_matrix(np.array([[10.0]]))
    correction = sp.csr_matrix(np.array([[0.5, 0.0]]))
    return ScoringPackage(
        technosphere=technosphere,
        biosphere=biosphere,
        methods={METHOD: q},
        corrections={METHOD: correction},
        content_hash="synthetic",
    )
