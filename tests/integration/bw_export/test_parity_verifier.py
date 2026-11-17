import dataclasses

import numpy as np
import pytest
from scipy import sparse as sp

pytest.importorskip("bw2calc", reason="needs the [bw] extra")
pytest.importorskip("bw_processing", reason="needs the [bw] extra")

from bw_export.correction_embedder import CorrectionEmbedder
from bw_export.datapackage_writer import DatapackageWriter
from bw_export.parity_verifier import ParityResult, ParityVerifier
from scoring.matrix_builder import BuiltMatrix
from tests.fixtures.bw_synthetic import METHOD, PRODUCT_IDS


def _write(tmp_path, package):
    embedded = CorrectionEmbedder().embed(package)
    return embedded, DatapackageWriter().write(embedded, out_root=tmp_path)


def test_parity_passes_for_correct_package(tmp_path, synthetic_package):
    _, written = _write(tmp_path, synthetic_package)
    result = ParityVerifier(tolerance=1e-9).verify(
        package=synthetic_package,
        written=written,
        product_ids=[PRODUCT_IDS[0]],
        methods=[METHOD],
    )
    assert isinstance(result, ParityResult)
    assert result.passed
    assert result.n_checked == 1
    assert result.max_rel_error < 1e-9


def test_square_check_rejects_non_square(synthetic_package):
    bad = dataclasses.replace(
        synthetic_package,
        technosphere=BuiltMatrix(
            matrix=sp.csr_matrix(np.array([[1.0, 0.0]])),  # 1x2, not square
            row_id_to_idx={101: 0},
            col_id_to_idx={201: 0, 202: 1},
        ),
    )
    with pytest.raises(ValueError, match="not square"):
        ParityVerifier().assert_square(bad)


class _FixedScores:
    """Stands in for the native scorer: returns the scores a test names."""

    def __init__(self, scores):
        self._scores = scores

    def __call__(self, package, use_pardiso=True):
        return self

    def score(self, products, methods):
        return {key: type("R", (), {"scores": dict(self._scores)})() for key, _ in products}


def _cancelling_package(synthetic_package):
    """Two flows with opposite factors and equal amounts: the method's score is exactly 0."""
    biosphere = BuiltMatrix(
        matrix=sp.csr_matrix(np.array([[2.0, 3.0], [2.0, 3.0]])),
        row_id_to_idx={301: 0, 302: 1},
        col_id_to_idx=synthetic_package.biosphere.col_id_to_idx,
    )
    return dataclasses.replace(
        synthetic_package,
        biosphere=biosphere,
        methods={METHOD: sp.csr_matrix(np.array([[10.0, -10.0]]))},
        corrections={},
    )


def _verify(tmp_path, package):
    _, written = _write(tmp_path, package)
    return written, ParityVerifier(tolerance=1e-9).verify(
        package=package, written=written, product_ids=[PRODUCT_IDS[0]], methods=[METHOD]
    )


def test_a_rounding_residue_where_terms_cancel_passes(tmp_path, synthetic_package, monkeypatch):
    monkeypatch.setattr("bw_export.parity_verifier.NativeLciaScorer", _FixedScores({METHOD: 1e-15}))

    _, result = _verify(tmp_path, _cancelling_package(synthetic_package))

    assert result.passed
    assert result.max_rel_error < 1e-9


def test_a_wrong_score_where_terms_cancel_fails(tmp_path, synthetic_package, monkeypatch):
    # One of the two cancelling terms is 20; an error of that size is no residue.
    monkeypatch.setattr("bw_export.parity_verifier.NativeLciaScorer", _FixedScores({METHOD: 20.0}))

    _, result = _verify(tmp_path, _cancelling_package(synthetic_package))

    assert not result.passed


def test_a_score_off_by_one_percent_fails(tmp_path, synthetic_package, monkeypatch):
    monkeypatch.setattr(
        "bw_export.parity_verifier.NativeLciaScorer", _FixedScores({METHOD: 20.5 * 1.01})
    )

    _, result = _verify(tmp_path, synthetic_package)

    assert not result.passed
    assert result.max_rel_error == pytest.approx(0.01 / 1.01, rel=1e-6)
