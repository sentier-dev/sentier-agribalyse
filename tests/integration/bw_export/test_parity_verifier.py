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
