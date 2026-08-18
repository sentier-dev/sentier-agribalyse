import dataclasses

import numpy as np

from bw_export.correction_embedder import CorrectionEmbedder, EmbeddedInventory
from tests.fixtures.bw_synthetic import BIO_FLOW_ID, METHOD


def test_embed_adds_one_synthetic_flow_per_corrected_method(synthetic_package):
    result = CorrectionEmbedder().embed(synthetic_package)
    assert isinstance(result, EmbeddedInventory)
    # One real flow + one synthetic correction flow.
    assert result.biosphere.shape == (2, 2)
    # The synthetic flow id is new, not colliding with existing ids.
    assert len(result.synthetic_flow_ids) == 1
    syn_id = result.synthetic_flow_ids[METHOD]
    assert syn_id not in synthetic_package.biosphere.row_id_to_idx
    assert syn_id not in synthetic_package.technosphere.row_id_to_idx
    assert syn_id not in synthetic_package.technosphere.col_id_to_idx


def test_synthetic_row_equals_correction_vector(synthetic_package):
    result = CorrectionEmbedder().embed(synthetic_package)
    syn_id = result.synthetic_flow_ids[METHOD]
    syn_idx = result.biosphere_row_id_to_idx[syn_id]
    row = result.biosphere.tocsr()[syn_idx].toarray().ravel()
    np.testing.assert_allclose(row, [0.5, 0.0])


def test_method_cf_includes_synthetic_flow_with_cf_one(synthetic_package):
    result = CorrectionEmbedder().embed(synthetic_package)
    syn_id = result.synthetic_flow_ids[METHOD]
    cfs = result.method_cfs[METHOD]  # dict flow_id -> cf
    assert cfs[BIO_FLOW_ID] == 10.0
    assert cfs[syn_id] == 1.0


def test_no_corrections_leaves_biosphere_unchanged(synthetic_package):
    pkg = dataclasses.replace(synthetic_package, corrections={})
    result = CorrectionEmbedder().embed(pkg)
    assert result.biosphere.shape == (1, 2)
    assert result.synthetic_flow_ids == {}
