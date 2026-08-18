import pytest

bc = pytest.importorskip("bw2calc", reason="needs the [bw] extra")
bwp = pytest.importorskip("bw_processing", reason="needs the [bw] extra")

from bw_export.correction_embedder import CorrectionEmbedder
from bw_export.datapackage_writer import DatapackageWriter
from tests.fixtures.bw_synthetic import CORRECTED_SCORE, METHOD, PRODUCT_IDS


def _load(path):
    return bwp.load_datapackage(bwp.generic_directory_filesystem(dirpath=path))


def test_written_datapackages_reproduce_corrected_score(tmp_path, synthetic_package):
    embedded = CorrectionEmbedder().embed(synthetic_package)
    result = DatapackageWriter().write(embedded, out_root=tmp_path)

    inv_dp = _load(result.inventory_path)
    method_dp = _load(result.method_paths[METHOD])

    # Demand is keyed by the product (row) id directly.
    lca = bc.LCA({PRODUCT_IDS[0]: 1.0}, data_objs=[inv_dp, method_dp])
    lca.lci()
    lca.lcia()
    # 20.5 = plain score 20 + correction 0.5 (embedded as synthetic flow).
    assert abs(lca.score - CORRECTED_SCORE) < 1e-9


def test_product_ids_are_the_demand_keys(tmp_path, synthetic_package):
    embedded = CorrectionEmbedder().embed(synthetic_package)
    result = DatapackageWriter().write(embedded, out_root=tmp_path)
    assert set(result.product_ids) == set(PRODUCT_IDS)
