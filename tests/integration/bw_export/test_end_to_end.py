import json

import pandas as pd
import pytest

pytest.importorskip("bw2calc", reason="needs the [bw] extra")
pytest.importorskip("bw_processing", reason="needs the [bw] extra")

from bw_export.bw_node_types import ActivityMeta, BioMeta
from bw_export.correction_embedder import CorrectionEmbedder
from bw_export.datapackage_writer import DatapackageWriter
from bw_export.metadata_emitter import MetadataEmitter
from bw_export.parity_verifier import ParityVerifier
from tests.fixtures.bw_synthetic import BIO_FLOW_ID, METHOD, PRODUCT_IDS


def test_full_export_chain_passes_parity(tmp_path, synthetic_package):
    embedded = CorrectionEmbedder().embed(synthetic_package)
    written = DatapackageWriter().write(embedded, out_root=tmp_path)
    result = ParityVerifier(tolerance=1e-9).verify(
        package=synthetic_package,
        written=written,
        product_ids=list(PRODUCT_IDS),
        methods=[METHOD],
    )
    assert result.passed
    assert result.n_checked == len(PRODUCT_IDS)

    MetadataEmitter().emit(
        out_root=tmp_path,
        embedded=embedded,
        written=written,
        activity_resolver=lambda cid: ActivityMeta(
            ("db", f"act-{cid}"), f"activity {cid}", "kg", "GLO", "ref"
        ),
        bio_resolver=lambda bid: BioMeta(
            ("bio", f"flow-{bid}"), f"flow {bid}", ("air",), "kg", bid != BIO_FLOW_ID
        ),
        product_catalog=pd.DataFrame(
            {
                "database": ["agribalyse-3.2"] * 2,
                "code": ["c101", "c102"],
                "name": ["one", "two"],
                "type": ["product"] * 2,
                "unit": ["kg"] * 2,
                "product_id": list(PRODUCT_IDS),
            }
        ),
        parity_scores={PRODUCT_IDS[0]: {METHOD: result.max_rel_error}},
        parity_tolerance=result.tolerance,
        manifest_extra={"content_hash": "synthetic", "ecoinvent_version": "3.9.1"},
        parity={
            "passed": result.passed,
            "n_checked": result.n_checked,
            "max_rel_error": result.max_rel_error,
            "tolerance": result.tolerance,
        },
    )
    manifest = json.loads((tmp_path / "metadata" / "manifest.json").read_text())
    assert manifest["parity"]["passed"] is True
    # DatapackageWriter writes the inventory as a directory (not a zip).
    assert (tmp_path / "inventory").is_dir()
    assert (tmp_path / "run_example.py").exists()
