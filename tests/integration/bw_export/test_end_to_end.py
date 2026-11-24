import json
import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("bw2calc", reason="needs the [bw] extra")
pytest.importorskip("bw_processing", reason="needs the [bw] extra")

from bw_export.bw_node_types import ActivityMeta, BioMeta
from bw_export.correction_embedder import CorrectionEmbedder
from bw_export.datapackage_writer import DatapackageWriter
from bw_export.metadata_emitter import MetadataEmitter
from bw_export.parity_verifier import ParityVerifier
from tests.fixtures.bw_synthetic import ACTIVITY_IDS, BIO_FLOW_ID, METHOD, PRODUCT_IDS


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


def test_the_example_prints_the_name_of_an_ecoinvent_product(tmp_path, synthetic_package):
    """In a real export the first product row is an ecoinvent product, absent
    from the product catalog. The example scores it and must print its name,
    not ``<NA>``."""
    embedded = CorrectionEmbedder().embed(synthetic_package)
    written = DatapackageWriter().write(embedded, out_root=tmp_path)
    MetadataEmitter().emit(
        out_root=tmp_path,
        embedded=embedded,
        written=written,
        activity_resolver=lambda cid: ActivityMeta(
            ("ecoinvent-3.9.1-cutoff", f"act-{cid}"), f"market for x {cid}", "kg", "GLO", f"x {cid}"
        ),
        bio_resolver=lambda bid: BioMeta(
            ("bio", f"flow-{bid}"), f"flow {bid}", ("air",), "kg", bid != BIO_FLOW_ID
        ),
        # Only the second product is in the catalog, as the AGRIBALYSE ones are.
        product_catalog=pd.DataFrame(
            {
                "database": ["agribalyse-3.2"],
                "code": ["c102"],
                "name": ["two"],
                "type": ["product"],
                "unit": ["kg"],
                "product_id": [PRODUCT_IDS[1]],
            }
        ),
        parity_scores={},
        parity_tolerance=1e-9,
        manifest_extra={"content_hash": "synthetic", "ecoinvent_version": "3.9.1"},
        parity={"passed": True, "n_checked": 0, "max_rel_error": 0.0, "tolerance": 1e-9},
    )
    out = subprocess.run(
        [sys.executable, "run_example.py"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout
    assert out.startswith(f"'x {ACTIVITY_IDS[0]}' | "), out
