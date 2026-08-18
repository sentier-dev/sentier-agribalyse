"""End-to-end: export -> real bw2data project -> parity, both generations.

Builds the datapackages + importer-sufficient metadata from the tiny
``synthetic_package`` fixture, copies in the standalone importer, then runs it
through ``uv`` against a single Brightway generation so it exercises the genuine
``bw2data`` / ``bw2calc`` stack and proves the reconstructed project reproduces
the export's score (20.5 for the fixture) — legacy AND bw2.5.

Gated behind ``RUN_BW_IMPORT_INTEGRATION=1`` because it provisions isolated
environments via ``uv`` (network + a few seconds per generation)::

    RUN_BW_IMPORT_INTEGRATION=1 pytest tests/integration/bw_import/
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("bw_processing", reason="needs the [bw] extra")

from bw_export.bw_node_types import ActivityMeta, BioMeta
from bw_export.correction_embedder import CorrectionEmbedder
from bw_export.datapackage_writer import DatapackageWriter
from bw_export.metadata_emitter import MetadataEmitter
from tests.fixtures.bw_synthetic import (
    BIO_FLOW_ID,
    CORRECTED_SCORE,
    METHOD,
    PRODUCT_IDS,
)

_IMPORTER = Path(__file__).resolve().parents[3] / "src" / "bw_import" / "import_into_brightway.py"

# Per-generation dependency pins for the user-side environment. The importer
# needs bw2data (to write) + bw2calc (to --verify) + numpy/pandas (to read).
GENERATION_PINS = {
    "legacy": ["bw2data>=3.6,<4", "bw2calc>=1.8,<2", "numpy<2", "pandas"],
    "bw25": ["bw2data>=4.0,<5", "bw2calc>=2.0", "pandas"],
}

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BW_IMPORT_INTEGRATION") != "1",
    reason="set RUN_BW_IMPORT_INTEGRATION=1 to run the uv-provisioned import",
)


def _build_export(synthetic_package, export: Path) -> None:
    embedded = CorrectionEmbedder().embed(synthetic_package)
    written = DatapackageWriter().write(embedded, out_root=export)
    MetadataEmitter().emit(
        out_root=export,
        embedded=embedded,
        written=written,
        activity_resolver=lambda cid: ActivityMeta(
            ("agribalyse-3.2", f"a{cid}"), f"activity {cid}", "kg", "GLO", "ref"
        ),
        bio_resolver=lambda bid: BioMeta(
            ("biosphere3", f"b{bid}"),
            f"flow {bid}",
            ("air",),
            "kg",
            bid != BIO_FLOW_ID,
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
        parity_scores={PRODUCT_IDS[0]: {METHOD: CORRECTED_SCORE}},
        parity_tolerance=1e-6,
        manifest_extra={"content_hash": "synthetic", "ecoinvent_version": "3.9.1"},
        parity={"passed": True, "n_checked": 1, "max_rel_error": 0.0, "tolerance": 1e-6},
    )
    shutil.copy2(_IMPORTER, export / "import_into_brightway.py")


@pytest.mark.parametrize("generation", ["bw25", "legacy"])
def test_importer_reconstructs_project_and_passes_parity(tmp_path, synthetic_package, generation):
    if shutil.which("uv") is None:
        pytest.skip("uv not available")
    export = tmp_path / "export"
    _build_export(synthetic_package, export)
    bw_dir = tmp_path / f"bw-{generation}"

    cmd = ["uv", "run", "--python", "3.11"]
    for pin in GENERATION_PINS[generation]:
        cmd += ["--with", pin]
    cmd += [
        str(export / "import_into_brightway.py"),
        "--bundle-dir",
        str(export),
        "--brightway-dir",
        str(bw_dir),
        "--project",
        f"import-test-{generation}",
        "--verify",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"importer failed:\n{proc.stdout}\n{proc.stderr}"
    assert "Parity OK" in proc.stdout
