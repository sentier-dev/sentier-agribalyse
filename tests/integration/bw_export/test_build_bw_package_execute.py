"""Full ``dds-build-bw-package`` execute() on a tmp project root.

Seeds the run_report + ScoringPackageStore + registry catalogs exactly as a
completed ``dds-link-all`` would, then runs the CLI end-to-end and checks the
export is complete and parity-verified.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

pytest.importorskip("bw2calc", reason="needs the [bw] extra")
pytest.importorskip("bw_processing", reason="needs the [bw] extra")

from cli.build_bw_package import BuildBwPackageCli
from scoring.scoring_package import ScoringPackageStore
from tests.fixtures.bw_synthetic import (
    ACTIVITY_IDS,
    METHOD,
    PRODUCT_IDS,
    make_synthetic_scoring_package,
)


@pytest.fixture
def linked_settings(settings):
    """Settings whose tmp root looks like a completed link run."""
    package = make_synthetic_scoring_package()
    ScoringPackageStore(root=settings.paths.scoring_packages_root).write(package)
    settings.paths.dashboard_run_report.write_text(
        json.dumps({"stages": {"scoring_package": {"content_hash": package.content_hash}}})
    )
    pd.DataFrame(
        {
            "product_id": PRODUCT_IDS,
            "database": ["agribalyse-3.2"] * 2,
            "code": ["c101", "c102"],
            "name": ["prod one", "prod two"],
            "type": ["product"] * 2,
            "unit": ["kg"] * 2,
        }
    ).to_parquet(settings.paths.registry_product_catalog)
    pd.DataFrame({"database": [], "code": [], "name": [], "categories": [], "unit": []}).to_parquet(
        settings.paths.registry_biosphere_catalog
    )
    pd.DataFrame(
        {
            "database": [],
            "code": [],
            "name": [],
            "unit": [],
            "location": [],
            "reference_product": [],
        }
    ).to_parquet(settings.paths.registry_ecoinvent_catalog)
    pd.DataFrame(
        {
            "activity_id": ACTIVITY_IDS,
            "database": ["agribalyse-3.2"] * 2,
            "code": ["c101", "c102"],
            "name": ["prod one", "prod two"],
            "unit": ["kg"] * 2,
            "location": ["FR"] * 2,
            "reference_product": ["prod one", "prod two"],
        }
    ).to_parquet(settings.paths.registry_activity_catalog)
    return settings


def test_execute_writes_verified_export(tmp_path, linked_settings, capsys):
    out = tmp_path / "export"
    BuildBwPackageCli(settings=linked_settings).execute(
        argparse.Namespace(out=out, parity_n=2, parity_full=False, skip_parity=False)
    )

    manifest = json.loads((out / "metadata" / "manifest.json").read_text())
    assert manifest["parity"]["passed"] is True
    assert manifest["content_hash"] == "synthetic"
    assert manifest["counts"] == {
        "n_activities": 2,
        "n_products": 2,
        "n_biosphere_flows": 2,  # 1 real + 1 synthetic AWARE row
        "n_synthetic_correction_flows": 1,
        "n_methods": 1,
    }

    # The importer is copied in, and metadata names every node.
    assert (out / "import_into_brightway.py").is_file()
    acts = pd.read_parquet(out / "metadata" / "activities.parquet")
    assert set(acts["name"]) == {"prod one", "prod two"}  # via activity_catalog
    samples = json.loads((out / "metadata" / "parity_samples.json").read_text())
    assert [s["product_id"] for s in samples["samples"]] == PRODUCT_IDS
    assert samples["samples"][0]["expected"][0][0] == list(METHOD)

    assert "parity check: PASS" in capsys.readouterr().out


def test_execute_skip_parity_marks_manifest(tmp_path, linked_settings):
    out = tmp_path / "export"
    BuildBwPackageCli(settings=linked_settings).execute(
        argparse.Namespace(out=out, parity_n=2, parity_full=False, skip_parity=True)
    )
    manifest = json.loads((out / "metadata" / "manifest.json").read_text())
    assert manifest["parity"] == {"skipped": True}
    samples = json.loads((out / "metadata" / "parity_samples.json").read_text())
    assert samples["samples"] == []  # no trusted scores embedded
