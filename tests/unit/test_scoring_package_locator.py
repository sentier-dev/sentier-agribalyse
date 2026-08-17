"""``ScoringPackageLocator`` — run_report → cached ScoringPackage resolution."""

from __future__ import annotations

import json

import pytest

from scoring.scoring_package import ScoringPackageStore
from scoring.scoring_package_locator import ScoringPackageLocator
from tests.fixtures.bw_synthetic import make_synthetic_scoring_package


def _write_run_report(settings, content_hash: str) -> None:
    settings.paths.dashboard_run_report.write_text(
        json.dumps({"stages": {"scoring_package": {"content_hash": content_hash}}})
    )


def test_missing_run_report_is_actionable(settings):
    with pytest.raises(FileNotFoundError, match="dds-link-all"):
        ScoringPackageLocator(settings=settings).content_hash()


def test_run_report_without_hash_is_actionable(settings):
    settings.paths.dashboard_run_report.write_text(json.dumps({"stages": {}}))
    with pytest.raises(KeyError, match="content_hash"):
        ScoringPackageLocator(settings=settings).content_hash()


def test_loads_the_package_the_report_points_at(settings):
    package = make_synthetic_scoring_package()
    ScoringPackageStore(root=settings.paths.scoring_packages_root).write(package)
    _write_run_report(settings, package.content_hash)

    loaded = ScoringPackageLocator(settings=settings).load()
    assert loaded.content_hash == package.content_hash
    assert loaded.technosphere.matrix.shape == (2, 2)
    assert list(loaded.methods) == list(package.methods)
