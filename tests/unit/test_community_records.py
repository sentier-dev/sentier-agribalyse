"""Run the public collaboration contracts in the repository's existing pytest CI."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "community/tests" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TestContributionRecords = load("community_record_tests", "test_records.py").ContributionTests
TestBuildAudit = load("community_build_tests", "test_build_checks.py").BuildChecks
TestCiConfig = load("community_ci_tests", "test_ci_config.py").CiConfigTests
_site_tests = load("community_site_tests", "test_site.py")
TestNotFoundPage = _site_tests.NotFoundPageTests
TestSiteCheck = _site_tests.SiteCheckTests
