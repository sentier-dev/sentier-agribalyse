"""Unit tests for ``config.paths.Paths``."""

from __future__ import annotations

from config import Settings


class TestDashboardPaths:
    def test_cf_comparison_csv_lives_under_dashboard(self) -> None:
        paths = Settings().paths
        assert paths.dashboard_cf_comparison_csv == paths.dashboard / "cf_comparison.csv"

    def test_cf_comparison_by_code_lives_under_registry(self) -> None:
        paths = Settings().paths
        assert (
            paths.registry_cf_comparison_by_code
            == paths.registry / "cf_comparison_by_code.parquet"
        )
