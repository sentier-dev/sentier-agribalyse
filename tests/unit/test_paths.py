"""Unit tests for ``config.paths.Paths``."""

from __future__ import annotations

from config import Settings


class TestDashboardPaths:
    def test_dashboard_cf_stats_csv_lives_under_dashboard(self) -> None:
        paths = Settings().paths
        assert paths.dashboard_cf_stats_csv == paths.dashboard / "cf_stats.csv"
