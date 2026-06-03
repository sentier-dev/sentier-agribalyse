"""Unit tests for ``config.paths`` and ``config.settings``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from config import Paths, Settings


class TestPaths:
    def test_default_root_resolves_to_repo_root(self):
        # When no override is given Paths picks the repo containing src/config/paths.py.
        p = Paths()
        assert (p.package_root / "src" / "config" / "paths.py").exists()

    def test_overridden_root_propagates_through_all_properties(self, tmp_path: Path):
        p = Paths(package_root=tmp_path)
        assert p.source == tmp_path / "source"
        assert p.cache == tmp_path / "cache"
        assert p.registry == tmp_path / "registry"
        assert p.dashboard == tmp_path / "dashboard"
        assert p.to_review == tmp_path / "to_review"
        assert p.unlinked == tmp_path / "unlinked"
        assert p.randonneur_packages == tmp_path / "source" / "randonneur_packages"
        # REFACTOR_FINAL F6 removed ``.bw_projects/`` — no Brightway project state.
        assert not hasattr(p, "bw_project")

    def test_source_file_paths_anchor_to_source_dir(self, tmp_path: Path):
        p = Paths(package_root=tmp_path)
        assert p.agribalyse_csv.parent == p.source
        assert p.placeholder_xlsx.parent == p.source
        assert p.harmonised_flows_gz.parent == p.source
        assert p.ef_cf_parquet.parent == p.source

    def test_registry_parquet_paths_anchor_to_registry_dir(self, tmp_path: Path):
        p = Paths(package_root=tmp_path)
        for path in (
            p.registry_meta,
            p.registry_mappings_biosphere,
            p.registry_mappings_technosphere,
            p.registry_unmatchable,
            p.registry_unit_conversions,
            p.registry_unit_aliases,
            p.registry_context_normalisation,
            p.registry_deletions,
            p.registry_edge_label_corrections,
            p.registry_target_index_ef,
        ):
            assert path.parent == p.registry

    def test_dashboard_paths_anchor_to_dashboard_dir(self, tmp_path: Path):
        p = Paths(package_root=tmp_path)
        assert p.dashboard_run_report.parent == p.dashboard
        assert p.dashboard_override_log.parent == p.dashboard
        assert p.dashboard_suppressed_strategy_log.parent == p.dashboard
        assert p.dashboard_drop_tally.parent == p.dashboard
        assert p.dashboard_link_log.parent == p.dashboard
        assert p.dashboard_backtest_dir == p.dashboard / "backtest"

    def test_ensure_runtime_dirs_creates_only_runtime_dirs(self, tmp_path: Path):
        p = Paths(package_root=tmp_path)
        p.ensure_runtime_dirs()
        for d in (p.cache, p.registry, p.dashboard, p.to_review, p.unlinked):
            assert d.is_dir()
        # source is *not* created (authoritative inputs only).
        assert not p.source.exists()

    def test_ensure_runtime_dirs_is_idempotent(self, tmp_path: Path):
        p = Paths(package_root=tmp_path)
        p.ensure_runtime_dirs()
        p.ensure_runtime_dirs()  # second call must not raise

    def test_paths_is_immutable(self, tmp_path: Path):
        p = Paths(package_root=tmp_path)
        with pytest.raises((AttributeError, Exception)):
            p.package_root = Path("/other")  # type: ignore[misc]


class TestSettings:
    def test_default_settings_carry_known_defaults(self):
        s = Settings()
        assert s.agribalyse_version == "3.2"
        assert s.ef_version == "3.1"
        assert s.ecoinvent_version is None
        assert s.apply_llm_overrides is True
        assert s.apply_transitive_layer is False

    def test_resolved_ecoinvent_falls_back_to_default_pairing(self):
        assert Settings().resolved_ecoinvent_version == "3.9.1"
        assert Settings(agribalyse_version="4.0-beta").resolved_ecoinvent_version == "3.11"

    def test_default_ecoinvent_for_raises_for_unknown_agb_version(self):
        with pytest.raises(ValueError, match="No ADEME-native ecoinvent pairing"):
            Settings.default_ecoinvent_for("9.9-bogus")

    def test_explicit_ecoinvent_version_overrides_default(self):
        s = Settings(ecoinvent_version="3.10")
        assert s.resolved_ecoinvent_version == "3.10"

    def test_resolved_project_name_uses_agb_version(self):
        s = Settings()
        assert s.resolved_project_name == "agribalyse-3.2-sentier"

    def test_explicit_project_name_overrides_default(self):
        s = Settings(project_name="custom")
        assert s.resolved_project_name == "custom"

    def test_resolved_agribalyse_db_name(self):
        s = Settings()
        assert s.resolved_agribalyse_db_name == "agribalyse-3.2"

    def test_ecoinvent_db_name_combines_version_and_system_model(self):
        s = Settings()
        assert s.ecoinvent_db_name == "ecoinvent-3.9.1-cutoff"
        s2 = Settings(ecoinvent_system_model="apos", ecoinvent_version="3.10")
        assert s2.ecoinvent_db_name == "ecoinvent-3.10-apos"

    def test_with_ecoinvent_returns_new_instance(self):
        s = Settings()
        s2 = s.with_ecoinvent("3.10")
        assert s is not s2
        assert s.ecoinvent_version is None
        assert s2.ecoinvent_version == "3.10"

    def test_with_no_llm_returns_new_instance(self):
        s = Settings()
        s2 = s.with_no_llm()
        assert s.apply_llm_overrides is True
        assert s2.apply_llm_overrides is False

    def test_settings_is_frozen(self):
        s = Settings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.agribalyse_version = "9.9"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "agb,expected_ei",
        [("3.2", "3.9.1"), ("4.0-beta", "3.11")],
    )
    def test_default_pairing_table(self, agb, expected_ei):
        assert Settings.default_ecoinvent_for(agb) == expected_ei
