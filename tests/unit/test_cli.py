"""Unit tests for the CLI classes — argparse contract + dispatch via execute."""

from __future__ import annotations

import sys

import pytest

from cli._base import BaseCli
from cli.backtest import BacktestCli
from cli.build_biosphere_catalog import BuildBiosphereCatalogCli
from cli.build_ef_flows_registry import BuildEfFlowsRegistryCli
from cli.build_method_cfs_registry import BuildMethodCfsRegistryCli
from cli.build_packages import BuildPackagesCli
from cli.build_registry import BuildRegistryCli
from cli.link_all import LinkAllCli
from cli.mappings_comparison import MappingsComparisonCli
from cli.reset import ResetCli
from cli.run_end_to_end import RunEndToEndCli

# ============================================================================
# BaseCli scaffolding


class _OkCli(BaseCli):
    PROG = "test.ok"
    DESCRIPTION = "ok"

    def __init__(self, settings=None, *, on_execute=None):
        super().__init__(settings=settings) if settings else super().__init__()
        self.on_execute = on_execute or (lambda args: None)
        self.executed_args = None

    def execute(self, args):
        self.executed_args = args
        self.on_execute(args)


class _BoomCli(BaseCli):
    PROG = "test.boom"
    DESCRIPTION = "boom"

    def execute(self, args):
        raise RuntimeError("die")


class TestBaseCli:
    def test_run_returns_zero_on_success(self):
        cli = _OkCli()
        assert cli.run([]) == 0
        assert cli.executed_args is not None

    def test_run_returns_one_on_exception_without_propagating(self):
        # The error path swallows the exception, logs it, and returns 1.
        assert _BoomCli().run([]) == 1

    def test_run_uses_sys_argv_when_argv_is_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])
        # No exception means argparse parsed the empty positional list cleanly.
        assert _OkCli().run() == 0


# ============================================================================
# BuildRegistryCli


class TestBuildRegistryCli:
    def test_build_registry_calls_pipeline_run(self, settings, monkeypatch):
        called = {"n": 0}

        class _FakePipeline:
            def __init__(self, _settings):
                pass

            def run(self):
                called["n"] += 1
                return {"row_counts": {}}

        monkeypatch.setattr("cli.build_registry.RegistryBuildPipeline", _FakePipeline)
        rc = BuildRegistryCli(settings=settings).run([])
        assert rc == 0
        assert called["n"] == 1


# ============================================================================
# BuildBiosphereCatalogCli


class TestBuildBiosphereCatalogCli:
    def test_runs_builder_directly_no_brightway_project(self, settings, monkeypatch):
        order: list[str] = []

        class _FakeBuilder:
            def __init__(self, *, settings):
                pass

            def build(self):
                order.append("build")
                return settings.paths.registry_biosphere_catalog

        monkeypatch.setattr("cli.build_biosphere_catalog.BiosphereRegistryBuilder", _FakeBuilder)
        rc = BuildBiosphereCatalogCli(settings=settings).run([])
        assert rc == 0
        assert order == ["build"]


# ============================================================================
# BuildEfFlowsRegistryCli + BuildMethodCfsRegistryCli


class TestBuildEfFlowsRegistryCli:
    def test_runs_registry_load_then_builder(self, settings, monkeypatch):
        order: list[str] = []

        class _FakeRegistry:
            @classmethod
            def load(cls, _settings):
                order.append("registry")
                return cls()

        class _FakeBuilder:
            def __init__(self, *, settings, registry):
                pass

            def build(self):
                order.append("build")
                return settings.paths.registry_ef_flows

        monkeypatch.setattr("cli.build_ef_flows_registry.MappingRegistry", _FakeRegistry)
        monkeypatch.setattr("cli.build_ef_flows_registry.EfFlowsRegistryBuilder", _FakeBuilder)
        rc = BuildEfFlowsRegistryCli(settings=settings).run([])
        assert rc == 0
        assert order == ["registry", "build"]


class TestBuildMethodCfsRegistryCli:
    def test_runs_cf_table_then_builder(self, settings, monkeypatch):
        order: list[str] = []

        class _FakeCfTable:
            def __init__(self, *, path):
                self.path = path
                order.append("cf_table")

        class _FakeBuilder:
            def __init__(self, *, settings, cf_table, **_kwargs):
                # Accept the optional kwargs the CLI now passes when
                # their source files are present (tmp_path tests don't
                # ship them, so they stay None — the assertion below
                # still proves CF-table-then-builder ordering).
                pass

            def build(self):
                order.append("build")

        monkeypatch.setattr("cli.build_method_cfs_registry.EfCfTable", _FakeCfTable)
        monkeypatch.setattr("cli.build_method_cfs_registry.MethodCfRegistryBuilder", _FakeBuilder)
        rc = BuildMethodCfsRegistryCli(settings=settings).run([])
        assert rc == 0
        assert order == ["cf_table", "build"]


# ============================================================================
# LinkAllCli — argparse + options carry through


class TestLinkAllCli:
    def test_default_flags_yield_default_options(self, settings, monkeypatch):
        captured = {}

        class _FakePipeline:
            def __init__(self, _settings, *, options):
                captured["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("cli.link_all.LinkAllPipeline", _FakePipeline)
        LinkAllCli(settings=settings).run([])
        opts = captured["options"]
        assert opts.skip_ecoinvent is False

    def test_no_llm_flag_propagates_to_settings(self, settings, monkeypatch):
        captured = {}

        class _FakePipeline:
            def __init__(self, _settings, *, options):
                captured["settings_apply_llm"] = _settings.apply_llm_overrides
                captured["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("cli.link_all.LinkAllPipeline", _FakePipeline)
        LinkAllCli(settings=settings).run(["--no-llm"])
        assert captured["settings_apply_llm"] is False

    def test_skip_ecoinvent_flag_propagates(self, settings, monkeypatch):
        captured = {}

        class _FakePipeline:
            def __init__(self, _settings, *, options):
                captured["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("cli.link_all.LinkAllPipeline", _FakePipeline)
        LinkAllCli(settings=settings).run(["--skip-ecoinvent"])
        opts = captured["options"]
        assert opts.skip_ecoinvent is True


# ============================================================================
# RunEndToEndCli — argparse coverage


class TestRunEndToEndCli:
    def test_default_solver_is_scipy(self, settings, monkeypatch):
        captured = {}

        class _FakePipeline:
            def __init__(self, _settings, *, options):
                captured["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("cli.run_end_to_end.EndToEndPipeline", _FakePipeline)
        RunEndToEndCli(settings=settings).run([])
        assert captured["options"].solver == "scipy"
        assert captured["options"].n_sample_products == 5

    def test_pardiso_and_n_products_override(self, settings, monkeypatch):
        captured = {}

        class _FakePipeline:
            def __init__(self, _settings, *, options):
                captured["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("cli.run_end_to_end.EndToEndPipeline", _FakePipeline)
        RunEndToEndCli(settings=settings).run(["--solver", "pardiso", "--n-products", "20"])
        assert captured["options"].solver == "pardiso"
        assert captured["options"].n_sample_products == 20

    def test_invalid_solver_choice_returns_nonzero(self, settings, monkeypatch):
        # argparse exits before our execute hook; the test wraps SystemExit.
        with pytest.raises(SystemExit):
            RunEndToEndCli(settings=settings).run(["--solver", "magic"])


# ============================================================================
# BacktestCli


class TestBacktestCli:
    def test_default_solver_is_scipy(self, settings, monkeypatch):
        captured = {}

        class _FakePipeline:
            def __init__(self, _settings, *, options):
                captured["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("cli.backtest.BacktestPipeline", _FakePipeline)
        BacktestCli(settings=settings).run([])
        assert captured["options"].solver == "scipy"
        assert captured["options"].n_workers == 1

    def test_workers_flag_threads_through_to_options(self, settings, monkeypatch):
        captured = {}

        class _FakePipeline:
            def __init__(self, _settings, *, options):
                captured["options"] = options

            def run(self):
                return None

        monkeypatch.setattr("cli.backtest.BacktestPipeline", _FakePipeline)
        BacktestCli(settings=settings).run(["--workers", "4"])
        assert captured["options"].n_workers == 4


# ============================================================================
# BuildPackagesCli


class TestBuildPackagesCli:
    def test_build_packages_invokes_exporter(self, settings, monkeypatch):
        called = {"n": 0}

        class _FakeExporter:
            def __init__(self, _settings):
                pass

            def export(self):
                called["n"] += 1
                return {"a": "b"}

        monkeypatch.setattr("cli.build_packages.RandonneurPackagesExporter", _FakeExporter)
        rc = BuildPackagesCli(settings=settings).run([])
        assert rc == 0
        assert called["n"] == 1


# ============================================================================
# MappingsComparisonCli


class TestMappingsComparisonCli:
    def test_loads_sp_data_from_pickle_then_exports(self, settings, monkeypatch):
        """The CLI now reads the cached importer pickle and hands ``sp.data``
        to the exporter — no Brightway project setup involved."""
        import pickle

        order: list[str] = []

        # Materialise a tiny importer pickle on disk. ``types.SimpleNamespace``
        # avoids the local-class pickle issue.
        import types

        cache_path = settings.paths.importer_cache_pkl
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(
            pickle.dumps(types.SimpleNamespace(data=[{"name": "P", "exchanges": []}]))
        )

        class _FakeExporter:
            def __init__(self, _settings, sp_data=None):
                order.append(f"init:{len(sp_data)}")

            def export(self):
                order.append("export")
                return settings.paths.to_review / "mappings_comparison.xlsx"

        monkeypatch.setattr("cli.mappings_comparison.MappingsComparisonExporter", _FakeExporter)
        rc = MappingsComparisonCli(settings=settings).run([])
        assert rc == 0
        assert order == ["init:1", "export"]

    def test_raises_when_importer_cache_missing(self, settings):
        rc = MappingsComparisonCli(settings=settings).run([])
        # The CLI returns non-zero when the cache file is absent (caller
        # logs the error; pytest sees rc=1).
        assert rc == 1


# ============================================================================
# ResetCli


class TestResetCli:
    def test_reset_deletes_scoring_packages_when_present(self, settings):
        """When scoring_packages_root exists, reset removes it."""
        target = settings.paths.scoring_packages_root
        target.mkdir(parents=True, exist_ok=True)
        (target / "some_pkg").mkdir()

        rc = ResetCli(settings=settings).run([])
        assert rc == 0
        assert not target.exists()

    def test_reset_is_noop_when_dir_absent(self, settings):
        """When scoring_packages_root does not exist, reset exits cleanly."""
        target = settings.paths.scoring_packages_root
        # Ensure the directory doesn't exist (project_root fixture only creates cache/).
        assert not target.exists()

        rc = ResetCli(settings=settings).run([])
        assert rc == 0
        assert not target.exists()

    def test_reset_propagates_os_error(self, settings, monkeypatch):
        """An OS error during rmtree causes the CLI to exit with code 1."""
        target = settings.paths.scoring_packages_root
        target.mkdir(parents=True, exist_ok=True)

        import shutil

        monkeypatch.setattr(
            shutil, "rmtree", lambda p: (_ for _ in ()).throw(PermissionError("denied"))
        )
        rc = ResetCli(settings=settings).run([])
        assert rc == 1
