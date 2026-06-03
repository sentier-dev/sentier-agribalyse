"""Shared pytest fixtures.

Two responsibilities:

1. Provide a Brightway-free import path. Every ``bw2*`` import in production
   code is deferred (lives inside a method body), so unit tests can install
   thin fakes into ``sys.modules`` once per session. The fakes are intentionally
   permissive — individual tests replace them with stricter ``Mock`` objects
   when they assert specific behavior.

2. Provide reusable fixture factories for tiny in-memory artifacts (Settings,
   registry DataFrames, fake importer objects, fake Brightway databases).
   Factories live in ``tests/fixtures/builders.py`` and are re-exported here
   so tests just write ``def test_foo(tiny_registry, fake_importer): ...``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# Brightway / randonneur stubs
#
# The production code defers every bw2* / randonneur* import to runtime, so
# we install permissive stub modules ahead of any test importing the package.
# Tests that need stricter doubles override these via monkeypatch.
from tests.fixtures import bw_stubs

bw_stubs.install_stubs()


# --------------------------------------------------------------------------
# Re-export fixture factories.

from tests.fixtures.builders import (
    FakeBwActivity,
    FakeBwDatabase,
    FakeBwExchange,
    FakeMethod,
    FakeSimaProImporter,
    make_audit_log,
    make_bio_catalog,
    make_drop_tracker,
    make_mappings_biosphere_df,
    make_mappings_technosphere_df,
    make_paths,
    make_registry,
    make_settings,
    make_unit_aliases_df,
    make_unit_conversions_df,
    make_unmatchable_df,
)

# --------------------------------------------------------------------------
# Filesystem fixtures.


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A pristine project root with the runtime directories created."""
    for d in ("source", "cache", "registry", "dashboard", "to_review", "unlinked"):
        (tmp_path / d).mkdir()
    (tmp_path / "source" / "randonneur_packages").mkdir()
    return tmp_path


@pytest.fixture
def settings(project_root: Path):
    """A Settings rooted at ``project_root`` — never touches the real repo."""
    return make_settings(project_root)


@pytest.fixture
def paths(project_root: Path):
    return make_paths(project_root)


# --------------------------------------------------------------------------
# Logging silencer.


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch):
    """Keep the test output clean: every Logging.get(name).info / .warning
    becomes a no-op. Tests that want to assert on log output replace this
    fixture's monkeypatch with their own."""
    import logging as stdlib_logging

    from core.logging import Logging

    class _NoopLogger:
        def bind(self, **_kw):
            return self

        def __getattr__(self, _name):
            return lambda *a, **k: None

    _singleton = _NoopLogger()

    monkeypatch.setattr(Logging, "get", classmethod(lambda cls, name=None: _singleton))
    monkeypatch.setattr(Logging, "configure", classmethod(lambda cls, **k: None))
    # Also throttle stdlib root logger so any leaked logging.* call is silent.
    stdlib_logging.disable(stdlib_logging.CRITICAL)
    yield
    stdlib_logging.disable(stdlib_logging.NOTSET)


# --------------------------------------------------------------------------
# Re-exported names live at module scope so tests can ``from tests.conftest
# import make_settings``; pytest auto-discovers fixture functions only.

__all__ = [
    "FakeBwActivity",
    "FakeBwDatabase",
    "FakeBwExchange",
    "FakeMethod",
    "FakeSimaProImporter",
    "make_audit_log",
    "make_bio_catalog",
    "make_drop_tracker",
    "make_mappings_biosphere_df",
    "make_mappings_technosphere_df",
    "make_paths",
    "make_registry",
    "make_settings",
    "make_unit_aliases_df",
    "make_unit_conversions_df",
    "make_unmatchable_df",
]
