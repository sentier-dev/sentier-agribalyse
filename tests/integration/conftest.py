"""Fixtures shared by the integration tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def real_bw_simapro_csv():
    """Swap the session-wide ``bw_simapro_csv`` stub for the real package.

    ``tests/fixtures/bw_stubs.py`` installs a permissive stub before any
    test imports; the integration tests that parse a SimaPro file need the
    genuine parser. Restored on teardown so the unit suite stays hermetic.
    """
    import importlib
    import sys

    # The real parse pulls a chain of stubbed packages
    # (bw_simapro_csv → multifunctional → bw2data ...), so evict every
    # ``_dds_test_stub``-marked module, not just the parser.
    saved = {
        k: v
        for k, v in sys.modules.items()
        if getattr(v, "_dds_test_stub", False)
        or getattr(sys.modules.get(k.split(".")[0]), "_dds_test_stub", False)
    }
    for key in list(saved):
        del sys.modules[key]
    real = importlib.import_module("bw_simapro_csv")
    yield real
    for key in list(sys.modules):
        root = key.split(".")[0]
        if root in {s.split(".")[0] for s in saved}:
            del sys.modules[key]
    sys.modules.update(saved)
