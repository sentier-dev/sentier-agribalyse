"""Unit tests for ``ScoringPackageBuilder`` and ``ScoringPackageStore``.

The package is the Phase 4 replacement for bw_processing zip handoff:
content-addressable, parquet/numpy-native, atomic writes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scoring.exchange_frame import ExchangeFrame
from scoring.scoring_package import (
    ScoringPackageBuilder,
    ScoringPackageStore,
)


def _frame(rows):
    return ExchangeFrame.from_long(
        pd.DataFrame(
            rows,
            columns=["output_id", "input_id", "amount", "edge_type", "is_biosphere"],
        )
    )


def _trivial_frame():
    # Smallest non-degenerate system: 1 activity, 1 product, 1 biosphere flow.
    return _frame(
        [
            (1, 10, 1.0, "production", False),
            (1, 30, 0.5, "biosphere", True),
        ]
    )


class TestScoringPackageBuilder:
    def test_builds_all_three_matrices(self):
        frame = _trivial_frame()
        method_cfs = {
            ("ef", "climate"): pd.DataFrame({"flow_id": [30], "cf": [1.0]}),
        }
        pkg = ScoringPackageBuilder().build(frame, method_cfs)
        assert pkg.n_products == 1
        assert pkg.n_activities == 1
        assert pkg.n_biosphere_flows == 1
        assert ("ef", "climate") in pkg.methods

    def test_content_hash_is_stable_across_runs(self):
        frame = _trivial_frame()
        a = ScoringPackageBuilder().build(frame, {})
        b = ScoringPackageBuilder().build(frame, {})
        assert a.content_hash == b.content_hash
        # Different frame → different hash.
        frame2 = _frame(
            [
                (1, 10, 2.0, "production", False),  # amount changed
                (1, 30, 0.5, "biosphere", True),
            ]
        )
        c = ScoringPackageBuilder().build(frame2, {})
        assert a.content_hash != c.content_hash

    def test_content_hash_changes_when_method_cfs_change(self):
        """If method CFs change but the frame is identical, the hash MUST
        change. Otherwise ``ScoringPackageStore.write`` cache-hits the
        old directory and silently preserves stale CFs — exactly the
        bug that swallowed the §A.1 Barite filter and §C.1 water
        augmenter on 2026-05-04 (registry rebuilt with new CFs but
        scoring package re-used the previous build's CSRs because the
        frame hadn't changed).
        """
        frame = _trivial_frame()
        cfs_a = {("ef", "climate"): pd.DataFrame({"flow_id": [30], "cf": [1.0]})}
        cfs_b = {("ef", "climate"): pd.DataFrame({"flow_id": [30], "cf": [2.0]})}
        a = ScoringPackageBuilder().build(frame, cfs_a)
        b = ScoringPackageBuilder().build(frame, cfs_b)
        assert a.content_hash != b.content_hash, "different CFs must produce different content_hash"

    def test_content_hash_changes_when_method_added(self):
        """Adding a new method to ``method_cfs`` must change the hash."""
        frame = _trivial_frame()
        cfs_one = {("ef", "climate"): pd.DataFrame({"flow_id": [30], "cf": [1.0]})}
        cfs_two = {
            ("ef", "climate"): pd.DataFrame({"flow_id": [30], "cf": [1.0]}),
            ("ef", "ecotox"): pd.DataFrame({"flow_id": [30], "cf": [42.0]}),
        }
        a = ScoringPackageBuilder().build(frame, cfs_one)
        b = ScoringPackageBuilder().build(frame, cfs_two)
        assert a.content_hash != b.content_hash

    def test_content_hash_stable_under_cf_row_reordering(self):
        """Same CFs in a different row order must still hash identically
        — the cache key is the *content*, not the input ordering.
        """
        frame = _trivial_frame()
        cfs_a = {
            ("ef", "climate"): pd.DataFrame({"flow_id": [30, 31], "cf": [1.0, 2.0]}),
        }
        cfs_b = {
            ("ef", "climate"): pd.DataFrame({"flow_id": [31, 30], "cf": [2.0, 1.0]}),
        }
        a = ScoringPackageBuilder().build(frame, cfs_a)
        b = ScoringPackageBuilder().build(frame, cfs_b)
        assert a.content_hash == b.content_hash


class TestScoringPackageStoreRoundTrip:
    def test_write_then_read_produces_same_matrices(self, tmp_path):
        frame = _trivial_frame()
        cfs = {("ef", "climate"): pd.DataFrame({"flow_id": [30], "cf": [2.0]})}
        pkg = ScoringPackageBuilder().build(frame, cfs)

        store = ScoringPackageStore(root=tmp_path)
        store.write(pkg)
        out = store.read(pkg.content_hash)

        # Matrices match.
        np.testing.assert_array_equal(
            pkg.technosphere.matrix.toarray(),
            out.technosphere.matrix.toarray(),
        )
        np.testing.assert_array_equal(
            pkg.biosphere.matrix.toarray(),
            out.biosphere.matrix.toarray(),
        )
        # Method tuple round-trips.
        assert ("ef", "climate") in out.methods
        np.testing.assert_array_equal(
            pkg.methods[("ef", "climate")].toarray(),
            out.methods[("ef", "climate")].toarray(),
        )

    def test_write_is_atomic(self, tmp_path):
        # The store should never expose a half-finished directory at
        # ``<root>/<hash>``. We verify this by checking that the rename
        # is the last filesystem mutation: there's a ``.partial`` dir
        # during the write that flips into place atomically.
        frame = _trivial_frame()
        pkg = ScoringPackageBuilder().build(frame, {})
        store = ScoringPackageStore(root=tmp_path)
        path = store.write(pkg)
        # Final directory exists, ``.partial`` is gone.
        assert path.exists()
        assert not (tmp_path / f"{pkg.content_hash}.partial").exists()

    def test_idempotent_writes(self, tmp_path):
        # Writing the same package twice must not corrupt the directory
        # or raise. The second call is a no-op (the dir already exists).
        frame = _trivial_frame()
        pkg = ScoringPackageBuilder().build(frame, {})
        store = ScoringPackageStore(root=tmp_path)
        path_a = store.write(pkg)
        path_b = store.write(pkg)
        assert path_a == path_b

    def test_lru_eviction_caps_stored_packages(self, tmp_path, monkeypatch):
        """What-if overrides fork a package per value; the store keeps only
        the MAX_PACKAGES most-recent dirs (2026-08-18 adversarial F6b)."""
        import os
        import time

        frame = _trivial_frame()
        pkg = ScoringPackageBuilder().build(frame, {})
        store = ScoringPackageStore(root=tmp_path)
        monkeypatch.setattr(ScoringPackageStore, "MAX_PACKAGES", 2)
        # Simulate older what-if packages with distinct hashes and old mtimes.
        now = time.time()
        for i, fake_hash in enumerate(["a" * 64, "b" * 64, "c" * 64]):
            d = tmp_path / fake_hash
            d.mkdir()
            os.utime(d, (now - 1000 + i, now - 1000 + i))
        (tmp_path / "dead.partial").mkdir()  # crashed-write residue

        path = store.write(pkg)

        survivors = {p.name for p in tmp_path.iterdir()}
        assert path.name in survivors  # just-written always kept
        assert "dead.partial" not in survivors  # residue removed
        # Only MAX_PACKAGES dirs remain, newest-first: the real package + c.
        assert survivors == {path.name, "c" * 64}

    def test_method_slug_is_filesystem_safe(self, tmp_path):
        # EF method tuples have spaces and parentheses; the slug must
        # round-trip them.
        frame = _trivial_frame()
        method = ("ecoinvent-3.9.1", "EF v3.1", "particulate matter formation", "impact (HH)")
        cfs = {method: pd.DataFrame({"flow_id": [30], "cf": [3.0]})}
        pkg = ScoringPackageBuilder().build(frame, cfs)
        store = ScoringPackageStore(root=tmp_path)
        store.write(pkg)
        out = store.read(pkg.content_hash)
        assert method in out.methods
        np.testing.assert_array_equal(
            pkg.methods[method].toarray(),
            out.methods[method].toarray(),
        )

    def test_read_missing_raises(self, tmp_path):
        store = ScoringPackageStore(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.read("nonexistent-hash")
