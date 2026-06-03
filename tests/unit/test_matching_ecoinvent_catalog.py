"""Unit tests for ``matching.EcoinventCatalog`` and ``EcoinventCatalogBuilder``.

After REFACTOR_FINAL F6 the builder reads the activities from a JSON
snapshot in ``source/`` instead of bw2data. These tests cover both sides
of the round-trip plus the on-disk schema contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from matching.ecoinvent_catalog import (
    EcoinventActivityRef,
    EcoinventCatalog,
    EcoinventCatalogBuilder,
)


def _write_snapshot(settings, activities: list[dict]) -> Path:
    path = settings.paths.source / "ecoinvent-3.9.1-cutoff-activities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"db_name": settings.ecoinvent_db_name, "activities": activities}
    path.write_text(json.dumps(payload))
    return path


def _ref(**kw) -> EcoinventActivityRef:
    base = dict(
        db="ecoinvent-3.9.1-cutoff",
        code="abc",
        name="market for tap water",
        unit="kilogram",
        location="RoW",
        reference_product="tap water",
    )
    base.update(kw)
    return EcoinventActivityRef(**base)


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    cols = ["database", "code", "name", "unit", "location", "reference_product"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(path, index=False)
    return path


# ============================================================================
# Builder


class TestEcoinventCatalogBuilder:
    @pytest.fixture
    def snapshot(self, settings):
        return _write_snapshot(
            settings,
            [
                {
                    "code": "act-1",
                    "name": "market for tap water",
                    "unit": "kilogram",
                    "location": "RoW",
                    "reference_product": "tap water",
                },
                {
                    "code": "act-2",
                    "name": "market for electricity, low voltage",
                    "unit": "kilowatt hour",
                    "location": "FR",
                    "reference_product": "electricity, low voltage",
                },
            ],
        )

    def test_build_writes_parquet_with_expected_schema(self, settings, snapshot):
        path = EcoinventCatalogBuilder(settings=settings).build()
        assert path == settings.paths.registry_ecoinvent_catalog
        assert path.exists()

        df = pd.read_parquet(path)
        expected = {"database", "code", "name", "unit", "location", "reference_product"}
        assert expected == set(df.columns)
        assert len(df) == 2

    def test_build_sorts_rows_for_determinism(self, settings, snapshot):
        path = EcoinventCatalogBuilder(settings=settings).build()
        df = pd.read_parquet(path)
        keys = list(zip(df["database"].tolist(), df["code"].tolist(), strict=True))
        assert keys == sorted(keys)

    def test_build_is_byte_equal_across_runs(self, settings, snapshot):
        path1 = EcoinventCatalogBuilder(settings=settings).build()
        bytes1 = path1.read_bytes()
        path2 = EcoinventCatalogBuilder(settings=settings).build()
        bytes2 = path2.read_bytes()
        assert bytes1 == bytes2

    def test_build_raises_when_snapshot_missing(self, settings):
        # No snapshot fixture — file is not on disk.
        with pytest.raises(FileNotFoundError, match="ecoinvent activity snapshot missing"):
            EcoinventCatalogBuilder(settings=settings).build()

    def test_build_atomic_write_leaves_no_partial_file(self, settings, snapshot):
        EcoinventCatalogBuilder(settings=settings).build()
        partial = settings.paths.registry_ecoinvent_catalog.with_suffix(
            settings.paths.registry_ecoinvent_catalog.suffix + ".partial"
        )
        assert not partial.exists()

    def test_row_for_handles_missing_optional_fields(self):
        row = EcoinventCatalogBuilder._row_for(
            "ecoinvent-3.9.1-cutoff",
            {"code": "x", "name": "n"},
        )
        assert row["unit"] == ""
        assert row["location"] == ""
        assert row["reference_product"] == ""

    def test_row_for_accepts_either_refprod_key(self):
        row = EcoinventCatalogBuilder._row_for(
            "ecoinvent-3.9.1-cutoff",
            {"code": "x", "reference product": "old-key"},
        )
        assert row["reference_product"] == "old-key"


# ============================================================================
# Catalog — load + lookups


class TestEcoinventCatalogLoad:
    def test_load_reads_parquet_into_activities(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "ecoinvent.parquet",
            [
                {
                    "database": "ecoinvent-3.9.1-cutoff",
                    "code": "a",
                    "name": "market for tap water",
                    "unit": "kilogram",
                    "location": "RoW",
                    "reference_product": "tap water",
                }
            ],
        )
        cat = EcoinventCatalog.load(path)
        assert cat.db_names == ("ecoinvent-3.9.1-cutoff",)
        assert len(cat.activities) == 1
        a = cat.activities[0]
        assert a.code == "a"
        assert a.location == "RoW"
        assert a.reference_product == "tap water"

    def test_load_filters_by_db_names(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "ecoinvent.parquet",
            [
                {
                    "database": "ecoinvent-3.9.1-cutoff",
                    "code": "a",
                    "name": "x",
                    "unit": "kg",
                    "location": "RoW",
                    "reference_product": "x",
                },
                {
                    "database": "ecoinvent-3.10-cutoff",
                    "code": "b",
                    "name": "y",
                    "unit": "kg",
                    "location": "RoW",
                    "reference_product": "y",
                },
            ],
        )
        cat = EcoinventCatalog.load(path, db_names=("ecoinvent-3.9.1-cutoff",))
        assert cat.db_names == ("ecoinvent-3.9.1-cutoff",)
        assert {a.code for a in cat.activities} == {"a"}

    def test_load_raises_when_path_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="ecoinvent catalog parquet"):
            EcoinventCatalog.load(tmp_path / "ghost.parquet")

    def test_load_handles_null_reference_product(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "ecoinvent.parquet",
            [
                {
                    "database": "ecoinvent-3.9.1-cutoff",
                    "code": "a",
                    "name": "n",
                    "unit": "kg",
                    "location": "RoW",
                    "reference_product": None,
                }
            ],
        )
        cat = EcoinventCatalog.load(path)
        assert cat.activities[0].reference_product == ""


class TestEcoinventCatalogLookup:
    def _cat(self, refs: list[EcoinventActivityRef]) -> EcoinventCatalog:
        db_names = tuple(sorted({r.db for r in refs}))
        return EcoinventCatalog(db_names=db_names, activities=tuple(refs))

    def test_get_returns_activity_by_db_code(self):
        cat = self._cat([_ref(code="a"), _ref(code="b")])
        assert cat.get("ecoinvent-3.9.1-cutoff", "a").code == "a"
        assert cat.get("ecoinvent-3.9.1-cutoff", "missing") is None
        assert cat.get("other-db", "a") is None

    def test_match_full_returns_unique_hit(self):
        cat = self._cat([_ref(code="a")])
        hit = cat.match_full(
            "ecoinvent-3.9.1-cutoff",
            "market for tap water",
            "kilogram",
            "RoW",
            "tap water",
        )
        assert hit is not None
        assert hit.code == "a"

    def test_match_full_is_case_insensitive_on_name_and_refprod(self):
        cat = self._cat([_ref(code="a")])
        hit = cat.match_full(
            "ecoinvent-3.9.1-cutoff",
            "MARKET FOR TAP WATER",
            "kilogram",
            "RoW",
            "TAP WATER",
        )
        assert hit is not None and hit.code == "a"

    def test_match_full_returns_none_when_ambiguous(self):
        cat = self._cat([_ref(code="a"), _ref(code="b")])  # same key, different codes
        assert (
            cat.match_full(
                "ecoinvent-3.9.1-cutoff",
                "market for tap water",
                "kilogram",
                "RoW",
                "tap water",
            )
            is None
        )

    def test_match_full_returns_none_when_missing(self):
        cat = self._cat([_ref(code="a")])
        assert (
            cat.match_full(
                "ecoinvent-3.9.1-cutoff",
                "ghost",
                "kilogram",
                "RoW",
                "tap water",
            )
            is None
        )

    def test_match_full_unit_mismatch_misses(self):
        cat = self._cat([_ref(code="a", unit="kilogram")])
        assert (
            cat.match_full(
                "ecoinvent-3.9.1-cutoff",
                "market for tap water",
                "litre",  # wrong unit
                "RoW",
                "tap water",
            )
            is None
        )

    def test_match_relaxed_drops_reference_product(self):
        cat = self._cat([_ref(code="a", reference_product="tap water")])
        hit = cat.match_relaxed(
            "ecoinvent-3.9.1-cutoff",
            "market for tap water",
            "kilogram",
            "RoW",
        )
        assert hit is not None and hit.code == "a"

    def test_match_relaxed_returns_none_when_ambiguous(self):
        # Two activities with same name+unit+location but different refprod.
        cat = self._cat(
            [
                _ref(code="a", reference_product="tap water"),
                _ref(code="b", reference_product="recycled water"),
            ]
        )
        # match_full with a specific refprod resolves uniquely.
        hit = cat.match_full(
            "ecoinvent-3.9.1-cutoff",
            "market for tap water",
            "kilogram",
            "RoW",
            "tap water",
        )
        assert hit is not None and hit.code == "a"
        # match_relaxed sees both — must not pick.
        assert (
            cat.match_relaxed(
                "ecoinvent-3.9.1-cutoff",
                "market for tap water",
                "kilogram",
                "RoW",
            )
            is None
        )

    def test_match_full_handles_empty_or_none_inputs(self):
        cat = self._cat([_ref(code="a", reference_product="")])
        # Empty refprod still matches (both stored and queried are empty).
        hit = cat.match_full(
            "ecoinvent-3.9.1-cutoff",
            "market for tap water",
            "kilogram",
            "RoW",
            "",
        )
        assert hit is not None and hit.code == "a"
        # ``None`` is treated as empty string.
        hit = cat.match_full(
            "ecoinvent-3.9.1-cutoff",
            "market for tap water",
            "kilogram",
            "RoW",
            None,  # type: ignore[arg-type]
        )
        assert hit is not None and hit.code == "a"


class TestEcoinventCatalogRoundTrip:
    @pytest.fixture
    def snapshot(self, settings):
        return _write_snapshot(
            settings,
            [
                {
                    "code": "act-1",
                    "name": "market for tap water",
                    "unit": "kilogram",
                    "location": "RoW",
                    "reference_product": "tap water",
                }
            ],
        )

    def test_build_then_load_yields_same_keys(self, settings, snapshot):
        path = EcoinventCatalogBuilder(settings=settings).build()
        cat = EcoinventCatalog.load(path)
        assert {(a.db, a.code) for a in cat.activities} == {(settings.ecoinvent_db_name, "act-1")}
        a = cat.activities[0]
        assert a.reference_product == "tap water"
        assert a.location == "RoW"
