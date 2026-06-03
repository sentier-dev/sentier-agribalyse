"""Unit tests for ``WaterUseBio3Bridge``.

Pins the load contract — schema version, duplicate detection,
``cf_rows`` shape — that downstream consumers in
``MethodCfRegistryBuilder`` depend on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ef.water_use_bio3_bridge import WaterUseBio3Bridge


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


class TestWaterUseBio3BridgeLoad:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = WaterUseBio3Bridge.load(tmp_path / "nope.json")
        assert len(result) == 0
        assert not result
        assert result.cf_rows() == []
        assert result.method_key == ()

    def test_empty_mappings_returns_empty_rows(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bridge.json",
            {
                "version": 1,
                "method_key": ["ecoinvent-3.9.1", "EF v3.1", "water use", "x"],
                "mappings": [],
            },
        )
        result = WaterUseBio3Bridge.load(path)
        assert len(result) == 0
        assert result.cf_rows() == []
        assert result.method_key == ("ecoinvent-3.9.1", "EF v3.1", "water use", "x")

    def test_reads_curated_rows_into_cf_rows(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bridge.json",
            {
                "version": 1,
                "method_key": ["ecoinvent-3.9.1", "EF v3.1", "water use", "x"],
                "mappings": [
                    {"code": "river", "cf": 42.95, "name": "Water, river"},
                    {"code": "release", "cf": -42.95, "name": "Water"},
                ],
            },
        )
        result = WaterUseBio3Bridge.load(path)
        assert len(result) == 2
        rows = result.cf_rows()
        # cf_rows yields rows sorted by code, all with the bio3 database.
        assert rows == [
            {"database": "ecoinvent-3.9.1-biosphere", "code": "release", "amount": -42.95},
            {"database": "ecoinvent-3.9.1-biosphere", "code": "river", "amount": 42.95},
        ]

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bridge.json",
            {
                "version": 999,
                "method_key": ["a", "b", "c", "d"],
                "mappings": [],
            },
        )
        with pytest.raises(ValueError, match="unsupported version"):
            WaterUseBio3Bridge.load(path)

    def test_duplicate_codes_raise(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bridge.json",
            {
                "version": 1,
                "method_key": ["a", "b", "c", "d"],
                "mappings": [
                    {"code": "dup", "cf": 1.0},
                    {"code": "dup", "cf": 2.0},
                ],
            },
        )
        with pytest.raises(ValueError, match="duplicate code"):
            WaterUseBio3Bridge.load(path)

    def test_missing_code_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bridge.json",
            {
                "version": 1,
                "method_key": ["a", "b", "c", "d"],
                "mappings": [{"cf": 1.0}],
            },
        )
        with pytest.raises(ValueError, match="missing code"):
            WaterUseBio3Bridge.load(path)

    def test_missing_cf_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bridge.json",
            {
                "version": 1,
                "method_key": ["a", "b", "c", "d"],
                "mappings": [{"code": "x"}],
            },
        )
        with pytest.raises(ValueError, match="missing cf"):
            WaterUseBio3Bridge.load(path)

    def test_invalid_method_key_arity_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "bridge.json",
            {
                "version": 1,
                "method_key": ["only-two", "parts"],
                "mappings": [],
            },
        )
        with pytest.raises(ValueError, match="method_key must be a 4-tuple"):
            WaterUseBio3Bridge.load(path)

    def test_real_source_file_loads(self) -> None:
        """The shipped source/water-use-bio3-bridge.json must parse and
        carry the water-use method key. Mappings are currently empty
        (see the file's description for the 2026-05-22 backtest
        rationale) but the loader must still accept the file."""
        bridge = WaterUseBio3Bridge.load(
            Path(__file__).resolve().parents[2] / "source" / "water-use-bio3-bridge.json"
        )
        assert bridge.method_key == (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "water use",
            "user deprivation potential (deprivation-weighted water consumption)",
        )
        # If non-empty (future targeted use), every CF must be ±42.95
        # (the SimaPro adapted m³ value) so balanced flows cancel.
        for _, cf in bridge.items():
            if cf != 0.0:
                assert abs(abs(cf) - 42.95) < 1e-9, (
                    f"unexpected CF magnitude {cf} (expected ±42.95 or 0)"
                )

    def test_empty_factory_is_noop(self) -> None:
        result = WaterUseBio3Bridge.empty()
        assert not result
        assert result.cf_rows() == []
        assert list(result.items()) == []
