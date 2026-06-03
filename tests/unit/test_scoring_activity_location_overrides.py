"""Unit tests for ``ActivityLocationOverrides``.

The override table is the curated escape hatch for AGB stand-in
activities whose aggregate location masks ADEME's regional context
(sweet pepper greenhouse GLO → FR, etc.). These tests pin the load
contract — schema version, duplicate detection, missing-file
no-op — that downstream consumers in the link pipeline rely on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scoring.activity_location_overrides import ActivityLocationOverrides


class TestActivityLocationOverridesLoad:
    def test_missing_file_returns_empty_overrides(self, tmp_path: Path) -> None:
        result = ActivityLocationOverrides.load(tmp_path / "nope.json")
        assert len(result) == 0
        assert not result
        assert result.location_for("agribalyse-3.2", "AGRIBALU000000003101321") is None

    def test_empty_overrides_list_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"version": 1, "overrides": []}))
        result = ActivityLocationOverrides.load(path)
        assert len(result) == 0

    def test_reads_curated_rows_and_normalises_iso_uppercase(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "overrides": [
                        {
                            "database": "agribalyse-3.2",
                            "code": "AGRIBALU000000003101321",
                            "location": "fr",
                            "rationale": "sweet pepper greenhouse",
                        },
                        {
                            "database": "agribalyse-3.2",
                            "code": "AGRIBALU000000003101322",
                            "location": "ES",
                        },
                    ],
                }
            )
        )
        result = ActivityLocationOverrides.load(path)
        assert len(result) == 2
        assert result.location_for("agribalyse-3.2", "AGRIBALU000000003101321") == "FR"
        assert result.location_for("agribalyse-3.2", "AGRIBALU000000003101322") == "ES"

    def test_items_yields_all_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "overrides": [
                        {"database": "agb", "code": "x", "location": "FR"},
                        {"database": "agb", "code": "y", "location": "ES"},
                    ],
                }
            )
        )
        result = ActivityLocationOverrides.load(path)
        triples = sorted(result.items())
        assert triples == [("agb", "x", "FR"), ("agb", "y", "ES")]

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"version": 999, "overrides": []}))
        with pytest.raises(ValueError, match="unsupported version"):
            ActivityLocationOverrides.load(path)

    def test_duplicate_db_code_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "overrides": [
                        {"database": "agb", "code": "dup", "location": "FR"},
                        {"database": "agb", "code": "dup", "location": "ES"},
                    ],
                }
            )
        )
        with pytest.raises(ValueError, match="duplicate"):
            ActivityLocationOverrides.load(path)

    def test_missing_database_or_code_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps({"version": 1, "overrides": [{"database": "agb", "location": "FR"}]})
        )
        with pytest.raises(ValueError, match="missing database/code"):
            ActivityLocationOverrides.load(path)

    def test_missing_location_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "overrides": [{"database": "agb", "code": "x", "location": ""}],
                }
            )
        )
        with pytest.raises(ValueError, match="empty location"):
            ActivityLocationOverrides.load(path)

    def test_empty_factory_is_a_noop(self) -> None:
        result = ActivityLocationOverrides.empty()
        assert not result
        assert result.location_for("any", "thing") is None
        assert list(result.items()) == []
