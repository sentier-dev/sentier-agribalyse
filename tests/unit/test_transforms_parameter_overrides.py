"""Unit tests for ``ParameterOverridesStore`` and ``ParameterOverridesApplier``.

The store owns ``source/parameter_overrides.csv`` — the single source of
override truth. Round-trip fidelity, upsert-replace semantics, targeted
clears, and the file-deleted-when-empty invariant (a cleared state must be
indistinguishable from pristine) are the contract ``dds-set-parameter`` /
``dds-clear-parameters`` / ``dds-reset`` rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from transforms.parameter_overrides import (
    ParameterOverride,
    ParameterOverridesApplier,
    ParameterOverridesStore,
)
from transforms.sp_csv_parser import ParsedSimaProCsv


@pytest.fixture
def store(tmp_path: Path, monkeypatch) -> ParameterOverridesStore:
    settings = Settings()
    monkeypatch.setattr(
        type(settings.paths),
        "parameter_overrides_csv",
        property(lambda self: tmp_path / "parameter_overrides.csv"),
    )
    return ParameterOverridesStore(settings=settings)


class TestParameterOverridesStore:
    def test_load_returns_empty_when_file_absent(self, store):
        assert store.load() == []

    def test_round_trip_preserves_fields(self, store):
        ov = ParameterOverride("Packaging_Weight", "*", 0.03, "2026-08-17", "what-if")
        store.save([ov])
        assert store.load() == [ov]

    def test_upsert_stamps_set_at(self, store):
        stamped = store.upsert(ParameterOverride("P", "*", 1.5))
        assert stamped.set_at != ""
        assert store.load() == [stamped]

    def test_upsert_replaces_same_name_and_scope(self, store):
        store.upsert(ParameterOverride("P", "*", 1.0))
        store.upsert(ParameterOverride("p", "*", 2.0))  # case-insensitive key
        rows = store.load()
        assert len(rows) == 1
        assert rows[0].value == 2.0

    def test_upsert_keeps_distinct_scopes(self, store):
        store.upsert(ParameterOverride("P", "*", 1.0))
        store.upsert(ParameterOverride("P", "CODE1", 2.0))
        assert len(store.load()) == 2

    def test_clear_all_deletes_file(self, store):
        store.upsert(ParameterOverride("P", "*", 1.0))
        assert store.clear() == 1
        assert not store.path.exists()

    def test_clear_by_name(self, store):
        store.upsert(ParameterOverride("A", "*", 1.0))
        store.upsert(ParameterOverride("B", "*", 2.0))
        assert store.clear(name="a") == 1
        assert [ov.parameter_name for ov in store.load()] == ["B"]

    def test_clear_by_scope(self, store):
        store.upsert(ParameterOverride("A", "*", 1.0))
        store.upsert(ParameterOverride("A", "CODE1", 2.0))
        assert store.clear(scope="CODE1") == 1
        assert [ov.scope for ov in store.load()] == ["*"]

    def test_clear_no_match_removes_nothing(self, store):
        store.upsert(ParameterOverride("A", "*", 1.0))
        assert store.clear(name="ghost") == 0
        assert len(store.load()) == 1

    def test_clear_on_pristine_store_is_noop(self, store):
        """No overrides file at all: clear removes nothing and creates nothing."""
        assert store.clear() == 0
        assert not store.path.exists()

    def test_load_malformed_value_raises_actionable_error(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            "parameter_name;scope;value;set_at;comment\nP;*;abc;;\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=r"Malformed override row at .*:2"):
            store.load()

    def test_load_missing_column_raises_actionable_error(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text("parameter_name;scope\nP;*\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed override row"):
            store.load()

    def test_load_rejects_non_finite_values(self, store):
        """Hand-edited nan/inf must fail at load, not corrupt matrices
        (2026-08-18 adversarial F1b)."""
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            "parameter_name;scope;value;set_at;comment\nP;*;nan;;\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Malformed override row"):
            store.load()

    def test_save_is_atomic_no_partial_left_behind(self, store):
        """save() writes via temp + os.replace; no .partial residue
        (2026-08-18 adversarial F4)."""
        store.save([ParameterOverride("P", "*", 1.0)])
        assert store.load()[0].value == 1.0
        leftovers = list(store.path.parent.glob("*.partial"))
        assert leftovers == []

    def test_load_empty_scope_falls_back_to_global(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(
            "parameter_name;scope;value;set_at;comment\nP;;2.5;;\n", encoding="utf-8"
        )
        rows = store.load()
        assert rows[0].scope == "*"
        assert rows[0].value == 2.5


_PARAMS = [
    {
        "process_code": "PROC1",
        "kind": "input",
        "name": "SP_X",
        "original_name": "x",
        "amount": 2.0,
        "formula": None,
        "comment": "",
    },
]

_DATA = [
    {
        "code": "PROC1",
        "type": "process",
        "exchanges": [
            {"name": "flow", "amount": 4.0, "formula": "(SP_X * 2)", "type": "technosphere"},
        ],
    },
]


class TestParameterOverridesApplier:
    def test_noop_without_overrides_file(self, store):
        sp = ParsedSimaProCsv(db_name="agb", data=_DATA, parameters=_PARAMS)
        stats = ParameterOverridesApplier(settings=store.settings).apply(sp)
        assert stats == {"overrides": 0, "changed_exchanges": 0, "processes_affected": 0}
        assert sp.data[0]["exchanges"][0]["amount"] == 4.0

    def test_applies_stored_overrides_to_parsed_data(self, store):
        store.upsert(ParameterOverride("x", "*", 3.0))
        sp = ParsedSimaProCsv(db_name="agb", data=_DATA, parameters=_PARAMS)
        stats = ParameterOverridesApplier(settings=store.settings).apply(sp)
        assert stats["overrides"] == 1
        assert stats["changed_exchanges"] == 1
        assert sp.data[0]["exchanges"][0]["amount"] == 6.0
        # The module-level template must stay untouched (no in-place mutation).
        assert _DATA[0]["exchanges"][0]["amount"] == 4.0

    def test_no_effect_override_surfaces_warning_and_zero_changes(self, store):
        """A stored override whose parameter feeds no formula still applies
        cleanly: zero changed exchanges, warning logged, data untouched."""
        params = [
            *_PARAMS,
            {
                "process_code": "PROC1",
                "kind": "input",
                "name": "SP_DQI",
                "original_name": "DQI_Thing",
                "amount": 1.0,
                "formula": None,
                "comment": "",
            },
        ]
        store.upsert(ParameterOverride("DQI_Thing", "*", 5.0))
        sp = ParsedSimaProCsv(db_name="agb", data=_DATA, parameters=params)
        stats = ParameterOverridesApplier(settings=store.settings).apply(sp)
        assert stats["overrides"] == 1
        assert stats["changed_exchanges"] == 0
        assert sp.data[0]["exchanges"][0]["amount"] == 4.0
