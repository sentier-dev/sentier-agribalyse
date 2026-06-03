"""Unit tests for the lifted bw2io strategy classes (Phase F3).

The strategies live under ``src/transforms/strategies/`` and are pure
dict transforms — no bw2data, no bw2io, no SQLite. Each test pins
behaviour we relied on from bw2io so the eventual deletion of bw2io
(F6) can land without surprise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transforms.strategies.biosphere import (
    BiosphereStrategyChain,
    DropUnspecifiedSubcategories,
    NormalizeBiosphereCategories,
    NormalizeBiosphereNames,
    NormalizeSimaproBiosphereCategories,
    NormalizeSimaproBiosphereNames,
    RemoveBiosphereLocationPrefixIfFlowInSameLocation,
    StripBiosphereExchangeLocations,
)
from transforms.strategies.labels import NormalizeSimaproLabelsToBrightwayStandard
from transforms.strategies.migrations import MigrationApplier, MigrationStore
from transforms.strategies.units import ChangeElectricityUnitMjToKwh, RescaleExchange

# ============================================================================
# RescaleExchange


class TestRescaleExchange:
    def test_rescales_undefined_uncertainty(self):
        exc = {"amount": 10.0, "uncertainty type": 0}
        RescaleExchange.apply(exc, 0.1)
        assert exc["amount"] == pytest.approx(1.0)
        assert exc["loc"] == pytest.approx(1.0)

    def test_rescales_no_uncertainty_with_existing_loc(self):
        exc = {"amount": 4.0, "loc": 4.0, "uncertainty type": 1}
        RescaleExchange.apply(exc, 2.0)
        assert exc["amount"] == 8.0
        assert exc["loc"] == 8.0

    def test_zero_factor_zeroes_amount_and_clears_scale(self):
        exc = {"amount": 5.0, "scale": 1.0, "shape": 2.0}
        RescaleExchange.apply(exc, 0)
        assert exc["amount"] == 0
        assert exc["loc"] == 0
        assert "scale" not in exc
        assert "shape" not in exc

    def test_formula_gets_wrapped_with_factor(self):
        exc = {"amount": 1.0, "formula": "x * 2", "uncertainty type": 0}
        RescaleExchange.apply(exc, 3.0)
        assert exc["formula"] == "(x * 2) * 3.0"

    def test_negative_factor_swaps_min_max_for_uniform(self):
        exc = {
            "amount": 1.0,
            "minimum": 1.0,
            "maximum": 2.0,
            "uncertainty type": 4,  # uniform
        }
        RescaleExchange.apply(exc, -1.0)
        # After scaling: minimum=-1, maximum=-2; sign-flip swaps them.
        assert exc["minimum"] == -2.0
        assert exc["maximum"] == -1.0

    def test_non_number_factor_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            RescaleExchange.apply({"amount": 1.0}, "not-a-number")  # type: ignore[arg-type]

    def test_bool_factor_rejected(self):
        with pytest.raises(ValueError):
            RescaleExchange.apply({"amount": 1.0}, True)  # type: ignore[arg-type]


# ============================================================================
# ChangeElectricityUnitMjToKwh


class TestChangeElectricityUnitMjToKwh:
    def test_converts_megajoule_electricity_to_kwh(self):
        data = [
            {
                "exchanges": [
                    {"name": "Electricity, low voltage", "unit": "megajoule", "amount": 3.6}
                ]
            }
        ]
        ChangeElectricityUnitMjToKwh()(data)
        exc = data[0]["exchanges"][0]
        assert exc["unit"] == "kilowatt hour"
        assert exc["amount"] == pytest.approx(1.0)

    def test_skips_non_electricity_names(self):
        data = [{"exchanges": [{"name": "Diesel", "unit": "megajoule", "amount": 1.0}]}]
        ChangeElectricityUnitMjToKwh()(data)
        assert data[0]["exchanges"][0]["unit"] == "megajoule"
        assert data[0]["exchanges"][0]["amount"] == 1.0

    def test_matches_market_for_electricity(self):
        data = [
            {"exchanges": [{"name": "market for electricity", "unit": "megajoule", "amount": 7.2}]}
        ]
        ChangeElectricityUnitMjToKwh()(data)
        assert data[0]["exchanges"][0]["unit"] == "kilowatt hour"
        assert data[0]["exchanges"][0]["amount"] == pytest.approx(2.0)

    def test_matches_market_group_for_electricity(self):
        data = [
            {
                "exchanges": [
                    {
                        "name": "market group for electricity, medium voltage",
                        "unit": "megajoule",
                        "amount": 3.6,
                    }
                ]
            }
        ]
        ChangeElectricityUnitMjToKwh()(data)
        assert data[0]["exchanges"][0]["unit"] == "kilowatt hour"

    def test_skips_when_unit_not_megajoule(self):
        data = [
            {"exchanges": [{"name": "Electricity, AC", "unit": "kilowatt hour", "amount": 1.0}]}
        ]
        ChangeElectricityUnitMjToKwh()(data)
        assert data[0]["exchanges"][0]["unit"] == "kilowatt hour"
        assert data[0]["exchanges"][0]["amount"] == 1.0


# ============================================================================
# Biosphere strategies


class TestStripBiosphereExchangeLocations:
    def test_drops_location_from_biosphere_only(self):
        data = [
            {
                "exchanges": [
                    {"type": "biosphere", "location": "GLO", "name": "CO2"},
                    {"type": "technosphere", "location": "FR", "name": "tap water"},
                ]
            }
        ]
        StripBiosphereExchangeLocations()(data)
        assert "location" not in data[0]["exchanges"][0]
        assert data[0]["exchanges"][1]["location"] == "FR"


class TestDropUnspecifiedSubcategories:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (("air", "unspecified"), ("air",)),
            (("air", "(unspecified)"), ("air",)),
            (("air", ""), ("air",)),
            (("air", None), ("air",)),
            (("air", "urban"), ("air", "urban")),
        ],
    )
    def test_strips_trailing_unspecified_from_exchange(self, raw, expected):
        data = [{"exchanges": [{"categories": raw}]}]
        DropUnspecifiedSubcategories()(data)
        assert data[0]["exchanges"][0]["categories"] == expected

    def test_strips_trailing_unspecified_from_dataset(self):
        data = [{"categories": ["air", "unspecified"], "exchanges": []}]
        DropUnspecifiedSubcategories()(data)
        assert data[0]["categories"] == ["air"]


class TestNormalizeSimaproBiosphereCategories:
    def test_maps_top_level_simapro_categories(self):
        data = [
            {
                "exchanges": [
                    {"type": "biosphere", "categories": ("Air",)},
                    {"type": "biosphere", "categories": ("Resources", "in ground")},
                    {"type": "biosphere", "categories": ("Air", "high. pop.")},
                ]
            }
        ]
        NormalizeSimaproBiosphereCategories()(data)
        out = [e["categories"] for e in data[0]["exchanges"]]
        assert out[0] == ("air",)
        assert out[1] == ("natural resource", "in ground")
        assert out[2] == ("air", "urban air close to ground")

    def test_skips_non_biosphere_exchanges(self):
        data = [
            {
                "exchanges": [
                    {"type": "technosphere", "categories": ("Air",)},
                ]
            }
        ]
        NormalizeSimaproBiosphereCategories()(data)
        assert data[0]["exchanges"][0]["categories"] == ("Air",)


class TestNormalizeSimaproBiosphereNames:
    @pytest.fixture
    def mapping_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "simapro-biosphere.json"
        path.write_text(
            json.dumps(
                [
                    ["air", "1-Butanol", "Butanol"],
                    ["water", "Methanol", "Methanol-water"],
                ]
            )
        )
        return path

    def test_renames_matching_flows(self, mapping_path):
        data = [
            {
                "exchanges": [
                    {"type": "biosphere", "categories": ("air",), "name": "1-Butanol"},
                    {"type": "biosphere", "categories": ("water",), "name": "Methanol"},
                    {"type": "biosphere", "categories": ("air",), "name": "Other"},
                ]
            }
        ]
        NormalizeSimaproBiosphereNames(json_path=mapping_path)(data)
        names = [e["name"] for e in data[0]["exchanges"]]
        assert names == ["Butanol", "Methanol-water", "Other"]

    def test_raises_when_mapping_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="simapro-biosphere"):
            NormalizeSimaproBiosphereNames(json_path=tmp_path / "ghost.json")(
                [{"exchanges": [{"type": "biosphere", "name": "x", "categories": ("air",)}]}]
            )


class TestRemoveBiosphereLocationPrefixIfFlowInSameLocation:
    def test_strips_location_suffix_when_match(self):
        data = [
            {
                "location": "AR",
                "exchanges": [
                    {"type": "biosphere", "name": "Ammonia, AR"},
                    {"type": "biosphere", "name": "Carbon dioxide"},
                ],
            }
        ]
        RemoveBiosphereLocationPrefixIfFlowInSameLocation()(data)
        out = data[0]["exchanges"]
        assert out[0]["name"] == "Ammonia"
        assert out[0]["simapro name"] == "Ammonia, AR"
        assert out[1]["name"] == "Carbon dioxide"

    def test_does_nothing_when_dataset_has_no_string_location(self):
        data = [{"location": None, "exchanges": [{"type": "biosphere", "name": "X, AR"}]}]
        RemoveBiosphereLocationPrefixIfFlowInSameLocation()(data)
        assert data[0]["exchanges"][0]["name"] == "X, AR"


# ============================================================================
# MigrationStore + MigrationApplier


class TestMigrationStore:
    def test_loads_existing_migration(self, tmp_path: Path):
        path = tmp_path / "m.json"
        payload = {"fields": ["name"], "data": [[["a"], {"name": "b"}]]}
        path.write_text(json.dumps(payload))
        store = MigrationStore(directory=tmp_path)
        loaded = store.load("m")
        assert loaded == payload

    def test_raises_clearly_when_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="migration 'ghost'"):
            MigrationStore(directory=tmp_path).load("ghost")


class TestMigrationApplier:
    def _store(self, tmp_path: Path, name: str, payload: dict) -> MigrationStore:
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
        return MigrationStore(directory=tmp_path)

    def test_apply_to_exchanges_overwrites_matching_fields(self, tmp_path: Path):
        store = self._store(
            tmp_path,
            "rename",
            {
                "fields": ["name"],
                "data": [
                    [["old"], {"name": "new"}],
                ],
            },
        )
        applier = MigrationApplier.from_store(store, "rename")
        data = [{"exchanges": [{"name": "old"}, {"name": "stay"}]}]
        applier.apply_to_exchanges(data)
        assert [e["name"] for e in data[0]["exchanges"]] == ["new", "stay"]

    def test_apply_to_datasets_overwrites_dataset_fields(self, tmp_path: Path):
        store = self._store(
            tmp_path,
            "ds",
            {
                "fields": ["name"],
                "data": [[["old"], {"name": "new"}]],
            },
        )
        applier = MigrationApplier.from_store(store, "ds")
        data = [{"name": "old"}, {"name": "stay"}]
        applier.apply_to_datasets(data)
        assert [d["name"] for d in data] == ["new", "stay"]

    def test_multiplier_field_rescales_amount(self, tmp_path: Path):
        store = self._store(
            tmp_path,
            "rescale",
            {
                "fields": ["name"],
                "data": [[["x"], {"multiplier": 0.5, "unit": "g"}]],
            },
        )
        applier = MigrationApplier.from_store(store, "rescale")
        data = [{"exchanges": [{"name": "x", "amount": 10, "uncertainty type": 0}]}]
        applier.apply_to_exchanges(data)
        exc = data[0]["exchanges"][0]
        assert exc["amount"] == pytest.approx(5.0)
        assert exc["unit"] == "g"

    def test_categories_round_trip_as_lists(self, tmp_path: Path):
        store = self._store(
            tmp_path,
            "cats",
            {
                "fields": ["categories", "type"],
                "data": [[[["a", "b"], "biosphere"], {"categories": ["a", "c"]}]],
            },
        )
        applier = MigrationApplier.from_store(store, "cats")
        data = [{"exchanges": [{"categories": ("a", "b"), "type": "biosphere"}]}]
        applier.apply_to_exchanges(data)
        assert data[0]["exchanges"][0]["categories"] == ["a", "c"]


# ============================================================================
# Migrations driving NormalizeBiosphereCategories / Names


class TestNormalizeBiosphereWithStore:
    @pytest.fixture
    def cats_store(self, tmp_path: Path) -> MigrationStore:
        path = tmp_path / "biosphere-2-3-categories.json"
        path.write_text(
            json.dumps(
                {
                    "fields": ["categories", "type"],
                    "data": [
                        [
                            [["resource", "in ground"], "biosphere"],
                            {"categories": ["natural resource", "in ground"]},
                        ]
                    ],
                }
            )
        )
        return MigrationStore(directory=tmp_path)

    def test_normalize_categories_walks_exchanges(self, cats_store):
        data = [
            {
                "exchanges": [
                    {
                        "type": "biosphere",
                        "categories": ("resource", "in ground"),
                        "name": "X",
                    }
                ]
            }
        ]
        NormalizeBiosphereCategories(store=cats_store)(data)
        assert data[0]["exchanges"][0]["categories"] == [
            "natural resource",
            "in ground",
        ]

    def test_normalize_names_keyed_on_fields(self, tmp_path: Path):
        path = tmp_path / "biosphere-2-3-names.json"
        path.write_text(
            json.dumps(
                {
                    "fields": ["name", "categories", "unit", "type"],
                    "data": [
                        [
                            ["OldName", ["air"], "kg", "biosphere"],
                            {"name": "NewName"},
                        ]
                    ],
                }
            )
        )
        store = MigrationStore(directory=tmp_path)
        data = [
            {
                "exchanges": [
                    {
                        "name": "OldName",
                        "categories": ("air",),
                        "unit": "kg",
                        "type": "biosphere",
                    }
                ]
            }
        ]
        NormalizeBiosphereNames(store=store)(data)
        assert data[0]["exchanges"][0]["name"] == "NewName"


# ============================================================================
# NormalizeSimaproLabelsToBrightwayStandard


class TestNormalizeSimaproLabelsToBrightwayStandard:
    def test_renames_context_to_categories_for_unlinked_only(self):
        data = [
            {
                "exchanges": [
                    {"context": ["air"]},
                    {"context": ["water"], "input": ("a", "b")},
                ]
            }
        ]
        NormalizeSimaproLabelsToBrightwayStandard()(data)
        unlinked, linked = data[0]["exchanges"]
        assert unlinked["categories"] == ("air",)
        assert "categories" not in linked  # linked exchange untouched

    def test_renames_identifier_to_code(self):
        data = [{"exchanges": [{"identifier": "abc"}]}]
        NormalizeSimaproLabelsToBrightwayStandard()(data)
        assert data[0]["exchanges"][0]["code"] == "abc"

    def test_does_not_overwrite_existing_categories(self):
        data = [{"exchanges": [{"context": ["air"], "categories": ("water",)}]}]
        NormalizeSimaproLabelsToBrightwayStandard()(data)
        assert data[0]["exchanges"][0]["categories"] == ("water",)

    def test_instance_exposes_dunder_name_for_bw2io_apply_strategy(self):
        # bw2io.importers.base.apply_strategy reads ``strategy.__name__`` to
        # name the step in its progress log. Class ``__name__`` is not
        # inherited by instances, so we expose it explicitly. Regression
        # guard: without this attribute, ``sp.apply_strategy(<instance>)``
        # raises AttributeError before any data transform happens.
        instance = NormalizeSimaproLabelsToBrightwayStandard()
        assert instance.__name__ == "normalize_simapro_labels_to_brightway_standard"


# ============================================================================
# BiosphereStrategyChain — composite from_paths


class TestBiosphereStrategyChain:
    def test_from_paths_builds_six_step_chain(self, tmp_path: Path):
        # Create the minimum data files the chain expects.
        (tmp_path / "simapro-biosphere.json").write_text("[]")
        (tmp_path / "biosphere-2-3-categories.json").write_text(
            json.dumps({"fields": ["name"], "data": []})
        )
        (tmp_path / "biosphere-2-3-names.json").write_text(
            json.dumps({"fields": ["name"], "data": []})
        )
        chain = BiosphereStrategyChain.from_paths(bw2io_data_dir=tmp_path)
        assert len(chain.strategies) == 6
        names = [type(s).__name__ for s in chain.strategies]
        assert names == [
            "NormalizeSimaproBiosphereCategories",
            "NormalizeSimaproBiosphereNames",
            "NormalizeBiosphereCategories",
            "NormalizeBiosphereNames",
            "StripBiosphereExchangeLocations",
            "DropUnspecifiedSubcategories",
        ]

    def test_apply_runs_every_strategy(self, tmp_path: Path):
        # Trivial data + empty migrations ⇒ should execute end-to-end.
        (tmp_path / "simapro-biosphere.json").write_text("[]")
        (tmp_path / "biosphere-2-3-categories.json").write_text(
            json.dumps({"fields": ["name"], "data": []})
        )
        (tmp_path / "biosphere-2-3-names.json").write_text(
            json.dumps({"fields": ["name"], "data": []})
        )
        chain = BiosphereStrategyChain.from_paths(bw2io_data_dir=tmp_path)

        class _Sp:
            def __init__(self):
                self.data = [
                    {
                        "exchanges": [
                            {
                                "type": "biosphere",
                                "name": "Ammonia",
                                "location": "GLO",
                                "categories": ("Air", "unspecified"),
                            }
                        ]
                    }
                ]

        sp = _Sp()
        chain.apply(sp)
        exc = sp.data[0]["exchanges"][0]
        # Air → air via SimaPro categories; unspecified dropped at the end;
        # location stripped because biosphere; name unchanged because mapping is empty.
        assert exc["categories"] == ("air",)
        assert "location" not in exc
