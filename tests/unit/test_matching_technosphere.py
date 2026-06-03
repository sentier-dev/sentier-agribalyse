"""Unit tests for ``TechnosphereMatcher`` — parquet-backed, no SQLite.

After Phase F3 the matcher uses ``EcoinventCatalog`` directly, so the
tests build a tiny in-memory catalog rather than mocking
``sp.match_database`` calls.
"""

from __future__ import annotations

import json
import sys

import pytest

from matching.audit import DropTallyTracker
from matching.ecoinvent_catalog import EcoinventActivityRef, EcoinventCatalog
from matching.technosphere import TechnosphereMatcher
from tests.fixtures.builders import (
    FakeSimaProImporter,
    make_dataset,
    make_exchange,
    make_registry,
    make_unit_conversions_df,
)


def _catalog(refs: list[EcoinventActivityRef]) -> EcoinventCatalog:
    db_names = tuple(sorted({r.db for r in refs})) if refs else ()
    return EcoinventCatalog(db_names=db_names, activities=tuple(refs))


def _ref(**kw) -> EcoinventActivityRef:
    base = dict(
        db="ecoinvent-3.9.1-cutoff",
        code="ei-1",
        name="market for tap water",
        unit="kilogram",
        location="RoW",
        reference_product="tap water",
    )
    base.update(kw)
    return EcoinventActivityRef(**base)


class TestTechnosphereMatcherStrict:
    def test_links_exchange_via_full_4_field_match(self, settings):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="market for tap water",
                            unit="kilogram",
                            amount=1.0,
                            location="RoW",
                            **{"reference product": "tap water"},
                        )
                    ],
                )
            ]
        )
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=make_registry(settings),
            catalog=_catalog([_ref(code="ei-tap-water")]),
            drops=DropTallyTracker(),
        )
        stats = matcher.match(sp)
        assert stats.n_total == 1
        assert stats.n_linked == 1
        assert sp.data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-cutoff",
            "ei-tap-water",
        )

    def test_leaves_unlinked_when_no_match_exists(self, settings):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="ghost",
                            unit="kilogram",
                            amount=1.0,
                        )
                    ],
                )
            ]
        )
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=make_registry(settings),
            catalog=_catalog([]),
            drops=DropTallyTracker(),
        )
        stats = matcher.match(sp)
        assert stats.n_linked == 0
        assert "input" not in sp.data[0]["exchanges"][0]


class TestTechnosphereMatcherUnitConversion:
    def test_unit_conversion_rescales_unlinked_technosphere_exchanges(self, settings):
        conv = make_unit_conversions_df(
            [{"source_unit": "m", "target_unit": "km", "multiplier": 0.001}]
        )
        registry = make_registry(settings, unit_conversions=conv)
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(type="technosphere", name="Pipe", unit="m", amount=2000.0)
                    ],
                )
            ]
        )
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=registry,
            catalog=_catalog([]),
            drops=DropTallyTracker(),
        )
        stats = matcher.match(sp)
        exc = sp.data[0]["exchanges"][0]
        assert exc["unit"] == "km"
        assert exc["amount"] == pytest.approx(2.0)
        assert stats.n_unit_conversions_applied == 1

    def test_skips_already_linked_exchanges(self, settings):
        conv = make_unit_conversions_df(
            [{"source_unit": "m", "target_unit": "km", "multiplier": 0.001}]
        )
        registry = make_registry(settings, unit_conversions=conv)
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="Pipe",
                            unit="m",
                            amount=2000.0,
                            input=("ecoinvent", "preexisting"),
                        )
                    ],
                )
            ]
        )
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=registry,
            catalog=_catalog([]),
            drops=DropTallyTracker(),
        )
        stats = matcher.match(sp)
        exc = sp.data[0]["exchanges"][0]
        assert exc["amount"] == 2000.0  # not rescaled
        assert exc["unit"] == "m"
        assert stats.n_unit_conversions_applied == 0


class TestTechnosphereMatcherCustomFixes:
    def test_custom_fixes_json_loaded_when_present(self, settings, monkeypatch):
        path = settings.paths.custom_technosphere_fixes_json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"replace": [{"a": 1}, {"a": 2}]}))

        rn = sys.modules["randonneur"]
        called = {"n": 0}
        monkeypatch.setattr(
            rn,
            "migrate_edges",
            lambda graph, migrations, config: called.__setitem__("n", called["n"] + 1) or graph,
        )

        sp = FakeSimaProImporter(data=[make_dataset("P", exchanges=[])])
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=make_registry(settings),
            catalog=_catalog([]),
            drops=DropTallyTracker(),
        )
        stats = matcher.match(sp)
        assert called["n"] == 1
        assert stats.custom_fixes_applied == 2


class TestTechnosphereMatcherRelaxed:
    def test_relaxed_match_recovers_capitalised_reference_product_drift(self, settings):
        """SimaPro encodes the refprod as 'Drying of feed grain' (capital D);
        ecoinvent stores 'drying of feed grain' (lowercase). Strict match misses;
        relaxed (3-field) match recovers it."""
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="drying of feed grain",
                            unit="kilogram",
                            amount=1.0,
                            location="CA-QC",
                            **{"reference product": "Drying of feed grain"},
                        )
                    ],
                )
            ]
        )
        # Ecoinvent has the lowercase refprod — strict match would miss
        # because of the capitalised refprod on the exchange. Relaxed match
        # drops refprod and links uniquely.
        catalog = _catalog(
            [
                _ref(
                    code="ei-drying-grain",
                    name="drying of feed grain",
                    unit="kilogram",
                    location="CA-QC",
                    reference_product="drying of feed grain",
                )
            ]
        )
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=make_registry(settings),
            catalog=catalog,
            drops=DropTallyTracker(),
        )
        stats = matcher.match(sp)
        assert stats.n_linked == 1
        assert sp.data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-cutoff",
            "ei-drying-grain",
        )


class TestTechnosphereMatcherCounts:
    def test_counts_technosphere_total_and_linked(self, settings):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(type="technosphere", name="A", unit="kg", input=("db", "a")),
                        make_exchange(type="technosphere", name="B", unit="kg"),
                        make_exchange(type="biosphere", name="bio", unit="kg"),
                    ],
                )
            ]
        )
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=make_registry(settings),
            catalog=_catalog([]),
            drops=DropTallyTracker(),
        )
        stats = matcher.match(sp)
        # Only technosphere exchanges counted; biosphere ignored.
        assert stats.n_total == 2
        assert stats.n_linked == 1


class TestTechnosphereMatcherElectricity:
    def test_converts_megajoule_electricity_before_strict_match(self, settings):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="Electricity, low voltage",
                            unit="megajoule",
                            amount=3.6,
                            location="FR",
                            **{"reference product": "electricity, low voltage"},
                        )
                    ],
                )
            ]
        )
        catalog = _catalog(
            [
                _ref(
                    code="ei-elec-fr",
                    name="electricity, low voltage",
                    unit="kilowatt hour",
                    location="FR",
                    reference_product="electricity, low voltage",
                )
            ]
        )
        matcher = TechnosphereMatcher(
            settings=settings,
            registry=make_registry(settings),
            catalog=catalog,
            drops=DropTallyTracker(),
        )
        # The strict match in the strict pass uses the lowercased name from
        # the source. We crafted the source name to match the catalog exactly
        # after the unit shift.
        stats = matcher.match(sp)
        exc = sp.data[0]["exchanges"][0]
        # Unit converted MJ → kWh.
        assert exc["unit"] == "kilowatt hour"
        assert exc["amount"] == pytest.approx(1.0)
        # Linked to the kWh-shaped ecoinvent activity.
        assert stats.n_linked == 1
