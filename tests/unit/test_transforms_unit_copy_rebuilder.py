"""``UnitCopyRebuilder`` — stripped ecoinvent unit processes rebuilt from the snapshot."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from matching.ecoinvent_catalog import EcoinventActivityRef, EcoinventCatalog
from tests.fixtures.builders import FakeSimaProImporter, make_registry, make_unit_aliases_df
from transforms import UnitCopyRebuilder

EI = "ecoinvent-3.9.1-cutoff"
BIO = "ecoinvent-3.9.1-biosphere"
AGB = "agribalyse-3.2"

MARKET = "Soybean {GLO}| market for soybean | Cut-off, U - Copied from Ecoinvent U"
MARKET_BR = "Soybean {BR}| market for soybean | Cut-off, U - Copied from Ecoinvent U"
ADAPTED = "Soybean {BR-GO}| soybean production | Cut-off, U - Adapted from Ecoinvent U"
EXTRACT = (
    "Combustion diesel (from data :Transport, barge {RER}| processing | Cut-off, U - Copied "
    "from Ecoinvent) U"
)


def _ref(code, name, location, product, unit="kilogram") -> EcoinventActivityRef:
    return EcoinventActivityRef(
        db=EI, code=code, name=name, unit=unit, location=location, reference_product=product
    )


CATALOG = EcoinventCatalog(
    db_names=(EI,),
    activities=(
        _ref("mkt-glo", "market for soybean", "GLO", "soybean"),
        _ref("mkt-br", "market for soybean", "BR", "soybean"),
        _ref("soy-go", "soybean production", "BR-GO", "soybean"),
        _ref("soy-us", "soybean production", "US", "soybean"),
        _ref("lorry", "market for transport, lorry", "GLO", "transport, lorry", "ton kilometer"),
        _ref("pest-mkt", "market for pesticide, unspecified", "GLO", "pesticide, unspecified"),
        _ref(
            "pest-dcp",
            "dichloropropene to generic market for pesticide, unspecified",
            "GLO",
            "pesticide, unspecified",
        ),
        _ref("barge", "transport, barge", "RER", "transport, barge", "ton kilometer"),
        _ref("diesel", "market for diesel", "RER", "diesel"),
        _ref("port", "market for port facilities", "GLO", "port facilities", "unit"),
    ),
)


def _exchanges() -> pd.DataFrame:
    rows = [
        # market GLO: 0.6 from market BR, 0.4 from US production, transport, own losses
        ("mkt-glo", EI, "mkt-glo", 1.0, "production"),
        ("mkt-glo", EI, "mkt-br", 0.6, "technosphere"),
        ("mkt-glo", EI, "soy-us", 0.4, "technosphere"),
        ("mkt-glo", EI, "lorry", 0.2, "technosphere"),
        ("mkt-glo", EI, "mkt-glo", 0.01, "technosphere"),
        ("mkt-glo", BIO, "co2-air", 0.05, "biosphere"),
        # market BR: all from Goiás, plus the pesticide market
        ("mkt-br", EI, "mkt-br", 1.0, "production"),
        ("mkt-br", EI, "soy-go", 1.0, "technosphere"),
        ("mkt-br", EI, "pest-mkt", 0.002, "technosphere"),
        # barge: diesel, infrastructure, air and water emissions
        ("barge", EI, "barge", 1.0, "production"),
        ("barge", EI, "diesel", 0.00939, "technosphere"),
        ("barge", EI, "port", 1e-9, "technosphere"),
        ("barge", BIO, "co2-air", 0.03, "biosphere"),
        ("barge", BIO, "nox-air", 0.0005, "biosphere"),
        ("barge", BIO, "oil-water", 0.0001, "biosphere"),
    ]
    return pd.DataFrame(
        [
            {
                "output_database": EI,
                "output_code": out,
                "input_database": db,
                "input_code": code,
                "amount": amount,
                "type": kind,
            }
            for out, db, code, amount, kind in rows
        ]
    )


def _adapted_dataset(unit: str = "kg") -> list[dict]:
    return [
        {
            "database": AGB,
            "code": "AGB-SOY-GO",
            "type": "process",
            "name": "soybean production BR-GO",
            "reference product": ADAPTED,
            "unit": unit,
            "exchanges": [
                {"type": "production", "amount": 1.0, "input": (AGB, "agb-soy-go-product")},
            ],
        },
        {"database": AGB, "code": "agb-soy-go-product", "type": "product", "name": ADAPTED},
    ]


@pytest.fixture
def rebuilder(settings):
    paths = settings.paths
    _exchanges().to_parquet(paths.ecoinvent_exchanges_parquet)
    paths.copied_unit_processes_json.write_text(json.dumps({"names": [MARKET, MARKET_BR, EXTRACT]}))
    pd.DataFrame(
        [
            {"database": BIO, "code": "co2-air", "categories": ["air"]},
            {
                "database": BIO,
                "code": "nox-air",
                "categories": ["air", "urban air close to ground"],
            },
            {"database": BIO, "code": "oil-water", "categories": ["water", "surface water"]},
        ]
    ).to_parquet(paths.registry_biosphere_catalog)
    paths.edge_label_corrections_json.write_text(
        json.dumps(
            {
                "replace": [
                    {
                        "source": {
                            "name": "Pesticide, unspecified {GLO}| market for | Cut-off, S - "
                            "Copied from Ecoinvent U"
                        },
                        "target": {
                            "name": "Pesticide, unspecified {GLO}| dichloropropene to generic "
                            "market for pesticide, unspecified | Cut-off, U"
                        },
                    }
                ]
            }
        )
    )
    registry = make_registry(
        settings,
        unit_aliases=make_unit_aliases_df([{"alias": "kg", "canonical": "kilogram"}]),
    )
    return UnitCopyRebuilder(
        settings=settings, catalog=CATALOG, unit_converter=registry.unit_converter
    )


def _by_code(sp) -> dict[str, dict]:
    return {ds["code"]: ds for ds in sp.data}


def _inputs(ds: dict, kind: str = "technosphere") -> dict[tuple[str, str], float]:
    return {tuple(e["input"]): e["amount"] for e in ds["exchanges"] if e["type"] == kind}


class TestRebuild:
    def test_references_are_redirected_to_the_rebuilt_products(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuild = rebuilder.apply(sp)

        assert rebuild.redirects == {
            (EI, "mkt-glo"): (AGB, "ecoinvent-unit-copy-product-mkt-glo"),
            (EI, "mkt-br"): (AGB, "ecoinvent-unit-copy-product-mkt-br"),
        }
        assert rebuild.stats["rebuilt"] == 2

    def test_rebuilt_process_keeps_ecoinvent_amounts(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuilder.apply(sp)

        market = _by_code(sp)["ecoinvent-unit-copy-mkt-glo"]
        assert market["name"] == MARKET
        assert market["database"] == AGB
        assert market["location"] == "GLO"
        assert _inputs(market, "production") == {(AGB, "ecoinvent-unit-copy-product-mkt-glo"): 1.0}
        assert _inputs(market, "biosphere") == {(BIO, "co2-air"): 0.05}
        assert _inputs(market)[(EI, "soy-us")] == 0.4
        assert _inputs(market)[(EI, "lorry")] == 0.2

    def test_input_that_is_a_unit_copy_points_to_its_rebuilt_twin(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuilder.apply(sp)

        market = _by_code(sp)["ecoinvent-unit-copy-mkt-glo"]
        assert _inputs(market)[(AGB, "ecoinvent-unit-copy-product-mkt-br")] == 0.6

    def test_own_consumption_stays_on_the_rebuilt_process(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuilder.apply(sp)

        market = _by_code(sp)["ecoinvent-unit-copy-mkt-glo"]
        assert _inputs(market)[(AGB, "ecoinvent-unit-copy-product-mkt-glo")] == 0.01

    def test_input_with_an_adapted_counterpart_points_to_ademes_dataset(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuild = rebuilder.apply(sp)

        market_br = _by_code(sp)["ecoinvent-unit-copy-mkt-br"]
        assert _inputs(market_br)[(AGB, "agb-soy-go-product")] == 1.0
        assert (EI, "soy-go") not in _inputs(market_br)
        assert rebuild.stats["inputs_to_adapted"] == 1

    def test_adapted_dataset_in_an_unconvertible_unit_is_not_used(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset(unit="m3"))

        rebuild = rebuilder.apply(sp)

        market_br = _by_code(sp)["ecoinvent-unit-copy-mkt-br"]
        assert _inputs(market_br)[(EI, "soy-go")] == 1.0
        assert rebuild.stats["inputs_unit_mismatch"] == 1

    def test_shared_simapro_name_follows_the_block_the_library_holds(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuild = rebuilder.apply(sp)

        market_br = _by_code(sp)["ecoinvent-unit-copy-mkt-br"]
        assert _inputs(market_br)[(EI, "pest-dcp")] == 0.002
        assert (EI, "pest-mkt") not in _inputs(market_br)
        assert rebuild.stats["inputs_to_shared_name_block"] == 1

    def test_product_node_points_back_to_its_process(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuilder.apply(sp)

        product = _by_code(sp)["ecoinvent-unit-copy-product-mkt-glo"]
        assert product["type"] == "product"
        assert product["reference process"] == (AGB, "ecoinvent-unit-copy-mkt-glo")


class TestCombustionExtract:
    def test_air_emissions_per_kilogram_of_fuel(self, rebuilder):
        sp = FakeSimaProImporter(data=[])

        rebuild = rebuilder.apply(sp)

        target = rebuild.by_name[EXTRACT]
        process = _by_code(sp)["ecoinvent-combustion-extract-barge"]
        assert target == (AGB, "ecoinvent-combustion-extract-product-barge")
        assert _inputs(process, "production") == {target: 0.00939}
        assert _inputs(process, "biosphere") == {(BIO, "co2-air"): 0.03, (BIO, "nox-air"): 0.0005}
        assert _inputs(process) == {}
        assert process["unit"] == "kilogram"


class TestNoOp:
    def test_without_the_name_list_nothing_is_rebuilt(self, rebuilder, settings):
        settings.paths.copied_unit_processes_json.unlink()
        sp = FakeSimaProImporter(data=_adapted_dataset())

        rebuild = rebuilder.apply(sp)

        assert rebuild.redirects == {}
        assert len(sp.data) == 2

    def test_without_the_ecoinvent_snapshot_nothing_is_rebuilt(self, rebuilder, settings):
        settings.paths.ecoinvent_exchanges_parquet.unlink()
        sp = FakeSimaProImporter(data=[])

        assert rebuilder.apply(sp).redirects == {}


class TestPruneUnused:
    def test_keeps_what_a_consumer_reaches_and_drops_the_rest(self, rebuilder):
        consumer = {
            "database": AGB,
            "code": "BREAD",
            "type": "process",
            "exchanges": [
                {
                    "type": "technosphere",
                    "amount": 1.0,
                    "input": (AGB, "ecoinvent-unit-copy-product-mkt-glo"),
                }
            ],
        }
        sp = FakeSimaProImporter(data=[*_adapted_dataset(), consumer])
        rebuild = rebuilder.apply(sp)

        dropped = rebuilder.prune_unused(sp, rebuild)

        codes = set(_by_code(sp))
        assert dropped == 1  # the combustion extract nobody references
        assert "ecoinvent-unit-copy-mkt-glo" in codes
        assert "ecoinvent-unit-copy-mkt-br" in codes  # reached through the GLO market
        assert "ecoinvent-combustion-extract-barge" not in codes
        assert "ecoinvent-combustion-extract-product-barge" not in codes

    def test_drops_everything_when_nothing_is_referenced(self, rebuilder):
        sp = FakeSimaProImporter(data=_adapted_dataset())
        rebuild = rebuilder.apply(sp)

        assert rebuilder.prune_unused(sp, rebuild) == 3
        assert set(_by_code(sp)) == {"AGB-SOY-GO", "agb-soy-go-product"}
