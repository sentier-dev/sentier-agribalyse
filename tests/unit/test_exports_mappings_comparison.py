"""Unit tests for ``MappingsComparisonExporter`` — focus on the in-memory
``sp.data`` extraction path that lets the report run before DB write.

The exporter aggregates AGB biosphere exchanges by ``(name, cats, unit)``,
counts each linked target, and exposes the dominant target as the source
record's ``target``. The Brightway target-metadata lookup is exercised
end-to-end by the integration suite, not here.
"""

from __future__ import annotations

import pytest

from exports import MappingsComparisonExporter
from tests.fixtures.builders import make_dataset, make_exchange, make_settings


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


class TestBuildSourceIndexFromSpData:
    def test_groups_by_name_cats_unit(self, settings):
        sp_data = [
            make_dataset(
                "P1",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="CO2",
                        unit="kg",
                        categories=("air",),
                        input=("ef", "uuid-co2"),
                    ),
                    make_exchange(type="technosphere", name="electricity"),
                    make_exchange(
                        type="biosphere",
                        name="Water",
                        unit="m3",
                        categories=("water",),
                        input=("biosphere3", "bio-water"),
                    ),
                ],
            ),
            make_dataset(
                "P2",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="CO2",
                        unit="kg",
                        categories=("air",),
                        input=("ef", "uuid-co2"),
                    ),
                ],
            ),
        ]
        exporter = MappingsComparisonExporter(settings=settings, sp_data=sp_data)
        idx = exporter._build_source_index()

        assert set(idx.keys()) == {
            ("co2", ("air",), "kg"),
            ("water", ("water",), "m3"),
        }
        co2 = idx[("co2", ("air",), "kg")]
        assert co2.name == "CO2"
        assert co2.target == ("ef", "uuid-co2")
        assert co2.n == 2

    def test_skips_non_biosphere_and_anonymous(self, settings):
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="technosphere", name="electricity"),
                    make_exchange(type="biosphere", name="", input=("ef", "X")),
                    make_exchange(
                        type="biosphere",
                        name="Methane",
                        unit="kg",
                        categories=("air",),
                        input=("ef", "uuid-ch4"),
                    ),
                ],
            )
        ]
        idx = MappingsComparisonExporter(settings=settings, sp_data=sp_data)._build_source_index()
        assert list(idx) == [("methane", ("air",), "kg")]

    def test_unlinked_exchanges_kept_with_no_target(self, settings):
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="Mystery",
                        unit="kg",
                        categories=("air",),
                        input=None,
                    )
                ],
            )
        ]
        idx = MappingsComparisonExporter(settings=settings, sp_data=sp_data)._build_source_index()
        rec = idx[("mystery", ("air",), "kg")]
        assert rec.target is None
        assert rec.n == 1

    def test_dominant_target_is_deterministic_on_ties(self, settings):
        # Two distinct targets each appear once → tiebreak by lex (db, code).
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="X",
                        unit="kg",
                        categories=("air",),
                        input=("ef", "ZZZ"),
                    ),
                    make_exchange(
                        type="biosphere",
                        name="X",
                        unit="kg",
                        categories=("air",),
                        input=("biosphere3", "AAA"),
                    ),
                ],
            )
        ]
        idx = MappingsComparisonExporter(settings=settings, sp_data=sp_data)._build_source_index()
        rec = idx[("x", ("air",), "kg")]
        assert rec.target == ("biosphere3", "AAA")  # lex-smallest db wins on tie
