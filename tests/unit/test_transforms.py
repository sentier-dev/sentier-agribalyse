"""Unit tests for the SimaPro CSV transform classes."""

from __future__ import annotations

import json
import math
import sys

import pandas as pd
import pytest

from matching.audit import SuppressedStrategyLog
from matching.strategy_runner import StrategyRunner
from tests.fixtures.builders import (
    FakeSimaProImporter,
    make_bio_catalog,
    make_dataset,
    make_exchange,
    make_registry,
)
from transforms import (
    AggregateDeleter,
    BiosphereFlowmapApplier,
    BiosphereLabelNormaliser,
    BioStrategyChain,
    EdgeLabelCorrector,
    InternalAgbLinker,
    OrphanActivityPurger,
    OrphanProductRelinker,
    ProductionReclassifier,
    RestoreSimaproNamesTransform,
    SimaProImporter,
    StandardLabelNormaliser,
    WasteTreatmentDummyFixer,
)

# ============================================================================
# Stateless transforms


class TestProductionReclassifier:
    def test_flips_functional_false_production_to_technosphere(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(type="production", name="Real product", functional=True),
                        make_exchange(type="production", name="Co-product waste", functional=False),
                    ],
                )
            ]
        )
        result = ProductionReclassifier().apply(sp)
        types = [e["type"] for e in sp.data[0]["exchanges"]]
        assert types == ["production", "technosphere"]
        assert result == {"reclassified": 1}


_DUMMY_COMMENT = "Dummy edge inserted to stop auto-generation of unitary production edge"


class TestWasteTreatmentDummyFixer:
    def test_promotes_dummy_to_functional_unit_production(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Waste-treatment-A",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="waste-A",
                            functional=True,
                            amount=1.0,
                        ),
                        make_exchange(
                            type="production",
                            name="waste-A",
                            functional=False,
                            amount=0,
                            comment=_DUMMY_COMMENT,
                        ),
                    ],
                )
            ]
        )
        result = WasteTreatmentDummyFixer().apply(sp)
        prod = next(e for e in sp.data[0]["exchanges"] if e["type"] == "production")
        waste = next(e for e in sp.data[0]["exchanges"] if e["type"] == "technosphere")
        assert prod["amount"] == 1.0
        assert prod["functional"] is True
        # Allocation contract for multifunctional packages: production receives 1.0,
        # every other functional edge receives 0.0, so the (zero) dummy impacts
        # all flow to the production and multifunctional's allocation walk succeeds.
        assert prod["properties"]["manual_allocation"] == 1.0
        assert waste["properties"]["manual_allocation"] == 0.0
        assert result == {"dummies_promoted": 1}

    def test_preserves_existing_properties_when_annotating_other_functional_edge(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Waste-treatment-B",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="waste-B",
                            functional=True,
                            amount=1.0,
                            properties={"price": 0.5},
                        ),
                        make_exchange(
                            type="production",
                            name="waste-B",
                            functional=False,
                            amount=0,
                            comment=_DUMMY_COMMENT,
                        ),
                    ],
                )
            ]
        )
        WasteTreatmentDummyFixer().apply(sp)
        waste = next(e for e in sp.data[0]["exchanges"] if e["type"] == "technosphere")
        assert waste["properties"]["price"] == 0.5
        assert waste["properties"]["manual_allocation"] == 0.0

    def test_no_property_changes_when_no_dummy_promoted(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Healthy",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="waste-C",
                            functional=True,
                            amount=1.0,
                        ),
                        make_exchange(
                            type="production",
                            name="real-product",
                            functional=True,
                            amount=1.0,
                        ),
                    ],
                )
            ]
        )
        WasteTreatmentDummyFixer().apply(sp)
        for e in sp.data[0]["exchanges"]:
            assert "manual_allocation" not in (e.get("properties") or {})

    def test_skips_co_product_productions(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="production",
                            name="Co-product",
                            functional=False,
                            amount=2.0,
                            comment="real co-product, not a dummy",
                        ),
                    ],
                )
            ]
        )
        result = WasteTreatmentDummyFixer().apply(sp)
        prod = sp.data[0]["exchanges"][0]
        assert prod["amount"] == 2.0
        assert prod["functional"] is False
        assert result == {"dummies_promoted": 0}

    def test_skips_already_functional_production(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="production",
                            name="Real",
                            functional=True,
                            amount=3.0,
                            comment=_DUMMY_COMMENT,
                        ),
                    ],
                )
            ]
        )
        result = WasteTreatmentDummyFixer().apply(sp)
        prod = sp.data[0]["exchanges"][0]
        assert prod["amount"] == 3.0
        assert result == {"dummies_promoted": 0}


class TestOrphanActivityPurger:
    def test_keeps_activities_with_production(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Healthy",
                    exchanges=[
                        make_exchange(type="production", name="Healthy", functional=True),
                    ],
                ),
            ]
        )
        result = OrphanActivityPurger().apply(sp)
        assert len(sp.data) == 1
        assert result == {"orphans_dropped": 0}

    def test_keeps_waste_treatment_with_functional_technosphere_edge(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "WasteTreat",
                    exchanges=[
                        make_exchange(type="technosphere", name="waste-X", functional=True),
                        make_exchange(type="technosphere", name="electricity", functional=False),
                    ],
                ),
            ]
        )
        result = OrphanActivityPurger().apply(sp)
        assert [d["name"] for d in sp.data] == ["WasteTreat"]
        assert result == {"orphans_dropped": 0}

    def test_drops_activity_with_no_functional_output(self):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Healthy",
                    exchanges=[make_exchange(type="production", name="Healthy", functional=True)],
                ),
                make_dataset(
                    "Orphan",
                    exchanges=[
                        make_exchange(type="technosphere", name="input-1", functional=False),
                        make_exchange(type="biosphere", name="bio-1"),
                    ],
                ),
            ]
        )
        result = OrphanActivityPurger().apply(sp)
        assert [d["name"] for d in sp.data] == ["Healthy"]
        assert result == {"orphans_dropped": 1}

    def test_keeps_zero_exchange_product_nodes(self):
        """``bw_simapro_csv`` materialises every SimaPro product as a
        zero-exchange ``type='product'`` ActivityDataset. They are vertices
        consumed via process exchanges' ``input`` tuples — dropping them
        leaves dangling references that fail in ``Database.process``."""
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Wastewater, average {CH}| treatment of wastewater",
                    type="product",
                    exchanges=[],
                ),
                make_dataset(
                    "[Dummy] Algae {FR} U",
                    type="product",
                    exchanges=[],
                ),
                make_dataset(
                    "Real-orphan-process",
                    type="process",
                    exchanges=[
                        make_exchange(type="technosphere", name="x", functional=False),
                    ],
                ),
            ]
        )
        result = OrphanActivityPurger().apply(sp)
        assert [d["name"] for d in sp.data] == [
            "Wastewater, average {CH}| treatment of wastewater",
            "[Dummy] Algae {FR} U",
        ]
        assert result == {"orphans_dropped": 1}

    def test_writes_diagnostic_parquet_when_settings_provided(self, tmp_path, settings):
        """Drops should be written to dashboard/orphan_activities.parquet
        with enough context for a reviewer to triage rescue vs purge."""
        import pandas as pd

        # Re-route the dashboard dir into a temp path so the test is hermetic.
        settings.paths.dashboard.mkdir(parents=True, exist_ok=True)

        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Orphan-A",
                    exchanges=[
                        make_exchange(type="technosphere", name="electricity", functional=False),
                        make_exchange(type="technosphere", name="water", functional=False),
                        make_exchange(type="biosphere", name="CO2"),
                    ],
                    code="orphan-a",
                    unit="kg",
                    location="FR",
                ),
            ]
        )
        result = OrphanActivityPurger(settings=settings).apply(sp)
        assert result == {"orphans_dropped": 1}

        df = pd.read_parquet(settings.paths.dashboard_orphan_activities)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["name"] == "Orphan-A"
        assert row["code"] == "orphan-a"
        assert row["unit"] == "kg"
        assert row["location"] == "FR"
        assert row["n_exchanges"] == 3
        # edge_counts and sample_inputs are stringified (parquet/pyarrow safety).
        assert "technosphere" in row["edge_counts"]
        assert "biosphere" in row["edge_counts"]
        assert "electricity" in row["sample_inputs"]

    def test_no_diagnostic_when_no_orphans(self, settings):
        """Don't write an empty parquet — the absence of the file is
        itself the signal that nothing was purged."""
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Healthy",
                    exchanges=[make_exchange(type="production", name="Healthy", functional=True)],
                ),
            ]
        )
        OrphanActivityPurger(settings=settings).apply(sp)
        assert not settings.paths.dashboard_orphan_activities.exists()


def _ei_ref(**kw):
    from matching.ecoinvent_catalog import EcoinventActivityRef

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


def _orphan_catalog(refs):
    """Build an ``EcoinventCatalog`` from a list of ``EcoinventActivityRef``."""
    from matching.ecoinvent_catalog import EcoinventCatalog

    db_names = tuple(sorted({r.db for r in refs})) if refs else ()
    return EcoinventCatalog(db_names=db_names, activities=tuple(refs))


def _make_relinker(settings, refs):
    return OrphanProductRelinker(settings=settings, catalog=_orphan_catalog(refs))


class TestOrphanProductRelinker:
    def test_no_orphans_short_circuits(self, settings):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Healthy",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="production",
                            name="Healthy",
                            functional=True,
                            input=("agribalyse-3.2", "p1"),
                        ),
                    ],
                ),
                make_dataset("Healthy product", type="product", code="p1", exchanges=[]),
            ]
        )
        result = _make_relinker(settings, []).apply(sp)
        assert result == {"orphan_products": 0, "relinked": 0, "dropped": 0}
        assert len(sp.data) == 2

    def test_relinks_consumer_to_ecoinvent(self, settings):
        refs = [
            _ei_ref(
                code="ei-tap-water-fr",
                name="market for tap water",
                location="FR",
                unit="kilogram",
                reference_product="tap water",
            )
        ]
        orphan_name = "Tap water {FR}| market for tap water | Cut-off, U - Adapted from Ecoinvent U"
        sp = FakeSimaProImporter(
            data=[
                make_dataset(orphan_name, type="product", code="orphan-1", exchanges=[]),
                make_dataset(
                    "consumer",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name=orphan_name,
                            unit="cubic meter",
                            input=("agribalyse-3.2", "orphan-1"),
                        ),
                    ],
                ),
            ]
        )
        result = _make_relinker(settings, refs).apply(sp)
        assert result["orphan_products"] == 1
        assert result["relinked"] == 1
        assert result["dropped"] == 0
        # Orphan product gone from sp.data.
        assert all(d.get("type") != "product" for d in sp.data)
        # Consumer rewritten to ecoinvent.
        exc = sp.data[0]["exchanges"][0]
        assert exc["input"] == ("ecoinvent-3.9.1-cutoff", "ei-tap-water-fr")
        assert exc["name"] == "market for tap water"
        assert exc["location"] == "FR"
        assert exc["unit"] == "kilogram"
        assert exc["reference product"] == "tap water"

    def test_drops_unmappable_orphan_consumer(self, settings):
        orphan_name = "Mystery {ZZ}| unknown activity | Cut-off, U - Made up"
        sp = FakeSimaProImporter(
            data=[
                make_dataset(orphan_name, type="product", code="o-mystery", exchanges=[]),
                make_dataset(
                    "consumer",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name=orphan_name,
                            input=("agribalyse-3.2", "o-mystery"),
                        ),
                    ],
                ),
            ]
        )
        result = _make_relinker(settings, []).apply(sp)
        assert result == {
            "orphan_products": 1,
            "relinked": 0,
            "dropped": 1,
            "unmapped": 1,
            "agb_sibling_hits": 0,
        }
        # Consumer still exists but its input is gone — drop_unlinked will remove it later.
        exc = sp.data[0]["exchanges"][0]
        assert "input" not in exc
        # Orphan product removed.
        assert all(d.get("type") != "product" for d in sp.data)

    def test_falls_back_to_row_then_glo_for_locations(self, settings):
        refs = [
            _ei_ref(
                code="ei-row",
                name="calendering, rigid sheets",
                location="RoW",
                unit="kilogram",
                reference_product="calendering, rigid sheets",
            )
        ]
        # Source location is FR but only RoW exists in ecoinvent — must still link.
        orphan_name = (
            "Calendering, rigid sheets {FR}| calendering, rigid sheets | Cut-off, S - Adapted"
        )
        sp = FakeSimaProImporter(
            data=[
                make_dataset(orphan_name, type="product", code="orph-cal", exchanges=[]),
                make_dataset(
                    "consumer",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name=orphan_name,
                            input=("agribalyse-3.2", "orph-cal"),
                        ),
                    ],
                ),
            ]
        )
        result = _make_relinker(settings, refs).apply(sp)
        assert result["relinked"] == 1
        assert result["dropped"] == 0
        assert sp.data[0]["exchanges"][0]["location"] == "RoW"

    def test_unparseable_simapro_name_dropped(self, settings):
        # No `{LOC}|` shape → can't parse → unmappable → consumer dropped.
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Disposal, cardboard plain string", type="product", code="o-x", exchanges=[]
                ),
                make_dataset(
                    "consumer",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="Disposal, cardboard plain string",
                            input=("agribalyse-3.2", "o-x"),
                        ),
                    ],
                ),
            ]
        )
        result = _make_relinker(settings, []).apply(sp)
        assert result["dropped"] == 1
        assert "input" not in sp.data[0]["exchanges"][0]

    def test_multiple_consumers_of_same_orphan_all_rewritten(self, settings):
        refs = [
            _ei_ref(
                code="ei-sheet-rer",
                name="sheet rolling, steel",
                location="RER",
                unit="kilogram",
                reference_product="sheet rolling, steel",
            )
        ]
        orphan_name = "Sheet rolling, steel {RER}| sheet rolling, steel | Cut-off, S - Adapted from"
        sp = FakeSimaProImporter(
            data=[
                make_dataset(orphan_name, type="product", code="o-sheet", exchanges=[]),
                make_dataset(
                    "consumer-A",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name=orphan_name,
                            input=("agribalyse-3.2", "o-sheet"),
                        ),
                    ],
                ),
                make_dataset(
                    "consumer-B",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name=orphan_name,
                            input=("agribalyse-3.2", "o-sheet"),
                        ),
                    ],
                ),
            ]
        )
        result = _make_relinker(settings, refs).apply(sp)
        assert result["relinked"] == 2
        for ds in sp.data:
            for e in ds.get("exchanges", []):
                assert e["input"] == ("ecoinvent-3.9.1-cutoff", "ei-sheet-rer")

    def test_matches_display_then_suffix_inverted_form(self, settings):
        """SimaPro sometimes encodes ecoinvent name as ``<display> <suffix>``
        instead of ``<suffix> <display>`` — e.g. ``Soybean {CA-QC}| production``
        → ecoinvent ``soybean production``."""
        refs = [
            _ei_ref(
                code="ei-soy-caqc",
                name="soybean production",
                location="CA-QC",
                unit="kilogram",
                reference_product="soybean",
            )
        ]
        orphan_name = "Soybean {CA-QC}| production | Cut-off, U - Adapted from Ecoinvent U"
        sp = FakeSimaProImporter(
            data=[
                make_dataset(orphan_name, type="product", code="o-soy", exchanges=[]),
                make_dataset(
                    "consumer",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name=orphan_name,
                            input=("agribalyse-3.2", "o-soy"),
                        ),
                    ],
                ),
            ]
        )
        result = _make_relinker(settings, refs).apply(sp)
        assert result["relinked"] == 1
        assert result["dropped"] == 0
        exc = sp.data[0]["exchanges"][0]
        assert exc["name"] == "soybean production"
        assert exc["location"] == "CA-QC"

    def test_does_not_touch_exchanges_for_non_orphan_products(self, settings):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "Producer",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="production",
                            name="Producer",
                            functional=True,
                            input=("agribalyse-3.2", "produced-1"),
                        ),
                    ],
                ),
                make_dataset("Producer product", type="product", code="produced-1", exchanges=[]),
                make_dataset(
                    "Consumer",
                    type="process",
                    exchanges=[
                        make_exchange(
                            type="technosphere",
                            name="Producer product",
                            input=("agribalyse-3.2", "produced-1"),
                        ),
                    ],
                ),
            ]
        )
        result = _make_relinker(settings, []).apply(sp)
        assert result == {"orphan_products": 0, "relinked": 0, "dropped": 0}
        # Consumer's input on the produced product untouched.
        assert sp.data[2]["exchanges"][0]["input"] == ("agribalyse-3.2", "produced-1")


class TestEdgeLabelCorrector:
    def test_renames_exchange_names_per_registry(self, settings):
        df = pd.DataFrame(
            [
                {
                    "source_name": "old name",
                    "target_name": "new name",
                    "edge_type": "biosphere",
                    "categories": [],
                    "provenance": "test",
                }
            ]
        )
        registry = make_registry(settings, edge_label_corrections=df)
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(type="biosphere", name="old name", unit="kg"),
                        make_exchange(type="biosphere", name="something else", unit="kg"),
                    ],
                )
            ]
        )
        result = EdgeLabelCorrector(registry=registry).apply(sp)
        names = [e["name"] for e in sp.data[0]["exchanges"]]
        assert names == ["new name", "something else"]
        assert result == {"renamed": 1}

    def test_empty_dataframe_returns_zero_renamed(self, settings):
        registry = make_registry(settings)
        sp = FakeSimaProImporter(data=[make_dataset("P", exchanges=[])])
        assert EdgeLabelCorrector(registry=registry).apply(sp) == {"renamed": 0}


class TestAggregateDeleter:
    def test_empty_deletions_short_circuits(self, settings):
        registry = make_registry(settings)
        sp = FakeSimaProImporter(
            data=[make_dataset("X", type="process"), make_dataset("Y", type="product")]
        )
        result = AggregateDeleter(registry=registry).apply(sp)
        assert result["deleted"] == 0
        assert result["kept_system_processes"] == 0
        assert len(sp.data) == 2

    def test_deletes_processes_by_name_or_code(self, settings):
        df = pd.DataFrame(
            [
                {"name": "killme", "code": None, "kind": "process", "provenance": "p"},
                {"name": "", "code": "code-X", "kind": "product", "provenance": "p"},
            ]
        )
        registry = make_registry(settings, deletions=df)
        sp = FakeSimaProImporter(
            data=[
                make_dataset("killme", type="process"),
                make_dataset("survivor", type="process"),
                make_dataset("Y", type="product", code="code-X"),
                make_dataset("Z", type="product", code="other"),
            ]
        )
        result = AggregateDeleter(registry=registry).apply(sp)
        names_left = [d["name"] for d in sp.data]
        assert "killme" not in names_left
        assert "Y" not in names_left
        assert result["deleted"] == 2

    def test_keeps_paired_product_of_kept_system_process(self, settings):
        """A kept system process must carry its paired product through
        too — otherwise the production edge dangles and the
        ``DanglingEdgePruner`` strips the activity, defeating the whole
        point of the system-process safeguard.
        """
        df = pd.DataFrame(
            [
                {
                    "name": "chem factory RER",
                    "code": "AGB-FAC-001",
                    "kind": "process",
                    "provenance": "agribalyse.delete-aggregated",
                },
                {
                    "name": "Chemical factory, organics {RER}|construction",
                    "code": None,
                    "kind": "product",
                    "provenance": "agribalyse.delete-aggregated",
                },
            ]
        )
        registry = make_registry(settings, deletions=df)
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "chem factory RER",
                    type="process",
                    code="AGB-FAC-001",
                    exchanges=[
                        {
                            "type": "production",
                            "name": "Chemical factory, organics {RER}|construction",
                            "amount": 1.0,
                            "input": ("agribalyse-3.2", "chem-fac-RER-PROD"),
                        },
                        make_exchange(type="biosphere", name="Radon-222", amount=1e10),
                    ],
                ),
                make_dataset(
                    "Chemical factory, organics {RER}|construction",
                    type="product",
                    code="chem-fac-RER-PROD",
                ),
            ]
        )
        result = AggregateDeleter(registry=registry).apply(sp)
        names_left = [d["name"] for d in sp.data]
        assert "chem factory RER" in names_left
        assert "Chemical factory, organics {RER}|construction" in names_left
        assert result["kept_system_processes"] == 1
        assert result["kept_paired_products"] == 1

    def test_does_not_delete_unit_process_sibling_when_only_system_process_code_is_listed(
        self, settings
    ):
        """Regression: AGB ships twin processes with identical display names but
        different ``code``s — one is a system-process copy (e.g. ``fish canning,
        small fish RoW`` ``Cut-off, S — Copied from Ecoinvent``, AGB-S-001) with
        baked-in cradle-to-gate biosphere, the other is the proper unit-process
        adaptation (``Cut-off, U — Adapted from Ecoinvent``, AGB-U-001) that
        downstream AGB consumers (``Fish canning, in brine {FR}``, …) actually
        link to. ``deletions.parquet`` only lists the S-twin's code, but historic
        ``code OR name`` matching also flagged the U-twin via the shared name;
        the system-process safeguard then kept the S-twin while silently dropping
        the U-twin, collapsing the canned-mussel / canned-salmon chains onto
        flat-aggregated biosphere (= 13 539 % cc_luc on 10028 Moule appert.).
        Code-only matching when ``row.code`` is set keeps the U-twin alive.
        """
        df = pd.DataFrame(
            [
                {
                    "name": "fish canning, small fish RoW",
                    "code": "AGB-S-001",
                    "kind": "process",
                    "provenance": "agribalyse.delete-aggregated",
                },
            ]
        )
        registry = make_registry(settings, deletions=df)
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "fish canning, small fish RoW",
                    type="process",
                    code="AGB-S-001",
                    exchanges=[
                        make_exchange(type="biosphere", name="Radon-222", amount=1.3e10),
                    ],
                ),
                make_dataset(
                    "fish canning, small fish RoW",
                    type="process",
                    code="AGB-U-001",
                    exchanges=[
                        make_exchange(type="technosphere", name="upstream input"),
                    ],
                ),
            ]
        )
        result = AggregateDeleter(registry=registry).apply(sp)
        codes_left = [d.get("code") for d in sp.data]
        assert "AGB-S-001" in codes_left, "S-twin must survive via the system-process safeguard"
        assert "AGB-U-001" in codes_left, "U-twin must not be deleted by name-collision"
        assert result["deleted"] == 0
        assert result["kept_system_processes"] == 1

    def test_keeps_system_processes_with_baked_in_biosphere(self, settings):
        """Regression for FIX_DATA.md § 2 — ionising radiation under-score.

        AGB ships *system processes* with pre-aggregated cradle-to-gate
        biosphere edges and zero technosphere inputs (e.g. chemical
        factory construction, organics RER carries 1.3×10¹⁰ kBq Radon-222).
        ``deletions.parquet`` lists them as duplicates of ecoinvent
        namesakes, but ecoinvent's namesakes are *unit* processes whose
        upstream chains don't reproduce the same totals. Deleting the AGB
        version strips ~46 GBq of Radon-222 from the entire system.

        The deleter must skip these (zero technosphere, ≥1 biosphere) and
        report them via ``kept_system_processes``.
        """
        df = pd.DataFrame(
            [
                {
                    "name": "chemical factory construction, organics RER",
                    "code": "AGB-FAC-001",
                    "kind": "process",
                    "provenance": "agribalyse.delete-aggregated",
                },
                {
                    "name": "regular unit process",
                    "code": "AGB-UNIT-001",
                    "kind": "process",
                    "provenance": "agribalyse.delete-aggregated",
                },
            ]
        )
        registry = make_registry(settings, deletions=df)
        sp = FakeSimaProImporter(
            data=[
                # System process: bio-only, must be kept.
                make_dataset(
                    "chemical factory construction, organics RER",
                    type="process",
                    code="AGB-FAC-001",
                    exchanges=[
                        make_exchange(type="biosphere", name="Radon-222", amount=1.3e10),
                        make_exchange(type="biosphere", name="Carbon dioxide", amount=1e6),
                    ],
                ),
                # Unit process: has a tech input, must be deleted.
                make_dataset(
                    "regular unit process",
                    type="process",
                    code="AGB-UNIT-001",
                    exchanges=[
                        make_exchange(type="technosphere", name="upstream input"),
                        make_exchange(type="biosphere", name="Radon-222", amount=1.0),
                    ],
                ),
            ]
        )
        result = AggregateDeleter(registry=registry).apply(sp)
        names_left = [d["name"] for d in sp.data]
        assert "chemical factory construction, organics RER" in names_left
        assert "regular unit process" not in names_left
        assert result["deleted"] == 1
        assert result["kept_system_processes"] == 1


# ============================================================================
# Strategy-runner-driven transforms


@pytest.fixture
def runner(tmp_path):
    return StrategyRunner(suppressed_log=SuppressedStrategyLog(output_path=tmp_path / "x.parquet"))


class TestInternalAgbLinker:
    def test_calls_apply_strategies_and_match_database(self, runner):
        sp = FakeSimaProImporter()
        InternalAgbLinker(runner=runner).apply(sp)
        assert "__chain__" in sp.applied_strategies
        # Subsequent match_database call carries name+unit fields and processes_to_products.
        last_call = sp.match_calls[-1]
        _args, kwargs = last_call[0], last_call[1]
        assert kwargs.get("fields") == ["name", "unit"]
        assert kwargs.get("processes_to_products") is True


class TestBiosphereLabelNormaliser:
    def test_runs_chain_via_runner(self, runner, settings):
        # Materialise the empty migration JSONs the chain expects so
        # BiosphereStrategyChain.from_paths can succeed.
        bw2io_dir = settings.paths.bw2io_data
        bw2io_dir.mkdir(parents=True, exist_ok=True)
        (bw2io_dir / "simapro-biosphere.json").write_text("[]")
        (bw2io_dir / "biosphere-2-3-categories.json").write_text(
            json.dumps({"fields": ["name"], "data": []})
        )
        (bw2io_dir / "biosphere-2-3-names.json").write_text(
            json.dumps({"fields": ["name"], "data": []})
        )

        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        make_exchange(
                            type="biosphere",
                            name="X",
                            categories=("Air", "high. pop."),
                        )
                    ],
                )
            ]
        )
        BiosphereLabelNormaliser(runner=runner, settings=settings).apply(sp)
        # Chain mapped 'Air' → 'air', 'high. pop.' → 'urban air close to ground'
        assert sp.data[0]["exchanges"][0]["categories"] == (
            "air",
            "urban air close to ground",
        )


class TestStandardLabelNormaliser:
    def test_invokes_label_norm_and_units_normalisation(self, runner):
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[{"context": ["air"]}],
                )
            ]
        )
        StandardLabelNormaliser(runner=runner).apply(sp)
        # context → categories rewritten by the in-house strategy.
        assert sp.data[0]["exchanges"][0]["categories"] == ("air",)
        assert sp.randonneur_calls
        assert sp.randonneur_calls[0][0][0] == "generic-brightway-units-normalization"


class TestBioStrategyChain:
    def test_links_biosphere_exchange_via_catalog_by_code(self, runner):
        from domain import Bucket
        from matching.bio_catalog import BioFlowRef

        catalog = make_bio_catalog(
            [
                BioFlowRef(
                    db="bio-db",
                    code="bio-co2",
                    name="Carbon dioxide",
                    unit="kg",
                    bucket=Bucket.AIR,
                    categories=("air",),
                )
            ]
        )
        sp = FakeSimaProImporter(
            data=[
                make_dataset(
                    "P",
                    exchanges=[
                        # Source carries the same code, so the prelinker should
                        # set ``input`` directly without further fields.
                        {
                            "type": "biosphere",
                            "name": "Carbon dioxide",
                            "unit": "kg",
                            "code": "bio-co2",
                            "categories": ("air",),
                        }
                    ],
                )
            ]
        )
        stats = BioStrategyChain(runner=runner, bio_db_name="bio-db", catalog=catalog).apply(sp)
        assert stats["linked_by_code"] == 1
        assert sp.data[0]["exchanges"][0]["input"] == ("bio-db", "bio-co2")


class TestRestoreSimaproNamesTransform:
    def test_calls_restore_via_runner_and_then_unconditional_simapro_call(self, runner):
        sp = FakeSimaProImporter()
        RestoreSimaproNamesTransform(runner=runner).apply(sp)
        labels = [c[0][0] for c in sp.randonneur_calls]
        assert "agribalyse-3.1.1-restore-simapro-ecoinvent-names" in labels
        assert "simapro-ecoinvent-3.9.1-cutoff" in labels


# ============================================================================
# Biosphere flowmap applier


class TestBiosphereFlowmapApplier:
    def test_returns_zero_when_file_missing(self, settings):
        # The default Settings paths don't materialise the flowmap.
        result = BiosphereFlowmapApplier(settings=settings).apply(FakeSimaProImporter())
        assert result == {"applied": 0, "patched_nan_cfs": 0, "patched_inverted_cfs": 0}

    def test_patches_water_kg_to_m3_nan_with_density(self, settings, monkeypatch):
        # Write a flowmap with one NaN water entry.
        path = settings.paths.biosphere_flowmap_json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"some": "datapackage"}))

        # Stub randonneur.Datapackage.from_json to return an in-memory object.
        class _Dp:
            def __init__(self):
                self.data = {
                    "update": [
                        {
                            "source": {"name": "Water", "unit": "kg"},
                            "target": {"name": "Water"},
                            "conversion_factor": float("nan"),
                        }
                    ]
                }

        rn = sys.modules["randonneur"]
        monkeypatch.setattr(rn.Datapackage, "from_json", classmethod(lambda cls, _p: _Dp()))

        applied_dp = {}
        sp = FakeSimaProImporter()

        def _record_randonneur(*a, **k):
            applied_dp["dp"] = k.get("datapackage")

        sp.randonneur = _record_randonneur

        result = BiosphereFlowmapApplier(settings=settings).apply(sp)
        assert result == {"applied": 1, "patched_nan_cfs": 1, "patched_inverted_cfs": 0}
        cf = applied_dp["dp"].data["update"][0]["conversion_factor"]
        assert cf == 0.001
        assert not math.isnan(cf)

    def test_patches_manganese_55_redirects_to_mn54(self, settings, monkeypatch):
        path = settings.paths.biosphere_flowmap_json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

        class _Dp:
            def __init__(self):
                self.data = {
                    "update": [
                        {
                            "source": {"name": "Manganese-55", "unit": "Bq"},
                            "target": {"name": "Manganese-55"},
                            "conversion_factor": float("nan"),
                            "comment": "old",
                        }
                    ]
                }

        rn = sys.modules["randonneur"]
        monkeypatch.setattr(rn.Datapackage, "from_json", classmethod(lambda cls, _p: _Dp()))

        sp = FakeSimaProImporter()
        applied_dp = {}
        sp.randonneur = lambda *a, **k: applied_dp.update(dp=k.get("datapackage"))

        BiosphereFlowmapApplier(settings=settings).apply(sp)
        entry = applied_dp["dp"].data["update"][0]
        assert entry["target"]["name"] == "Manganese-54"
        assert entry["target"]["unit"] == "kBq"
        assert entry["conversion_factor"] == 0.001
        assert "patched" in entry["comment"]

    def test_unrecognised_nan_entry_raises(self, settings, monkeypatch):
        path = settings.paths.biosphere_flowmap_json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

        class _Dp:
            def __init__(self):
                self.data = {
                    "update": [
                        {
                            "source": {"name": "Unknown", "unit": "kg"},
                            "target": {"name": "x"},
                            "conversion_factor": float("nan"),
                        }
                    ]
                }

        rn = sys.modules["randonneur"]
        monkeypatch.setattr(rn.Datapackage, "from_json", classmethod(lambda cls, _p: _Dp()))

        with pytest.raises(ValueError, match="Unexpected NaN"):
            BiosphereFlowmapApplier(settings=settings).apply(FakeSimaProImporter())


# ============================================================================
# SimaProImporter (cache)


# Module-level so pickle can serialise instances. ``_PARSE_CALLS`` is the
# shared counter used by the test to verify cache reuse.
_PARSE_CALLS: list[dict] = []


class _RecordedParse:
    """Pickle-friendly fake for ``SimaProCsvParser.parse()`` results."""

    def __init__(self, data, db_name="agb"):
        self.data = data
        self.db_name = db_name


def _fake_parse(self):
    _PARSE_CALLS.append({"csv_path": self.csv_path, "database_name": self.database_name})
    return _RecordedParse(data=[{"name": "P", "exchanges": []}], db_name="agb")


class TestSimaProImporter:
    def test_cache_pickle_round_trip(self, settings, monkeypatch):
        from transforms import sp_csv_parser

        path = settings.paths.agribalyse_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake csv contents")

        _PARSE_CALLS.clear()
        monkeypatch.setattr(sp_csv_parser.SimaProCsvParser, "parse", _fake_parse)

        first = SimaProImporter(settings=settings).load()
        assert first.data == [{"name": "P", "exchanges": []}]
        assert len(_PARSE_CALLS) == 1
        assert settings.paths.importer_cache_pkl.exists()

        # Second load: pickle hit, no fresh parse.
        second = SimaProImporter(settings=settings).load()
        assert second.data == [{"name": "P", "exchanges": []}]
        assert len(_PARSE_CALLS) == 1
