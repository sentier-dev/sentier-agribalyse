"""Tests for ``utils.unmatched_diagnostic``.

The diagnostic re-walks ``BiosphereMatcher`` candidate logic and labels
each input flow with a status. We verify each status path with hand-built
registries and a synthetic catalog so the assertions don't depend on the
live Brightway sqlite.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

from domain import Bucket, Tier
from matching.bio_catalog import BioFlowRef, BiosphereCatalog
from registry.indexes import _IndexEntry  # noqa: F401 — sanity import
from tests.fixtures.builders import (  # type: ignore[import-not-found]
    make_mappings_biosphere_df,
    make_registry,
    make_settings,
    make_unit_conversions_df,
)
from utils.unmatched_diagnostic import (
    UnlinkedXlsxLoader,
    UnmatchedFlowDiagnostic,
    UnmatchedInputRow,
)


def _catalog(
    *flows: BioFlowRef, db_names: tuple[str, ...] = ("ecoinvent-3.9.1-biosphere", "ef")
) -> BiosphereCatalog:
    return BiosphereCatalog(db_names=db_names, flows=flows)


class TestUnmatchedFlowDiagnostic:
    def test_no_registry_when_index_lacks_entry(self, tmp_path: Path):
        settings = make_settings(tmp_path)
        registry = make_registry(settings)
        diag = UnmatchedFlowDiagnostic(settings=settings, registry=registry, catalog=_catalog())
        outcome = diag.diagnose(
            UnmatchedInputRow(name="Mystery", unit="kilogram", top_cat="air", sub_cat=None)
        )
        assert outcome.status == "no_registry"
        assert outcome.candidates == ()

    def test_unit_mismatch_when_no_conversion_registered(self, tmp_path: Path):
        settings = make_settings(tmp_path)
        registry = make_registry(
            settings,
            mappings_biosphere=make_mappings_biosphere_df(
                [
                    {
                        "source_name": "Cesium-134",
                        "source_top_bucket": Bucket.SOIL,
                        "source_unit": "Becquerel",
                        "target_db": "",
                        "target_code": "",
                        "target_name": "Cesium-134",
                        "target_unit": "",
                        "priority_tier": int(Tier.EF_PLACEHOLDER),
                    }
                ]
            ),
        )
        catalog = _catalog(
            BioFlowRef(
                db="ecoinvent-3.9.1-biosphere",
                code="abc",
                name="Cesium-134",
                unit="kilo Becquerel",
                bucket=Bucket.SOIL,  # same bucket — target is found, unit conversion absent
            )
        )
        diag = UnmatchedFlowDiagnostic(settings=settings, registry=registry, catalog=catalog)
        outcome = diag.diagnose(
            UnmatchedInputRow(name="Cesium-134", unit="Becquerel", top_cat="soil", sub_cat=None)
        )
        assert outcome.status == "unit_mismatch_only"
        assert outcome.matched is False

    def test_cross_emission_compartment_not_matched(self, tmp_path: Path):
        """Cross-emission-compartment fallback is disabled: Cesium-134 (soil)
        must NOT resolve to the air-bucket target — compartment mismatches are
        discarded per reviewer policy."""
        settings = make_settings(tmp_path)
        registry = make_registry(
            settings,
            mappings_biosphere=make_mappings_biosphere_df(
                [
                    {
                        "source_name": "Cesium-134",
                        "source_top_bucket": Bucket.SOIL,
                        "source_unit": "Becquerel",
                        "target_db": "",
                        "target_code": "",
                        "target_name": "Cesium-134",
                        "target_unit": "",
                        "priority_tier": int(Tier.EF_PLACEHOLDER),
                    }
                ]
            ),
            unit_conversions=make_unit_conversions_df(
                [
                    {
                        "source_unit": "Becquerel",
                        "target_unit": "kilo Becquerel",
                        "multiplier": 0.001,
                    }
                ]
            ),
        )
        catalog = _catalog(
            BioFlowRef(
                db="ecoinvent-3.9.1-biosphere",
                code="ec-bio-cs134",
                name="Cesium-134",
                unit="kilo Becquerel",
                bucket=Bucket.AIR,
            )
        )
        diag = UnmatchedFlowDiagnostic(settings=settings, registry=registry, catalog=catalog)
        outcome = diag.diagnose(
            UnmatchedInputRow(name="Cesium-134", unit="Becquerel", top_cat="soil", sub_cat=None)
        )
        assert outcome.matched is False

    def test_target_unresolved_when_target_name_absent_from_catalog(self, tmp_path: Path):
        settings = make_settings(tmp_path)
        registry = make_registry(
            settings,
            mappings_biosphere=make_mappings_biosphere_df(
                [
                    {
                        "source_name": "Asbestos",
                        "source_top_bucket": Bucket.AIR,
                        "target_db": "",
                        "target_code": "",
                        "target_name": "Chrysotile",
                        "target_unit": "",
                        "priority_tier": int(Tier.LLM_OVERRIDES),
                    }
                ]
            ),
        )
        diag = UnmatchedFlowDiagnostic(settings=settings, registry=registry, catalog=_catalog())
        outcome = diag.diagnose(
            UnmatchedInputRow(name="Asbestos", unit="kilogram", top_cat="air", sub_cat=None)
        )
        assert outcome.status == "target_unresolved_only"

    def test_unmatchable_candidate_halts_walk(self, tmp_path: Path):
        settings = make_settings(tmp_path)
        registry = make_registry(
            settings,
            mappings_biosphere=make_mappings_biosphere_df(
                [
                    {
                        "source_name": "Air",
                        "source_top_bucket": Bucket.RESOURCE,
                        "is_unmatchable": True,
                        "priority_tier": int(Tier.UNMATCHABLE),
                    }
                ]
            ),
        )
        diag = UnmatchedFlowDiagnostic(settings=settings, registry=registry, catalog=_catalog())
        outcome = diag.diagnose(
            UnmatchedInputRow(
                name="Air", unit="kilogram", top_cat="natural resource", sub_cat="in air"
            )
        )
        assert outcome.matched is False
        assert outcome.candidates[0].outcome == "halt_unmatchable"


class TestUnmatchedDiagnosticReport:
    def test_status_counts_aggregates_outcomes(self, tmp_path: Path):
        settings = make_settings(tmp_path)
        registry = make_registry(settings)
        diag = UnmatchedFlowDiagnostic(settings=settings, registry=registry, catalog=_catalog())
        rows = [
            UnmatchedInputRow(name="A", unit="kg", top_cat="air", sub_cat=None),
            UnmatchedInputRow(name="B", unit="kg", top_cat="water", sub_cat=None),
            UnmatchedInputRow(name="C", unit="kg", top_cat="soil", sub_cat=None),
        ]
        report = diag.diagnose_many(rows)
        assert report.status_counts() == {"no_registry": 3}

    def test_render_text_describes_flows(self, tmp_path: Path):
        settings = make_settings(tmp_path)
        registry = make_registry(settings)
        diag = UnmatchedFlowDiagnostic(settings=settings, registry=registry, catalog=_catalog())
        rows = [UnmatchedInputRow(name="X", unit="kg", top_cat="air", sub_cat=None)]
        report = diag.diagnose_many(rows)
        text = report.render_text()
        assert "X" in text
        assert "NO REGISTRY CANDIDATES" in text


class TestUnlinkedXlsxLoader:
    def test_skips_header_and_empty_rows(self, tmp_path: Path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["source_name", "source_unit", "source_top_cat", "source_sub_cat"])
        ws.append(["Methane", "kilogram", "natural resource", None])
        ws.append([None, None, None, None])
        ws.append(["", "", "", ""])
        path = tmp_path / "unlinked.xlsx"
        wb.save(path)

        loader = UnlinkedXlsxLoader(path=path)
        rows = loader.read()
        assert len(rows) == 1
        assert rows[0].name == "Methane"
        assert rows[0].bucket == Bucket.RESOURCE
