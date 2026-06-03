"""Integration test: a slice of the linking pipeline on tiny synthetic data.

Wires together real instances of: ``MappingRegistry`` →
``BiosphereMatcher`` → ``AuditLog`` → ``DropTallyTracker`` →
``CoverageReporter`` → ``UnlinkedExporter`` → ``RunReport``.

We avoid the bw2 layer (Brightway database write, matrix purge, ecoinvent
load) — those are exercised by their own unit tests. The point here is to
prove the OOP composition produces a coherent run report from end to end.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from domain import Bucket, Tier
from matching.audit import AuditLog, DropTallyTracker
from matching.bio_catalog import BioFlowRef, BiosphereCatalog
from matching.biosphere import BiosphereMatcher
from reporting import CoverageReporter, RunReport, UnlinkedExporter
from tests.fixtures.builders import (
    make_dataset,
    make_exchange,
    make_mappings_biosphere_df,
    make_registry,
    make_unit_conversions_df,
    make_unmatchable_df,
)


@pytest.fixture
def tiny_registry(settings):
    """A registry with a curated CO2 mapping, an unmatchable flow, and a unit conversion."""
    return make_registry(
        settings,
        mappings_biosphere=make_mappings_biosphere_df(
            [
                {
                    "source_name": "Carbon dioxide",
                    "source_top_bucket": Bucket.AIR,
                    "target_db": "biosphere3",
                    "target_code": "uuid-co2",
                    "target_unit": "kg",
                    "priority_tier": Tier.CURATED_TARGETED,
                    "provenance": "tier-1",
                },
                {
                    "source_name": "Water",
                    "source_top_bucket": Bucket.WATER,
                    "target_db": "biosphere3",
                    "target_code": "uuid-water-m3",
                    "target_unit": "m3",
                    "priority_tier": Tier.CURATED_TARGETED,
                    "provenance": "tier-1",
                },
            ]
        ),
        unmatchable=make_unmatchable_df(
            [
                {
                    "source_name": "Forever",
                    "source_top_bucket": Bucket.AIR,
                    "is_unmatchable": True,
                }
            ]
        ),
        unit_conversions=make_unit_conversions_df(
            [{"source_unit": "kg", "target_unit": "m3", "multiplier": 0.001}]
        ),
    )


@pytest.fixture
def tiny_catalog():
    return BiosphereCatalog(
        db_names=("biosphere3",),
        flows=(
            BioFlowRef(
                db="biosphere3",
                code="uuid-co2",
                name="Carbon dioxide",
                unit="kg",
                bucket=Bucket.AIR,
            ),
            BioFlowRef(
                db="biosphere3",
                code="uuid-water-m3",
                name="Water",
                unit="m3",
                bucket=Bucket.WATER,
            ),
        ),
    )


@pytest.fixture
def tiny_sp_data():
    """Three processes with a mix of linked, unmatched, unit-converted, and unmatchable flows."""
    return [
        make_dataset(
            "Wheat production",
            exchanges=[
                make_exchange(
                    type="biosphere",
                    name="Carbon dioxide",
                    unit="kg",
                    amount=4.2,
                    categories=("air",),
                ),
                make_exchange(
                    type="biosphere",
                    name="Water",
                    unit="kg",
                    amount=1000.0,
                    categories=("water", "river"),
                ),
                make_exchange(
                    type="biosphere",
                    name="UnknownFlow",
                    unit="kg",
                    amount=0.5,
                    categories=("air",),
                ),
                make_exchange(
                    type="biosphere",
                    name="Forever",
                    unit="kg",
                    amount=1.0,
                    categories=("air",),
                ),
                make_exchange(
                    type="technosphere",
                    name="Tractor diesel",
                    unit="kg",
                    amount=2.0,
                ),
            ],
        ),
    ]


@pytest.mark.integration
class TestLinkingSliceEndToEnd:
    def test_full_slice_produces_coherent_report(
        self, settings, tiny_registry, tiny_catalog, tiny_sp_data, tmp_path
    ):
        # 1. Real audit + drop trackers.
        audit = AuditLog(output_path=tmp_path / "audit.parquet")
        drops = DropTallyTracker()

        # 2. Pre-match coverage.
        report = RunReport()
        report.add_coverage(CoverageReporter.from_sp_data("pre_match", tiny_sp_data))

        # 3. Real BiosphereMatcher.
        bio_stats = BiosphereMatcher(
            settings=settings,
            registry=tiny_registry,
            catalog=tiny_catalog,
            audit=audit,
            drops=drops,
        ).match(tiny_sp_data)

        report.add_stage(
            "biosphere_matched",
            {
                "n_total": bio_stats.n_total,
                "n_linked": bio_stats.n_linked,
                "n_unmatchable_recognised": bio_stats.n_unmatchable_recognised,
            },
        )

        # 4. Post-match coverage + unlinked export.
        report.add_coverage(CoverageReporter.from_sp_data("post_match", tiny_sp_data))
        unlinked_stats = UnlinkedExporter(settings=settings).export(tiny_sp_data)
        report.add_stage("unlinked_export", unlinked_stats)

        # 5. Aggregate audit + drops into the report.
        report.set_drops(drops.totals_by_strategy())

        # 6. Persist artifacts.
        report_path = report.write(settings.paths.dashboard_run_report)
        audit.write()

        # ===== ASSERTIONS =====

        # CO2 + Water (with unit conversion) linked. UnknownFlow stayed unlinked.
        # Forever was recognised as unmatchable. Stats reflect that.
        assert bio_stats.n_total == 4
        assert bio_stats.n_linked == 2
        assert bio_stats.n_unmatchable_recognised == 1

        co2 = next(e for e in tiny_sp_data[0]["exchanges"] if e.get("name") == "Carbon dioxide")
        assert co2["input"] == ("biosphere3", "uuid-co2")

        water = next(e for e in tiny_sp_data[0]["exchanges"] if e.get("name") == "Water")
        assert water["input"] == ("biosphere3", "uuid-water-m3")
        assert water["unit"] == "m3"  # rescaled
        assert water["amount"] == pytest.approx(1.0)  # 1000 kg → 1 m3

        # Audit log has at least two new-link entries.
        assert len([e for e in audit.entries if e.kind.value == "new_link"]) == 2
        audit_path = audit.output_path
        assert audit_path.exists()
        df = pd.read_parquet(audit_path)
        assert len(df) >= 2

        # Run report file is well-formed JSON with all sections populated.
        data = json.loads(report_path.read_text())
        assert "stages" in data
        assert data["stages"]["biosphere_matched"]["n_linked"] == 2
        assert any(c["label"] == "pre_match" for c in data["coverage"])
        assert any(c["label"] == "post_match" for c in data["coverage"])

        # Unlinked export wrote both files.
        assert (settings.paths.unlinked / "technosphere_unlinked.json").exists()
        # Bio xlsx is only written if there are unlinked bio flows.
        assert (settings.paths.unlinked / "biosphere_unlinked.xlsx").exists()


@pytest.mark.integration
class TestLinkingNoLlmGate:
    def test_llm_tier_skipped_under_no_llm_setting(self, settings, tmp_path):
        """End-to-end gate: LLM tier rows should be ignored under
        ``Settings.with_no_llm()`` even when they're the only candidate."""
        s = settings.with_no_llm()

        registry = make_registry(
            s,
            mappings_biosphere=make_mappings_biosphere_df(
                [
                    {
                        "source_name": "Borderline",
                        "source_top_bucket": Bucket.AIR,
                        "target_db": "biosphere3",
                        "target_code": "uuid-llm",
                        "target_unit": "kg",
                        "priority_tier": Tier.LLM_OVERRIDES,
                    }
                ]
            ),
        )
        catalog = BiosphereCatalog(
            db_names=("biosphere3",),
            flows=(
                BioFlowRef(
                    db="biosphere3",
                    code="uuid-llm",
                    name="Borderline",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
            ),
        )
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere", name="Borderline", unit="kg", categories=("air",)
                    )
                ],
            )
        ]
        BiosphereMatcher(
            settings=s,
            registry=registry,
            catalog=catalog,
            audit=AuditLog(output_path=tmp_path / "a.parquet"),
            drops=DropTallyTracker(),
        ).match(sp_data)
        # No link: LLM gate symmetrically blocked tier 10.
        assert "input" not in sp_data[0]["exchanges"][0]
