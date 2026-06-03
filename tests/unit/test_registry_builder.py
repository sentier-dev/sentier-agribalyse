"""Unit tests for ``RegistryBuilder``.

The builder wires every source ingester to a Settings-derived path and writes
parquet files. We test the orchestration shape (dedupe, idempotency, meta
JSON) by replacing ``_build_sources`` with a tiny synthetic bundle so we
don't need real on-disk source data.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from domain import (
    Bucket,
    ContextNorm,
    Deletion,
    EdgeLabelCorrection,
    EfTargetIndexRow,
    Mapping,
    SourceKind,
    Tier,
    UnitAlias,
    UnitConversion,
)
from registry import RegistryBuilder
from registry.builder import _SourceBundle


class _FakeSource:
    """Stand-in that returns a fixed list of rows."""

    def __init__(self, rows):
        self._rows = rows

    def read(self):
        return list(self._rows)


def _bundle_with(
    biosphere=(),
    technosphere=(),
    unmatchable=(),
    unit_conversions=(),
    unit_aliases=(),
    context=(),
    deletions=(),
    edge_labels=(),
    ef_targets=(),
):
    return _SourceBundle(
        biosphere_mapping_sources=(_FakeSource(biosphere),),
        technosphere_mapping_sources=(_FakeSource(technosphere),),
        unmatchable_sources=(_FakeSource(unmatchable),),
        unit_conversion_sources=(_FakeSource(unit_conversions),),
        unit_alias_sources=(_FakeSource(unit_aliases),),
        context_sources=(_FakeSource(context),),
        deletion_sources=(_FakeSource(deletions),),
        edge_label_sources=(_FakeSource(edge_labels),),
        ef_target_sources=(_FakeSource(ef_targets),),
    )


class TestRegistryBuilderBuild:
    @pytest.fixture
    def builder_with_tiny_bundle(self, settings, monkeypatch):
        """Inject a tiny synthetic bundle so the builder doesn't touch real source files."""
        biosphere_rows = [
            Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name="CO2",
                source_unit="kg",
                source_top_bucket=Bucket.AIR,
                target_db="biosphere3",
                target_code="abc",
                priority_tier=Tier.CURATED_TARGETED,
                provenance="tier-1",
            ),
            Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name="CO2",
                source_unit="kg",
                source_top_bucket=Bucket.AIR,
                target_db="biosphere3",
                target_code="abc",
                priority_tier=Tier.CURATED_TARGETED,
                provenance="duplicate",  # same dedupe key as the row above
            ),
            Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name="Methane",
                source_unit="kg",
                source_top_bucket=Bucket.AIR,
                target_db="ef",
                target_code="ef-uuid",
                priority_tier=Tier.HARMONISED_FLOWS,
                provenance="tier-4",
            ),
            # Tech-tagged rows in the biosphere source list should not flow into the parquet.
            Mapping(
                source_kind=SourceKind.AGB_NODE,
                source_name="some node",
                source_unit="kg",
                priority_tier=Tier.CURATED_TARGETED,
            ),
        ]
        unmatchable_rows = [
            Mapping(
                source_kind=SourceKind.AGB_FLOW,
                source_name="Forever",
                source_unit="kg",
                source_top_bucket=Bucket.AIR,
                priority_tier=Tier.UNMATCHABLE,
                is_unmatchable=True,
            )
        ]
        bundle = _bundle_with(
            biosphere=biosphere_rows,
            unmatchable=unmatchable_rows,
            unit_conversions=[
                UnitConversion(source_unit="m", target_unit="km", multiplier=0.001),
                UnitConversion(source_unit="m", target_unit="km", multiplier=0.001),  # dedupe
            ],
            unit_aliases=[
                UnitAlias(alias="A", canonical="year"),
                UnitAlias(alias="a", canonical="year"),  # case-insensitive dedupe
            ],
            context=[ContextNorm(source_context=("Emissions to air",), target_context=("air",))],
            deletions=[Deletion(name="X", code="c", kind="process")],
            edge_labels=[EdgeLabelCorrection(source_name="old", target_name="new", edge_type="t")],
            ef_targets=[EfTargetIndexRow(code="u1", name="CO2", unit="kg", bucket=Bucket.AIR)],
        )
        # Builder is frozen; patch the class method instead of the instance.
        monkeypatch.setattr(RegistryBuilder, "_build_sources", lambda self: bundle)
        return RegistryBuilder(settings)

    def test_build_writes_all_registry_parquets(self, builder_with_tiny_bundle, settings):
        out = builder_with_tiny_bundle.build()
        for path in (
            settings.paths.registry_mappings_biosphere,
            settings.paths.registry_mappings_technosphere,
            settings.paths.registry_unmatchable,
            settings.paths.registry_unit_conversions,
            settings.paths.registry_unit_aliases,
            settings.paths.registry_context_normalisation,
            settings.paths.registry_deletions,
            settings.paths.registry_edge_label_corrections,
            settings.paths.registry_target_index_ef,
            settings.paths.registry_meta,
        ):
            assert path.exists(), path

        assert "row_counts" in out
        assert out["row_counts"]["mappings_biosphere"] >= 1

    def test_build_dedupes_biosphere_rows(self, builder_with_tiny_bundle, settings):
        builder_with_tiny_bundle.build()
        df = pd.read_parquet(settings.paths.registry_mappings_biosphere)
        # Despite three input rows for CO2 (one duplicate, one tech-only) we get
        # exactly two: CO2 + Methane. Tech rows must not leak into the biosphere parquet.
        assert df["source_name"].tolist().count("CO2") == 1
        assert "some node" not in df["source_name"].tolist()

    def test_build_dedupes_unit_conversions_and_aliases(self, builder_with_tiny_bundle, settings):
        builder_with_tiny_bundle.build()
        conv = pd.read_parquet(settings.paths.registry_unit_conversions)
        aliases = pd.read_parquet(settings.paths.registry_unit_aliases)
        assert len(conv) == 1
        assert len(aliases) == 1

    def test_build_writes_meta_json_with_row_counts(self, builder_with_tiny_bundle, settings):
        builder_with_tiny_bundle.build()
        meta = json.loads(settings.paths.registry_meta.read_text())
        assert "row_counts" in meta
        assert "tiers" in meta
        assert meta["agribalyse_version"] == settings.agribalyse_version
        assert "built_at" in meta

    def test_build_is_idempotent(self, builder_with_tiny_bundle, settings):
        first = builder_with_tiny_bundle.build()
        second = builder_with_tiny_bundle.build()
        # Row counts must match across re-runs.
        assert first["row_counts"] == second["row_counts"]
