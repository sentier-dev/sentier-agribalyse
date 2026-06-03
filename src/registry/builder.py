"""``RegistryBuilder``: orchestrate every source ingester into ``registry/*.parquet``.

Each tier's mappings are deduplicated by ``(source_kind, source_name_lower,
source_top_bucket, source_unit, target_db, target_code, target_name,
priority_tier)`` so re-runs are idempotent and the harmonised-flows altLabel
explosion (≈1.6M rows) shrinks before write.

Sources are constructed once in ``_build_sources()`` so the builder owns
its dependency graph; callers pass only ``Settings``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from config import Settings
from core import Logging, ParquetCache, StepTimer
from domain import (
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
from readers import RandonneurDataLoader, XlsxReader
from registry.schema import (
    ContextNormTable,
    DeletionTable,
    EdgeLabelTable,
    EfTargetIndexTable,
    MappingTable,
    UnitAliasTable,
    UnitConversionTable,
)
from registry.sources import (
    AGB32FlowmapperBiosphereSource,
    AgbDeleteAggregatedSource,
    AgbEdgeLabelsSource,
    CuratedOverridesSource,
    EfCfTargetIndexSource,
    ExtraUnitConversionSource,
    HarmonisedFlowsSource,
    LlmReviewedSource,
    PlaceholderEcoinventSource,
    PlaceholderEfNativeSource,
    PlaceholderTransitiveSource,
    PlaceholderUnmatchableSource,
    RandonneurAgribalyseBiosphereSource,
    RandonneurSimaproBiosphereSource,
    RandonneurSimaproContextSource,
    RandonneurUnitAliasSource,
    RandonneurUnitConversionSource,
    RandonneurUnitNormalisationSource,
    RandonneurUnlinkedListSource,
    RandonneurWaterSlashM3Source,
)


@dataclass(frozen=True)
class _SourceBundle:
    """All concrete source instances bound to one ``Settings``."""

    biosphere_mapping_sources: tuple[Any, ...]
    technosphere_mapping_sources: tuple[Any, ...]
    unmatchable_sources: tuple[Any, ...]
    unit_conversion_sources: tuple[Any, ...]
    unit_alias_sources: tuple[Any, ...]
    context_sources: tuple[Any, ...]
    deletion_sources: tuple[Any, ...]
    edge_label_sources: tuple[Any, ...]
    ef_target_sources: tuple[Any, ...]


@dataclass(frozen=True)
class RegistryBuilder:
    """Build ``registry/*.parquet`` from every authoritative source."""

    settings: Settings

    @property
    def _log(self):
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Public entrypoint.

    def build(self) -> dict[str, Any]:
        """Materialise every parquet + write registry.meta.json. Idempotent."""
        self.settings.paths.ensure_runtime_dirs()
        bundle = self._build_sources()
        log = self._log

        with StepTimer(log, "registry.build"):
            row_counts = {
                "mappings_biosphere": self._write_biosphere(bundle),
                "mappings_technosphere": self._write_technosphere(bundle),
                "unmatchable": self._write_unmatchable(bundle),
                "unit_conversions": self._write_unit_conversions(bundle),
                "unit_aliases": self._write_unit_aliases(bundle),
                "context_normalisation": self._write_context_norm(bundle),
                "deletions": self._write_deletions(bundle),
                "edge_label_corrections": self._write_edge_labels(bundle),
                "target_index_ef": self._write_ef_target_index(bundle),
            }
            meta = self._write_meta(row_counts)

        log.info("registry.build.done", **row_counts)
        return {"row_counts": row_counts, "meta": meta}

    # ------------------------------------------------------------------
    # Source construction.

    def _build_sources(self) -> _SourceBundle:
        s = self.settings
        cache = ParquetCache(cache_dir=s.paths.cache)
        xlsx = XlsxReader(cache=cache)
        loader = RandonneurDataLoader()

        ef_target_source = EfCfTargetIndexSource(path=s.paths.ef_cf_parquet)
        ef_valid_uuids = frozenset(row.code for row in ef_target_source.read())

        bio_sources = [
            PlaceholderEcoinventSource(xlsx=xlsx, xlsx_path=s.paths.placeholder_xlsx),
            PlaceholderEfNativeSource(xlsx=xlsx, xlsx_path=s.paths.placeholder_xlsx),
            RandonneurAgribalyseBiosphereSource(loader=loader),
            RandonneurSimaproBiosphereSource(loader=loader),
            AGB32FlowmapperBiosphereSource(path=s.paths.biosphere_flowmap_json),
            RandonneurWaterSlashM3Source(loader=loader),
            HarmonisedFlowsSource(
                path=s.paths.harmonised_flows_gz,
                valid_uuids=ef_valid_uuids,
            ),
            CuratedOverridesSource(path=s.paths.curated_overrides_json),
            LlmReviewedSource(path=s.paths.llm_reviewed_xlsx, cache=cache),
        ]
        if s.apply_transitive_layer:
            bio_sources.append(
                PlaceholderTransitiveSource(xlsx=xlsx, xlsx_path=s.paths.placeholder_xlsx)
            )

        # No technosphere mapping sources are registered yet — the existing
        # technosphere chain operates entirely through bw2io's
        # ``match_database`` + the ``custom-technosphere-fixes.json`` migration.
        # When we author technosphere-specific mappings they go here.
        tech_sources: list[Any] = []

        unmatchable_sources = [
            PlaceholderUnmatchableSource(xlsx=xlsx, xlsx_path=s.paths.placeholder_xlsx),
            RandonneurUnlinkedListSource(loader=loader),
            CuratedOverridesSource(
                path=s.paths.curated_overrides_json,
                select_unmatchable=True,
            ),
        ]

        return _SourceBundle(
            biosphere_mapping_sources=tuple(bio_sources),
            technosphere_mapping_sources=tuple(tech_sources),
            unmatchable_sources=tuple(unmatchable_sources),
            # Extras come first so they win the (source_unit, target_unit) dedup;
            # the upstream ``generic-brightway-unit-conversions`` datapackage has
            # several time-unit entries with the multipliers inverted (e.g.
            # ``s → hour: 3600`` instead of ``1/3600``), which compounds 3600^2
            # through the supply chain when an AGB activity consumes power-sawing
            # in seconds. Letting the extras file override those entries is
            # cheaper than waiting for an upstream fix.
            unit_conversion_sources=(
                ExtraUnitConversionSource(path=s.paths.extra_unit_conversions_json),
                RandonneurUnitConversionSource(loader=loader),
            ),
            unit_alias_sources=(
                RandonneurUnitAliasSource(loader=loader),
                RandonneurUnitNormalisationSource(loader=loader),
            ),
            context_sources=(RandonneurSimaproContextSource(loader=loader),),
            deletion_sources=(
                AgbDeleteAggregatedSource(
                    processes_path=s.paths.delete_aggregated_processes_json,
                    products_path=s.paths.delete_aggregated_products_json,
                ),
            ),
            edge_label_sources=(AgbEdgeLabelsSource(path=s.paths.edge_label_corrections_json),),
            ef_target_sources=(ef_target_source,),
        )

    # ------------------------------------------------------------------
    # Per-parquet writers.

    def _write_biosphere(self, bundle: _SourceBundle) -> int:
        rows: list[Mapping] = []
        for src in bundle.biosphere_mapping_sources:
            tag = src.__class__.__name__
            with StepTimer(self._log, f"registry.source.{tag}"):
                rows.extend(src.read())
        rows = [r for r in rows if r.source_kind == SourceKind.AGB_FLOW]
        df = MappingTable().to_dataframe(self._dedupe_mappings(rows))
        df.to_parquet(self.settings.paths.registry_mappings_biosphere, index=False)
        return len(df)

    def _write_technosphere(self, bundle: _SourceBundle) -> int:
        rows: list[Mapping] = []
        for src in bundle.technosphere_mapping_sources:
            with StepTimer(self._log, f"registry.source.{src.__class__.__name__}"):
                rows.extend(src.read())
        df = MappingTable().to_dataframe(self._dedupe_mappings(rows))
        df.to_parquet(self.settings.paths.registry_mappings_technosphere, index=False)
        return len(df)

    def _write_unmatchable(self, bundle: _SourceBundle) -> int:
        rows: list[Mapping] = []
        for src in bundle.unmatchable_sources:
            with StepTimer(self._log, f"registry.source.{src.__class__.__name__}"):
                rows.extend(src.read())
        df = MappingTable().to_dataframe(self._dedupe_mappings(rows))
        df.to_parquet(self.settings.paths.registry_unmatchable, index=False)
        return len(df)

    def _write_unit_conversions(self, bundle: _SourceBundle) -> int:
        rows: list[UnitConversion] = []
        for src in bundle.unit_conversion_sources:
            rows.extend(src.read())
        # Dedupe by (source_unit, target_unit) — first-seen wins.
        seen: set[tuple[str, str]] = set()
        deduped: list[UnitConversion] = []
        for r in rows:
            k = (r.source_unit, r.target_unit)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(r)
        df = UnitConversionTable().to_dataframe(deduped)
        df.to_parquet(self.settings.paths.registry_unit_conversions, index=False)
        return len(df)

    def _write_unit_aliases(self, bundle: _SourceBundle) -> int:
        rows: list[UnitAlias] = []
        for src in bundle.unit_alias_sources:
            rows.extend(src.read())
        seen: set[tuple[str, str]] = set()
        deduped: list[UnitAlias] = []
        for r in rows:
            k = (r.alias.lower(), r.canonical.lower())
            if k in seen:
                continue
            seen.add(k)
            deduped.append(r)
        df = UnitAliasTable().to_dataframe(deduped)
        df.to_parquet(self.settings.paths.registry_unit_aliases, index=False)
        return len(df)

    def _write_context_norm(self, bundle: _SourceBundle) -> int:
        rows: list[ContextNorm] = []
        for src in bundle.context_sources:
            rows.extend(src.read())
        df = ContextNormTable().to_dataframe(rows)
        df.to_parquet(self.settings.paths.registry_context_normalisation, index=False)
        return len(df)

    def _write_deletions(self, bundle: _SourceBundle) -> int:
        rows: list[Deletion] = []
        for src in bundle.deletion_sources:
            rows.extend(src.read())
        df = DeletionTable().to_dataframe(rows)
        df.to_parquet(self.settings.paths.registry_deletions, index=False)
        return len(df)

    def _write_edge_labels(self, bundle: _SourceBundle) -> int:
        rows: list[EdgeLabelCorrection] = []
        for src in bundle.edge_label_sources:
            rows.extend(src.read())
        df = EdgeLabelTable().to_dataframe(rows)
        df.to_parquet(self.settings.paths.registry_edge_label_corrections, index=False)
        return len(df)

    def _write_ef_target_index(self, bundle: _SourceBundle) -> int:
        rows: list[EfTargetIndexRow] = []
        for src in bundle.ef_target_sources:
            rows.extend(src.read())
        df = EfTargetIndexTable().to_dataframe(rows)
        df.to_parquet(self.settings.paths.registry_target_index_ef, index=False)
        return len(df)

    # ------------------------------------------------------------------
    # Dedupe + meta.

    @staticmethod
    def _dedupe_mappings(rows: list[Mapping]) -> list[Mapping]:
        """Per (source signature, target, tier, conversion), keep the first row.

        The harmonised flows source emits one row per altLabel, so the same
        (name_lower, bucket, target_uuid) appears thousands of times — they
        all collapse to one row here. Re-runs of the builder are
        byte-identical (modulo timestamps in the meta file).

        ``unit_conversion`` is part of the key so a curated row that
        overrides the conversion factor (substance-specific kg↔m³ for
        water/wood/gas, specific activity for radionuclides) can co-exist
        with an upstream placeholder row that has the same target name but
        a default 1.0 conversion. The matcher's tier+provenance sort then
        picks the curated row first (curated_overrides < placeholder.* in
        provenance ordering at the same tier).
        """
        seen: set[tuple] = set()
        out: list[Mapping] = []
        for r in rows:
            key = (
                r.source_kind.value,
                r.name_lower,
                r.source_top_bucket.value,
                r.source_unit,
                r.target_db,
                r.target_code,
                r.target_name.strip().lower(),
                int(r.priority_tier),
                float(r.unit_conversion or 1.0),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _write_meta(self, row_counts: dict[str, int]) -> dict[str, Any]:
        sources_meta = self._hash_sources()
        meta = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agribalyse_version": self.settings.agribalyse_version,
            "ecoinvent_version": self.settings.resolved_ecoinvent_version,
            "ef_version": self.settings.ef_version,
            "row_counts": row_counts,
            "source_hashes": sources_meta,
            "tiers": {t.name: int(t) for t in Tier},
            "tiers_fill_only": [t.name for t in Tier if t.is_fill_only()],
            "tiers_llm_gated": [t.name for t in Tier if t.is_llm_gated()],
        }
        self.settings.paths.registry_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    def _hash_sources(self) -> dict[str, str]:
        """SHA-256 of every on-disk source artifact for reproducibility tracking."""
        s = self.settings.paths
        files = {
            "placeholder_xlsx": s.placeholder_xlsx,
            "harmonised_flows_gz": s.harmonised_flows_gz,
            "ef_cf_parquet": s.ef_cf_parquet,
            "biosphere_flowmap_json": s.biosphere_flowmap_json,
            "agb_ei_biosphere_json": s.biosphere_flowmap_json,
            "edge_label_corrections_json": s.edge_label_corrections_json,
            "delete_aggregated_processes_json": s.delete_aggregated_processes_json,
            "delete_aggregated_products_json": s.delete_aggregated_products_json,
            "llm_reviewed_xlsx": s.llm_reviewed_xlsx,
            "curated_overrides_json": s.curated_overrides_json,
        }
        out: dict[str, str] = {}
        for label, path in files.items():
            if path.exists():
                h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                out[label] = f"sha256:{h}"
            else:
                out[label] = "missing"
        return out
