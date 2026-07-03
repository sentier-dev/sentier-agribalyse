"""``LinkAllPipeline`` — orchestrate every linking step.

Replaces the legacy ``scripts/link_all.py`` (~1633 lines) with a
composition of small classes. Each step is named so the run report
can attribute behavior to it.

Order:

1. Brightway project bootstrap + ecoinvent.
2. EF layer precondition (parquets must exist on disk).
3. SimaPro CSV (cached).
4. ``AggregateDeleter``.
5. Internal AGB linker (apply_strategies + first match_database).
6. Restore SimaPro names + simapro→ecoinvent transform.
7. Edge label corrector + biosphere flowmap (NaN-cf patched).
8. Standard label normaliser.
9. Production reclassifier (functional=False → technosphere).
10. Biosphere label normaliser + parquet-backed prelinker (StrategyRunner-wrapped).
11. ``BiosphereMatcher`` walks tiers.
12. ``TechnosphereMatcher``.
13. Pre-write coverage snapshot + ``UnlinkedExporter``.
14. NaN guard.
15. ``ScoringPackage`` emit:
    drop_unlinked → ``ExchangeFrameBuilder`` → ``Allocator`` →
    ``MethodCfRegistryLoader`` → ``ScoringPackageBuilder`` →
    ``ScoringPackageStore.write`` (+ ``ProductCatalogBuilder``).
16. ``RunReport`` writes JSON; ``AuditLog`` + ``SuppressedStrategyLog`` write parquet.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from config import Settings
from core import StepTimer
from core.logging import Logging
from ef.cf_registry import MethodCfRegistryBuilder, MethodCfRegistryLoader
from ef.cf_simapro_filter import SimaProCfFilter
from ef.cf_table import EfCfTable
from ef.regional_cf import RegionalCfRegistryBuilder, RegionalCfTable
from ef.regional_water_cf import RegionalWaterCfRegistryBuilder, RegionalWaterCfTable
from ef.simapro_cf_table import SimaProEFCfTable
from ef.water_use_bio3_bridge import WaterUseBio3Bridge
from exports import MappingsComparisonExporter
from matching import (
    AuditLog,
    BiosphereCatalog,
    BiosphereCatalogAugmenter,
    BiosphereMatcher,
    DropTallyTracker,
    EcoinventCatalog,
    RegionalSuffixApplier,
    StrategyRunner,
    SuppressedStrategyLog,
    TechnosphereMatcher,
)
from registry import MappingRegistry
from reporting import CoverageReporter, RunReport, UnlinkedExporter
from scoring import (
    ActivityLocationOverrides,
    ActivityLocationParser,
    Allocator,
    DanglingEdgeAuditor,
    DanglingEdgePruner,
    EcoinventExchangesIngester,
    ExchangeFrameBuilder,
    IdNameResolver,
    PerennialLifecycleFilter,
    ProductCatalogBuilder,
    ProductDeduplicator,
    ScoringPackage,
    ScoringPackageBuilder,
    ScoringPackageStore,
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
    RegionalSourceNameSnapshotter,
    RestoreSimaproNamesTransform,
    SimaProImporter,
    StandardLabelNormaliser,
    WasteTreatmentDummyFixer,
    WasteTreatmentFunctionalPromoter,
)


@dataclass(frozen=True)
class LinkAllOptions:
    """Pipeline options.

    ``skip_ecoinvent`` is a vestigial flag — the runtime no longer loads
    ecoinvent into a bw2data project (REFACTOR_FINAL F4). It is kept
    here so existing CLI invocations don't break; build-time downloads
    happen in ``dds-build-ecoinvent-catalog``.
    """

    skip_ecoinvent: bool = True


@dataclass
class LinkAllPipeline:
    settings: Settings
    options: LinkAllOptions = field(default_factory=LinkAllOptions)
    report: RunReport = field(default_factory=RunReport)

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------

    def run(self) -> RunReport:
        s = self.settings
        s.paths.ensure_runtime_dirs()
        log = self._log
        report = self.report
        report.settings_summary = {
            "agribalyse_version": s.agribalyse_version,
            "ecoinvent_version": s.resolved_ecoinvent_version,
            "ef_version": s.ef_version,
            "apply_llm_overrides": s.apply_llm_overrides,
            "apply_transitive_layer": s.apply_transitive_layer,
        }

        # 1. Registry — no Brightway project setup required at runtime
        # (REFACTOR_FINAL F4: ``BrightwayProject`` is gone; the catalog
        # parquets are pre-built by the dedicated build CLIs).
        # 2. Registry
        with StepTimer(log, "link.registry.load"):
            registry = MappingRegistry.load(s)

        # 3. EF data plane (parquets, precomputed via dedicated CLIs).
        # Phase L2: ``EfDatabase.install`` and ``EfMethodAugmenter.apply`` are
        # gone — both wrote into SQLite. Their replacements
        # (``EfFlowsRegistryBuilder``, ``MethodCfRegistryBuilder``) are run
        # by the build-time CLIs. The pipeline only asserts the parquets
        # exist, so any drift surfaces here rather than deep inside matchers.
        with StepTimer(log, "link.ef.precondition"):
            ef_stats = self._assert_ef_parquets(s)
        report.add_stage("ef_layer", ef_stats)

        # 4. CSV + transforms
        with StepTimer(log, "link.csv.load"):
            sp = SimaProImporter(s).load()

        with StepTimer(log, "link.transforms.delete_aggregated"):
            del_stats = AggregateDeleter(registry=registry).apply(sp)
        report.add_stage("delete_aggregated", del_stats)

        # Audit + suppression accumulators (used through the rest of the run).
        audit = AuditLog(output_path=s.paths.dashboard_override_log)
        suppressed = SuppressedStrategyLog(output_path=s.paths.dashboard_suppressed_strategy_log)
        runner = StrategyRunner(suppressed_log=suppressed)
        drops = DropTallyTracker()

        with StepTimer(log, "link.transforms.internal_agb"):
            InternalAgbLinker(runner=runner).apply(sp)

        # Edge labels MUST run before RestoreSimaproNamesTransform: the edge-labels
        # datapackage rewrites "... S - Copied from Ecoinvent U" → "... U", which is
        # the form the simapro-ecoinvent-3.9.1-cutoff datapackage matches against.
        # Reversing this order leaves all SimaPro-form ecoinvent names unmatchable.
        with StepTimer(log, "link.transforms.edge_labels"):
            edge_stats = EdgeLabelCorrector(registry=registry).apply(sp)
        report.add_stage("edge_label_corrections", edge_stats)

        with StepTimer(log, "link.transforms.restore_names"):
            RestoreSimaproNamesTransform(runner=runner).apply(sp)

        # Snapshot every biosphere exchange's name BEFORE the flowmap
        # rewrites ``Water, IN`` → ``Water``. Without this snapshot the
        # downstream ``RegionalSuffixApplier`` has no way to recover the
        # country-of-extraction suffix and falls back to a single matrix
        # row per base bio3 UUID — defeating the per-region CF lookup
        # this fix adds.
        with StepTimer(log, "link.transforms.regional_name_snapshot"):
            snapshot_stats = RegionalSourceNameSnapshotter().apply(sp.data)
        report.add_stage("regional_name_snapshot", snapshot_stats)

        with StepTimer(log, "link.transforms.biosphere_flowmap"):
            flowmap_stats = BiosphereFlowmapApplier(settings=s).apply(sp)
        report.add_stage("biosphere_flowmap", flowmap_stats)

        with StepTimer(log, "link.transforms.standard_normalise"):
            StandardLabelNormaliser(runner=runner).apply(sp)

        # ``WasteTreatmentFunctionalPromoter`` MUST run before
        # ``WasteTreatmentDummyFixer``. The dummy fixer rewrites the
        # ``functional=False, amount=0`` stub to ``functional=True,
        # amount=1.0`` whenever it finds the canonical comment, which
        # masks the real waste-treatment amount sitting on a sibling
        # ``technosphere, functional=True`` row (e.g. ``Compost, of
        # biowaste (amendment)`` carries ``technosphere 900 kg`` =
        # 900 kg of biowaste treated per process run). Promoting the
        # functional technosphere row first preserves that 900-kg
        # reference — otherwise the dummy fixer pins production at 1
        # and downstream consumers pull 900x the consumption inputs
        # they should.
        with StepTimer(log, "link.transforms.waste_treatment_functional_promote"):
            promo_stats = WasteTreatmentFunctionalPromoter().apply(sp)
        report.add_stage("waste_treatment_functional_promote", promo_stats)

        with StepTimer(log, "link.transforms.waste_treatment_dummy_fix"):
            dummy_stats = WasteTreatmentDummyFixer().apply(sp)
        report.add_stage("waste_treatment_dummy_fix", dummy_stats)

        with StepTimer(log, "link.transforms.production_reclassify"):
            recl_stats = ProductionReclassifier().apply(sp)
        report.add_stage("production_reclassify", recl_stats)

        with StepTimer(log, "link.transforms.orphan_activity_purge"):
            orphan_stats = OrphanActivityPurger(settings=s).apply(sp)
        report.add_stage("orphan_activity_purge", orphan_stats)

        # 5. Biosphere chain + registry-driven matcher
        with StepTimer(log, "link.biosphere.normalise"):
            BiosphereLabelNormaliser(runner=runner, settings=s).apply(sp)

        with StepTimer(log, "link.biosphere.catalog"):
            # ``BiosphereCatalog.load`` raises ``FileNotFoundError`` with the
            # actionable message; no need to re-check existence here.
            catalog = BiosphereCatalog.load(
                s.paths.registry_biosphere_catalog,
                db_names=(s.biosphere_db_name, "biosphere3", s.ef_db_name),
            )

        bio_db_name = (
            s.biosphere_db_name if s.biosphere_db_name in catalog.db_names else "biosphere3"
        )
        with StepTimer(log, "link.biosphere.prelink"):
            prelink_stats = BioStrategyChain(
                runner=runner,
                bio_db_name=bio_db_name,
                catalog=catalog,
            ).apply(sp)
        report.add_stage("biosphere_prelink", prelink_stats)

        with StepTimer(log, "link.biosphere.match"):
            bio_stats = BiosphereMatcher(
                settings=s,
                registry=registry,
                catalog=catalog,
                audit=audit,
                drops=drops,
            ).match(sp.data)
        report.add_stage(
            "biosphere_matched",
            {
                "n_total": bio_stats.n_total,
                "n_linked": bio_stats.n_linked,
                "by_tier": {t.name: c for t, c in bio_stats.by_tier.items()},
                "n_unit_rejected": bio_stats.n_unit_rejected,
                "n_ambiguous_skipped": bio_stats.n_ambiguous_skipped,
                "n_unmatchable_recognised": bio_stats.n_unmatchable_recognised,
                "n_regional_suffixed": bio_stats.n_regional_suffixed,
            },
        )

        # Apply the regional ``@<region>`` suffix to EVERY linked
        # biosphere exchange whose source name carries a recognised
        # country / aggregate code. The matcher's own suffix logic only
        # fires when ``_best_candidate`` resolves a new outcome, but
        # ``BiospherePrelinker`` already linked ~99 % of biosphere
        # exchanges by code BEFORE the matcher runs — for those the
        # matcher's outcome equals the prior link and the regional
        # dimension would otherwise be silently dropped. This sweep
        # closes that gap.
        with StepTimer(log, "link.biosphere.regional_suffix_apply"):
            suffix_stats = RegionalSuffixApplier().apply(sp.data)
        report.add_stage(
            "biosphere_regional_suffix",
            {
                "n_visited": suffix_stats.n_visited,
                "n_rewritten": suffix_stats.n_rewritten,
                "n_already_synthetic": suffix_stats.n_already_synthetic,
                "n_unsupported_db": suffix_stats.n_unsupported_db,
                "n_no_input": suffix_stats.n_no_input,
            },
        )

        # Augment the biosphere catalog parquet with one synthetic row per
        # unique ``(db, base_code, region)`` synthesised by the matcher.
        # Downstream the augmented catalog is consumed by
        # ``MethodCfRegistryBuilder`` to emit per-region CFs.
        with StepTimer(log, "link.biosphere.catalog_augment"):
            n_added = BiosphereCatalogAugmenter(
                catalog_path=s.paths.registry_biosphere_catalog,
            ).augment_from_sp_data(sp.data)
        report.add_stage(
            "biosphere_catalog_augment",
            {"n_synthetic_rows_added": n_added},
        )

        # Rebuild the per-method CF parquets so the synthetic regional
        # codes synthesised above pick up the matching SimaPro per-region
        # CFs. Otherwise the scoring package would carry stale CFs (with
        # no rows for the @region-suffixed codes), and the matrix rows
        # for regional water flows would score 0 — defeating the whole
        # point of preserving the regional suffix. This duplicates the
        # wiring done by ``dds-build-method-cfs-registry`` but threads
        # the same fresh CFs into the scoring package emitted below.
        with StepTimer(log, "link.method_cfs.rebuild"):
            self._rebuild_method_cfs(s)
        report.add_stage("method_cfs_rebuild", {"ok": True})

        # 6. Orphan-product re-link (must run before TechnosphereMatcher because
        # the technosphere catalog can't repair these — the consumer's input
        # already points at an AGB phantom product, so name-based matching
        # has nothing left to do).
        with StepTimer(log, "link.technosphere.catalog"):
            ei_catalog = EcoinventCatalog.load(s.paths.registry_ecoinvent_catalog)
        with StepTimer(log, "link.transforms.orphan_product_relink"):
            orphan_stats = OrphanProductRelinker(settings=s, catalog=ei_catalog).apply(sp)
        report.add_stage("orphan_product_relink", orphan_stats)

        # 7. Technosphere
        with StepTimer(log, "link.technosphere.match"):
            tech_stats = TechnosphereMatcher(
                settings=s,
                registry=registry,
                catalog=ei_catalog,
                drops=drops,
            ).match(sp)
        report.add_stage(
            "technosphere_matched",
            {
                "n_total": tech_stats.n_total,
                "n_linked": tech_stats.n_linked,
                "n_unit_conversions_applied": tech_stats.n_unit_conversions_applied,
                "custom_fixes_applied": tech_stats.custom_fixes_applied,
            },
        )

        # 7. Pre-purge coverage + residual export
        report.add_coverage(CoverageReporter.from_sp_data("pre_write", sp.data))

        with StepTimer(log, "link.unlinked.export"):
            UnlinkedExporter(settings=s, registry=registry).export(sp.data)

        # 7b. Mappings-comparison report (in-memory, before DB write so reviewers
        # can start the moment matching is done — saves the DB-write + purge wait).
        with StepTimer(log, "link.mappings_comparison.export"):
            MappingsComparisonExporter(settings=s, sp_data=sp.data).export()

        # 8. NaN guard (defensive)
        nans = self._collect_nan_samples(sp.data)
        if nans:
            raise ValueError(
                f"NaN exchange amounts detected after linking. Diagnose upstream:\n{nans[:5]}"
            )

        # 9. Scoring-package emit (replaces sp.write_database + MatrixPurger
        # + find_graph_dependents loop). All matrices, ids, and per-method
        # CFs are laid out under ``cache/scoring_packages/<hash>/`` and the
        # content hash becomes the handoff to scoring.
        with StepTimer(log, "link.scoring_package.write"):
            stage_payload = self._emit_scoring_package(sp, s, report)
        report.add_stage("scoring_package", stage_payload)

        # 10. Reports + audits
        report.set_drops(drops.totals_by_strategy())
        report.set_suppressed(suppressed.counts_by_strategy())
        audit.write()
        suppressed.write()
        report.write(s.paths.dashboard_run_report)

        log.info(
            "pipeline.link_all.done",
            audit_rows=len(audit),
            suppressed=sum(suppressed.counts_by_strategy().values()),
            drops=sum(drops.totals_by_strategy().values()),
        )
        return report

    # ------------------------------------------------------------------
    # CF registry rebuild — runs after the catalog augmentation so the
    # per-region synthetic codes pick up SimaPro's per-region CFs.

    @staticmethod
    def _rebuild_method_cfs(s: Settings) -> None:
        cf_table = EfCfTable(path=s.paths.ef_cf_parquet)
        simapro_filter: SimaProCfFilter | None = None
        if s.paths.simapro_ef31_xlsx.exists():
            simapro_filter = SimaProCfFilter(
                simapro_table=SimaProEFCfTable(
                    xlsx_path=s.paths.simapro_ef31_xlsx,
                    cache_path=s.paths.simapro_ef31_cache,
                ),
                biosphere_catalog_path=s.paths.registry_biosphere_catalog,
                flowmap_path=s.paths.biosphere_flowmap_json,
            )
        regional_water_cf_builder = RegionalWaterCfRegistryBuilder(
            regional_table=RegionalWaterCfTable(cf_table=cf_table),
        )
        regional_allowlist: frozenset[tuple[str, str]] = frozenset(
            {
                (
                    "ecotoxicity: freshwater",
                    "comparative toxic unit for ecosystems (CTUe)",
                ),
            }
        )
        regional_cf_builder = RegionalCfRegistryBuilder(
            regional_table=RegionalCfTable(cf_table=cf_table),
            biosphere_catalog_path=s.paths.registry_biosphere_catalog,
            enabled_methods=regional_allowlist,
        )
        water_use_bio3_bridge = WaterUseBio3Bridge.load(s.paths.water_use_bio3_bridge_json)
        # ``SpRegionalWaterCfLoader`` wiring is DISABLED — see the build
        # CLI for the explanation. Both bilateral (±CF) and
        # resource-only emission against our inventory amplify
        # imbalance noise unacceptably.
        MethodCfRegistryBuilder(
            settings=s,
            cf_table=cf_table,
            simapro_filter=simapro_filter,
            water_use_bio3_bridge=water_use_bio3_bridge,
            regional_water_cf_builder=regional_water_cf_builder,
            regional_cf_builder=regional_cf_builder,
            sp_regional_water_cf_loader=None,
        ).build()

    @classmethod
    def _emit_scoring_package(
        cls,
        sp: Any,
        settings: Settings,
        report: RunReport,
    ) -> dict[str, Any]:
        """Drop unlinked, build the frame + matrices, persist the package.

        Side effects (in order):

        1. ``sp.drop_unlinked(i_am_reckless=True)`` so unlinked exchanges
           never reach the matrix builders.
        2. ``ExchangeFrameBuilder().long_from_sp_data(sp.data)`` projects
           AGB exchanges to a long-form DataFrame keyed by integer ids.
        3. ``EcoinventExchangesIngester(...).load_long()`` reads the
           one-time bw2data → parquet snapshot and emits ecoinvent
           supply-chain rows in the same long-form schema. Without this
           the AGB consumption edges would dangle (their input ids
           reference ecoinvent products that no activity column produces),
           and the technosphere would be non-square. Pre-refactor that
           closure was reached via ``Database.process()``'s walker; the
           parquet replaces the walker.
        4. The two long-form frames are concatenated and rebuilt into a
           single ``ExchangeFrame`` carrying ``allocation_factor``.
        5. ``Allocator().allocate(frame)`` splits multifunctional
           activities; the technosphere becomes square.
        6. ``MethodCfRegistryLoader(...).load_all()`` rehydrates
           ``dict[method_tuple, DataFrame[(database, code, amount)]]``
           and we project each row to ``(flow_id, cf)`` using the
           canonical ``ExchangeFrameBuilder.flow_id_for`` hash.
        7. ``ScoringPackageBuilder().build(frame, method_cfs)`` materialises
           the technosphere/biosphere/CF CSR matrices in memory.
        8. ``ScoringPackageStore(...).write(pkg)`` lays out the package
           directory atomically.
        9. ``ProductCatalogBuilder().build(sp.data, ...)`` emits
           ``registry/product_catalog.parquet`` for L4's product mapping.
        """
        n_before = sum(len(p.get("exchanges", [])) for p in sp.data)
        sp.drop_unlinked(i_am_reckless=True)
        n_after = sum(len(p.get("exchanges", [])) for p in sp.data)

        builder = ExchangeFrameBuilder()
        agb_long = builder.long_from_sp_data(sp.data)
        ei_long = EcoinventExchangesIngester(
            path=settings.paths.ecoinvent_exchanges_parquet
        ).load_long()
        combined_long = pd.concat([agb_long, ei_long], ignore_index=True)
        frame = builder.frame_from_long(combined_long)
        ei_catalog_path = settings.paths.registry_ecoinvent_catalog
        ei_catalog_df = (
            pd.read_parquet(ei_catalog_path)
            if ei_catalog_path.exists()
            else pd.DataFrame(
                columns=["database", "code", "name", "unit", "location", "reference_product"]
            )
        )
        auditor = DanglingEdgeAuditor(resolver=IdNameResolver.from_sources(sp.data, ei_catalog_df))
        auditor.snapshot("initial", frame)
        frame, synthetic_provenance = Allocator().allocate_with_provenance(frame)
        auditor.snapshot("after_allocator", frame)
        # Pre-refactor's ``MatrixPurger`` deleted excess producers in a
        # fixed-point loop after every ``Database.process()``. Replacing
        # that with one deterministic pass: drop everyone but the
        # canonical (smallest-id) producer for each product. Required
        # because AGB's orphan-product relinker can pin several
        # processes to the same ecoinvent target product.
        frame = ProductDeduplicator().deduplicate(frame)
        auditor.snapshot("after_dedup", frame)
        auditor.capture_consumer_edges(frame)
        # Drop technosphere consumption edges whose input is no longer
        # produced (or whose output isn't an activity). The
        # ``OrphanProductRelinker`` covers the common case via a
        # ``functional=True`` heuristic, but a residual handful of
        # ``type='product'`` rows escape it; pre-refactor those
        # collapsed via ``MatrixPurger._orphan_products``.
        frame, dangling_stats = DanglingEdgePruner().prune(frame)
        auditor.snapshot("after_pruner", frame)
        report.add_stage("dangling_edges", dangling_stats)

        audit_stats = auditor.write(settings.paths.dashboard / "dangling_edges.parquet")
        report.add_stage("dangling_edges_audit", audit_stats)

        # Drop the NH3 biosphere edges on the Black pepper {VN}
        # non-productive / grubbing-up activities. ADEME's published
        # synthese excludes those acidification emissions (capital-
        # goods convention) but keeps the biogenic CO2 and land-use
        # contributions; the filter is therefore at the (activity,
        # flow) level, not on the technosphere link. The 14 spice
        # SKUs that share the black-pepper proxy run +55.5 % over
        # ADEME's acid reference at baseline.
        # See docs/FIX_ACID_SPICE_LIFECYCLE.md.
        biosphere_catalog_df = pd.read_parquet(settings.paths.registry_biosphere_catalog)
        frame, perennial_stats = PerennialLifecycleFilter(
            biosphere_catalog=biosphere_catalog_df
        ).purge(frame, sp_data=sp.data)
        report.add_stage("perennial_lifecycle_filter", perennial_stats)

        cf_loader = MethodCfRegistryLoader(settings.paths.registry_method_cfs_dir)
        raw_method_cfs = cf_loader.load_all()
        method_cfs = {key: cls._project_method_cfs(df) for key, df in raw_method_cfs.items()}

        # Regional CFs (AWARE per ISO country for water-use). Optional —
        # if the registry has no regional parquets we hand an empty dict
        # to the builder and the scoring path stays as before. The
        # column-location map joins ecoinvent activities (which carry
        # ``location``) onto the technosphere columns the builder will
        # produce in the next step.
        raw_regional_cfs = cf_loader.load_regional_all()
        regional_method_cfs = {
            key: cls._project_regional_method_cfs(df) for key, df in raw_regional_cfs.items()
        }
        activity_location_overrides = ActivityLocationOverrides.load(
            settings.paths.activity_location_overrides_json
        )
        col_id_to_location = cls._col_id_to_location(
            ei_catalog_df, sp.data, overrides=activity_location_overrides
        )

        # AWARE per-activity net-consumption correction for water use.
        # Reads the JRC EF v3.1 parquet's regional CFs (~208 ISO codes)
        # and lets ``ScoringPackageBuilder`` build a per-activity
        # consumption correction row. The 95% asymmetric gate is
        # essential: ecoinvent biosphere has distinct turbine/cooling
        # *input* codes but a single generic *release* code, so naively
        # characterising both sides on the Q row (the ``WaterUseBio3Bridge``
        # approach) accumulates ~1-5% per-activity inventory noise into
        # thousands of m³_eq supply-chain errors. The gate filters that
        # noise and isolates the consumption signal on truly one-sided
        # activities (irrigation, aquaculture, dehydration). See
        # docs/FIX_WATER_USE_AUGMENTATION.md for the derivation.
        aware_regional_cf_by_location = RegionalWaterCfTable(
            cf_table=EfCfTable(path=settings.paths.ef_cf_parquet),
        ).cf_by_location

        pkg = ScoringPackageBuilder().build(
            frame,
            method_cfs,
            regional_cfs=regional_method_cfs,
            col_id_to_location=col_id_to_location,
            biosphere_catalog=biosphere_catalog_df,
            aware_regional_cf_by_location=aware_regional_cf_by_location,
        )
        store = ScoringPackageStore(root=settings.paths.scoring_packages_root)
        package_path = store.write(pkg)

        catalog_path = settings.paths.registry_product_catalog
        # product_catalog stays agribalyse-only (backtest product mapping).
        ProductCatalogBuilder().build(sp.data, catalog_path)
        # activity_catalog is the divergence-free label catalog: one row per
        # *final* technosphere column (agribalyse + ecoinvent + Allocator
        # synthetic splits), keyed by the column's own id. Built from the
        # same run's ScoringPackage so catalog and matrix can never skew —
        # this is what closes the "column id with no catalog row" gap. The
        # synthetic columns (ids that are not flow_id_for hashes) are
        # labelled from their (parent, product) provenance.
        ProductCatalogBuilder().build_from_columns(
            col_ids=pkg.technosphere.col_id_to_idx.keys(),
            sp_data=sp.data,
            ei_catalog_df=ei_catalog_df,
            synthetic_provenance=synthetic_provenance,
            target=settings.paths.registry_activity_catalog,
        )

        report.set_matrix_shape(pkg.n_products, pkg.n_activities)
        return cls._scoring_package_payload(
            pkg=pkg,
            n_before=n_before,
            n_after=n_after,
            package_path=str(package_path),
            catalog_path=str(catalog_path),
        )

    @staticmethod
    def _project_method_cfs(cf_df: pd.DataFrame) -> pd.DataFrame:
        """Project ``(database, code, amount)`` rows → ``(flow_id, cf)``.

        ``MethodCfRegistryLoader`` returns CFs keyed by the natural
        ``(database, code)`` pair from the EF source CSV plus inherited
        biosphere3 entries. ``CharacterizationBuilder`` wants integer
        ``flow_id``s that match the biosphere matrix's row map. Joining is
        a pure function of the canonical key→id hash, so we compute it
        here in a single vectorised pass.
        """
        if cf_df.empty:
            return pd.DataFrame(
                {
                    "flow_id": pd.Series([], dtype="int64"),
                    "cf": pd.Series([], dtype="float64"),
                }
            )
        out = cf_df[["database", "code", "amount"]].copy().reset_index(drop=True)
        out["flow_id"] = [
            ExchangeFrameBuilder.flow_id_for((db, code))
            for db, code in zip(out["database"], out["code"], strict=False)
        ]
        return (
            out[["flow_id", "amount"]]
            .rename(columns={"amount": "cf"})
            .astype({"flow_id": "int64", "cf": "float64"})
        )

    @staticmethod
    def _project_regional_method_cfs(cf_df: pd.DataFrame) -> pd.DataFrame:
        """Project ``(database, code, location, amount)`` → ``(flow_id, location, cf)``.

        Same hash function as :meth:`_project_method_cfs` so a regional
        row's ``flow_id`` matches its global counterpart exactly. The
        ``location`` column survives the projection so the downstream
        ``RegionalCorrectionBuilder`` can join against per-activity
        locations.
        """
        if cf_df.empty:
            return pd.DataFrame(
                {
                    "flow_id": pd.Series([], dtype="int64"),
                    "location": pd.Series([], dtype="string"),
                    "cf": pd.Series([], dtype="float64"),
                }
            )
        out = cf_df[["database", "code", "location", "amount"]].copy().reset_index(drop=True)
        out["flow_id"] = [
            ExchangeFrameBuilder.flow_id_for((db, code))
            for db, code in zip(out["database"], out["code"], strict=False)
        ]
        return (
            out[["flow_id", "location", "amount"]]
            .rename(columns={"amount": "cf"})
            .astype({"flow_id": "int64", "location": "string", "cf": "float64"})
        )

    # Activities tagged with an aggregate region (``GLO`` / ``RoW`` /
    # ``RER`` / ``RNA`` / ``RLA`` / …) carry no national context, so by
    # default they get no regional CF and the scorer falls back to the
    # global CF (the JRC NULL-location row, +42.95 m³/m³ for water).
    # ADEME's reference scores appear to assume the consumer's location
    # (FR) for these un-located inputs when the final product is a FR
    # consumption AGB recipe: a backtest decomposition (2026-05-13) on
    # the +381 % pepper outlier showed 89 % of the score lands on
    # ``bell pepper production, in heated greenhouse GLO``; for the
    # +219 % sea salt outlier 68 % lands on
    # ``market for sodium chloride, powder GLO``. Treating those as
    # FR-equivalent (regional CF 6.98) closes the gap.
    AGGREGATE_REGIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "GLO",
            "ROW",
            "RoW",
            "RER",
            "RNA",
            "RLA",
            "RAS",
            "RAF",
            "OECD",
            "RME",
            "ENTSO-E",
            "UCTE",
            "WECC",
            "FRCC",
            "MRO",
            "NPCC",
            "RFC",
            "SERC",
            "SPP",
            "TRE",
            "WORLD",
            "WORLDPROD",
        }
    )

    # ISO-2 fallback applied to activities whose authoritative or
    # parsed location resolves to an aggregate region. Tested on
    # 2026-05-13: setting this to ``"FR"`` (the natural assumption
    # since AGB consumer products are FR-coded) pushes the water
    # method mean |Δ| from 35.9 % to 66.9 % and outliers from 307 to
    # 980 — over-corrects every product whose ADEME reference is
    # small relative to ours, dragging ~600 products to ~-100 %.
    # ADEME's reference apparently doesn't simply apply the consumer's
    # CF to GLO activities; they use a per-activity regional context
    # we can't recover from the SimaPro export alone. Left at ``None``
    # so the global CF (+42.95 m³/m³) remains the fallback for GLO /
    # RoW / RER / RNA / RLA / OECD aggregate codes.
    AGGREGATE_FALLBACK: ClassVar[str | None] = None

    @classmethod
    def _col_id_to_location(
        cls,
        ei_catalog_df: pd.DataFrame,
        sp_data: Iterable[dict] | None = None,
        *,
        overrides: ActivityLocationOverrides | None = None,
    ) -> dict[int, str]:
        """Map technosphere column ids (``flow_id_for((db, code))``) to
        the activity's location.

        Three sources are stitched, in order of authority:

        * ``ei_catalog_df`` — the ecoinvent catalog, the ground truth
          for any column keyed under ``ecoinvent-3.9.1-cutoff`` (or
          analogous ecoinvent databases).
        * ``sp_data`` — the AGB SimaPro datasets. The metadata's
          ``location`` field is null on 95 % of rows, but the location
          IS encoded in the activity name (see
          :class:`ActivityLocationParser`). Without parsing names, the
          ~14 000 AGB activities that drive mango / banana / asparagus
          water emissions all fall back to the global CF — the
          regional correction touches only the ~2 000 ecoinvent
          columns and the dashboard's water tail stays at +1000 %.
        * ``overrides`` — curated per-``(db, code)`` ISO codes that win
          over both of the above. Used for FR-greenhouse AGB stand-ins
          whose aggregate location masks the FR context ADEME actually
          assumes in the synthese (cf. sweet pepper greenhouse).

        Aggregate locations (``GLO`` / ``RoW`` / ``RER`` / …) are
        re-routed to ``AGGREGATE_FALLBACK`` (default ``None`` — see
        the rationale block on that class attribute for why a blanket
        ``FR`` fallback over-corrected the tropical-fruit tail).
        """
        out: dict[int, str] = {}
        if not ei_catalog_df.empty and "location" in ei_catalog_df.columns:
            for db, code, location in zip(
                ei_catalog_df["database"],
                ei_catalog_df["code"],
                ei_catalog_df["location"],
                strict=False,
            ):
                resolved = cls._resolve_location(location)
                if not resolved:
                    continue
                out[ExchangeFrameBuilder.flow_id_for((str(db), str(code)))] = resolved

        if sp_data is not None:
            parser = ActivityLocationParser()
            for ds in sp_data:
                db = str(ds.get("database") or "")
                code = str(ds.get("code") or "")
                if not db or not code:
                    continue
                # Prefer the dataset's explicit ``location`` when set
                # (~5 % of AGB rows, mostly aggregated regions like
                # ``World`` or ``Europe, Western`` — same form the
                # ecoinvent catalog uses and equally useful here).
                meta = ds.get("location")
                if isinstance(meta, str) and meta:
                    parsed = cls._resolve_location(meta)
                else:
                    # Parser returns a normalised ISO-2 or None;
                    # ``_resolve_location(None)`` then applies the
                    # aggregate fallback.
                    parsed = parser.location_for(ds.get("name"))
                    parsed = cls._resolve_location(parsed)
                if not parsed:
                    continue
                key = ExchangeFrameBuilder.flow_id_for((db, code))
                # Don't overwrite an ecoinvent-catalog entry: the
                # catalog's ISO code is authoritative when present.
                out.setdefault(key, parsed)

        if overrides is not None:
            for db, code, location in overrides.items():
                out[ExchangeFrameBuilder.flow_id_for((db, code))] = location
        return out

    @classmethod
    def _resolve_location(cls, raw: str | None) -> str | None:
        """Normalise an authoritative or parsed location string.

        Returns the ISO-2 prefix for sub-regional codes (``CA-QC`` →
        ``CA``), the :attr:`AGGREGATE_FALLBACK` for aggregate regions
        (``GLO`` / ``RoW`` / ``RER`` / …) and the literal code
        otherwise. Empty / None / single-letter inputs return ``None``.
        """
        if raw is None:
            return cls.AGGREGATE_FALLBACK
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text:
            return None
        head = text.split("-", 1)[0]
        if head.upper() in cls.AGGREGATE_REGIONS:
            return cls.AGGREGATE_FALLBACK
        if len(head) < 2:
            return None
        return head

    @staticmethod
    def _scoring_package_payload(
        *,
        pkg: ScoringPackage,
        n_before: int,
        n_after: int,
        package_path: str,
        catalog_path: str,
    ) -> dict[str, Any]:
        return {
            "exchanges_dropped": n_before - n_after,
            "exchanges_kept": n_after,
            "n_products": pkg.n_products,
            "n_activities": pkg.n_activities,
            "n_biosphere_flows": pkg.n_biosphere_flows,
            "n_methods": len(pkg.methods),
            "content_hash": pkg.content_hash,
            "path": package_path,
            "product_catalog_path": catalog_path,
        }

    @staticmethod
    def _assert_ef_parquets(settings: Settings) -> dict[str, int]:
        """Phase L2 precondition: the two EF parquets must exist before linking."""
        flows_path = settings.paths.registry_ef_flows
        index_path = settings.paths.registry_method_cfs_index
        if not flows_path.exists():
            raise FileNotFoundError(
                f"EF flows registry not found: {flows_path}. "
                f"Run `dds-build-ef-flows-registry` before `dds-link-all`."
            )
        if not index_path.exists():
            raise FileNotFoundError(
                f"Method-CFs registry index not found: {index_path}. "
                f"Run `dds-build-method-cfs-registry` before `dds-link-all`."
            )
        # Cheap row counts — surface them in the run report so a stale build
        # is visible without re-reading the parquets in every consumer.
        n_flows = len(pd.read_parquet(flows_path, columns=["code"]))
        index_payload = json.loads(index_path.read_text())
        n_methods = len(index_payload.get("methods", []))
        return {"ef_flows": n_flows, "methods": n_methods}

    @staticmethod
    def _collect_nan_samples(sp_data: list[dict], limit: int = 5) -> list[dict]:
        out: list[dict] = []
        for ds in sp_data:
            for exc in ds.get("exchanges", []):
                amt = exc.get("amount")
                if isinstance(amt, float) and math.isnan(amt) and len(out) < limit:
                    out.append(
                        {
                            "activity": ds.get("name", ""),
                            "exchange_name": exc.get("name", ""),
                            "type": exc.get("type", ""),
                            "unit": exc.get("unit", ""),
                            "input": exc.get("input"),
                        }
                    )
        return out
