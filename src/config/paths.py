"""Filesystem layout. Frozen dataclass — single instance passed everywhere.

Layout::

    source/      authoritative inputs + the randonneur packages we publish
    cache/       derived parquet/pickle caches (gitignored)
    registry/    built MappingRegistry parquets (one per concern)
    dashboard/   run reports, audit logs, dashboards
    to_review/   human-review artifacts
    unlinked/    residual unlinked exports

REFACTOR_FINAL F6 removed ``.bw_projects/`` — the runtime is bw2data-free
and never bootstraps a Brightway project. There is no legacy ``inputs/``
(renamed to ``source/``) and no ``outputs/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# src/config/paths.py → src/config → src → repo root
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """All filesystem locations the pipeline reads or writes."""

    package_root: Path = field(default_factory=lambda: _REPO_ROOT)

    # --- Top-level directories --------------------------------------------

    @property
    def source(self) -> Path:
        return self.package_root / "source"

    @property
    def randonneur_packages(self) -> Path:
        return self.source / "randonneur_packages"

    @property
    def bw2io_data(self) -> Path:
        """Static reference data lifted from bw2io (migration JSON files)."""
        return self.source / "bw2io-data"

    @property
    def cache(self) -> Path:
        return self.package_root / "cache"

    @property
    def registry(self) -> Path:
        return self.package_root / "registry"

    @property
    def dashboard(self) -> Path:
        return self.package_root / "dashboard"

    @property
    def to_review(self) -> Path:
        return self.package_root / "to_review"

    @property
    def unlinked(self) -> Path:
        return self.package_root / "unlinked"

    # --- Source files -----------------------------------------------------

    @property
    def agribalyse_csv(self) -> Path:
        return self.source / "AGB32_final.CSV"

    @property
    def placeholder_xlsx(self) -> Path:
        return self.source / "placeholder_flow_classification.xlsx"

    @property
    def harmonised_flows_gz(self) -> Path:
        return self.source / "harmonised-flows-simple.json.gz"

    @property
    def ef_cf_parquet(self) -> Path:
        return self.source / "EF-LCIAMethod_CF(EF-v3.1)__lciamethods_CF.parquet"

    @property
    def simapro_ef31_xlsx(self) -> Path:
        """SimaPro EF 3.1 (adapted) method export — the CF table ADEME used
        to compute the AGRIBALYSE 3.2 reference scores. Source for cross-
        checking our JRC-derived ``ef_cf_parquet`` (see FIX_DATA.md § 2)."""
        return self.source / "EF3.1 (adapted) (1).XLSX"

    @property
    def simapro_ef31_cache(self) -> Path:
        return self.cache / "simapro-EF31-adapted-cfs.parquet"

    @property
    def biosphere_flowmap_json(self) -> Path:
        return self.source / "agribalyse-3.2-ecoinvent-3.10-biosphere.json"

    @property
    def edge_label_corrections_json(self) -> Path:
        return self.source / "agribalyse-3.2-correct-ecoinvent-edge-labels.json"

    @property
    def delete_aggregated_processes_json(self) -> Path:
        return self.source / "agribalyse-3.2-delete-aggregated-ecoinvent-processes.json"

    @property
    def delete_aggregated_products_json(self) -> Path:
        return self.source / "agribalyse-3.2-delete-aggregated-ecoinvent-products.json"

    @property
    def llm_reviewed_xlsx(self) -> Path:
        return self.source / "agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx"

    @property
    def curated_overrides_json(self) -> Path:
        return self.source / "curated_overrides.json"

    @property
    def extra_unit_conversions_json(self) -> Path:
        return self.source / "agribalyse-3.2-extra-unit-conversions.json"

    @property
    def activity_location_overrides_json(self) -> Path:
        """Per-activity ISO-2 location overrides for AGB stand-ins whose
        aggregate location (GLO / RoW / RER) hides ADEME's intended
        regional context. Consumed by ``ActivityLocationOverrides`` →
        ``_col_id_to_location`` in the link pipeline."""
        return self.source / "agribalyse-3.2-activity-location-overrides.json"

    @property
    def water_use_bio3_bridge_json(self) -> Path:
        """Curated EF v3.1 (adapted) water-use CF mapping to ecoinvent-
        3.9.1-biosphere flow codes. Replaces the broken SimaPro per-kg /
        per-m³ augmentation logic with explicit m³-normalised CFs.
        Consumed by ``MethodCfRegistryBuilder`` for the water-use method."""
        return self.source / "water-use-bio3-bridge.json"

    @property
    def ademe_reference_synthese_raw(self) -> Path:
        return self.source / "AGRIBALYSE3.2_reference_synthese_raw.parquet"

    @property
    def custom_technosphere_fixes_json(self) -> Path:
        return self.randonneur_packages / "agribalyse-3.2-custom-technosphere-fixes.json"

    @property
    def biosphere3_flows_json(self) -> Path:
        """One-time snapshot of biosphere3 flows lifted from bw2io (F6)."""
        return self.source / "biosphere3-flows.json"

    @property
    def ecoinvent_biosphere_flows_json(self) -> Path:
        """One-time snapshot of the ecoinvent biosphere database (F6)."""
        return self.source / "ecoinvent-3.9.1-biosphere-flows.json"

    @property
    def ef_methods_snapshot_json(self) -> Path:
        """One-time snapshot of EF v3.1 method tuples and their inherited CFs (F6)."""
        return self.source / "ef-v31-methods.json"

    @property
    def ecoinvent_exchanges_parquet(self) -> Path:
        """One-time snapshot of ecoinvent-3.9.1-cutoff exchanges (REFACTOR_FINAL F7).

        Required so the ScoringPackage carries the full ecoinvent supply
        chain — ``Database.process()``'s dependency walker is gone, so the
        exchanges have to live in the ``source/`` snapshot just like the
        biosphere/EF flow snapshots.
        """
        return self.source / "ecoinvent-3.9.1-cutoff-exchanges.parquet"

    @property
    def importer_cache_pkl(self) -> Path:
        return self.cache / "importer_cache.pkl"

    @property
    def linked_cache_pkl(self) -> Path:
        """Fully linked graph snapshot for the fast rescore path."""
        return self.cache / "linked_cache.pkl"

    @property
    def parameter_overrides_csv(self) -> Path:
        """Local what-if parameter overrides (gitignored; cleared by ``dds-reset``)."""
        return self.source / "parameter_overrides.csv"

    # --- Registry parquets ------------------------------------------------

    @property
    def registry_meta(self) -> Path:
        return self.registry / "registry.meta.json"

    @property
    def registry_parameters(self) -> Path:
        """Per-process parameter definitions (``dds-build-parameters``)."""
        return self.registry / "parameters.parquet"

    @property
    def registry_exchange_formulas(self) -> Path:
        """Exchange-formula table (``dds-build-parameters``)."""
        return self.registry / "exchange_formulas.parquet"

    @property
    def registry_mappings_biosphere(self) -> Path:
        return self.registry / "mappings_biosphere.parquet"

    @property
    def registry_mappings_technosphere(self) -> Path:
        return self.registry / "mappings_technosphere.parquet"

    @property
    def registry_unmatchable(self) -> Path:
        return self.registry / "unmatchable.parquet"

    @property
    def registry_unit_conversions(self) -> Path:
        return self.registry / "unit_conversions.parquet"

    @property
    def registry_unit_aliases(self) -> Path:
        return self.registry / "unit_aliases.parquet"

    @property
    def registry_context_normalisation(self) -> Path:
        return self.registry / "context_normalisation.parquet"

    @property
    def registry_deletions(self) -> Path:
        return self.registry / "deletions.parquet"

    @property
    def registry_edge_label_corrections(self) -> Path:
        return self.registry / "edge_label_corrections.parquet"

    @property
    def registry_target_index_ef(self) -> Path:
        return self.registry / "target_index_ef.parquet"

    @property
    def registry_biosphere_catalog(self) -> Path:
        return self.registry / "biosphere_catalog.parquet"

    @property
    def registry_ecoinvent_catalog(self) -> Path:
        return self.registry / "ecoinvent_catalog.parquet"

    @property
    def registry_ef_flows(self) -> Path:
        return self.registry / "ef_flows.parquet"

    @property
    def registry_method_cfs_dir(self) -> Path:
        return self.registry / "method_cfs"

    @property
    def registry_method_cfs_index(self) -> Path:
        return self.registry_method_cfs_dir / "_index.json"

    @property
    def registry_product_catalog(self) -> Path:
        return self.registry / "product_catalog.parquet"

    @property
    def registry_activity_catalog(self) -> Path:
        """Label catalog keyed by the technosphere's own column id
        (``activity_id``): one row per matrix column, covering agribalyse
        foreground, ecoinvent background, and Allocator multifunctional
        splits. Built from the same run's ScoringPackage so it can never
        skew from the matrix. Consumed by the bundle's key resolver to
        name every activity/technosphere-flow in exports (e.g. Activity
        Browser); distinct from ``product_catalog`` (agribalyse-only,
        backtest product mapping)."""
        return self.registry / "activity_catalog.parquet"

    @property
    def registry_cf_comparison_join(self) -> Path:
        """Complete per-flow join of SimaPro adapted EF 3.1 CFs against the built
        registry CFs (one row per registry biosphere code × method, matched +
        registry-only). Written by ``dds-compare-cfs`` via
        :class:`reporting.CfComparisonJoinBuilder`; the matched rows are flattened
        to the dashboard by :pyattr:`dashboard_cf_comparison_csv`."""
        return self.registry / "cf_comparison_join.parquet"

    @property
    def registry_cf_comparison_by_code(self) -> Path:
        """Per-registry-``code`` SimaPro CF sidecar ``(code, method, cf_simapro,
        match_basis, name_simapro, name_registry)``. Written by ``dds-compare-cfs``
        via :class:`reporting.CfComparisonByCodeBuilder` (all 19 methods); consumed
        by :class:`reporting.SimaProCfLookup` to put SimaPro's properly-matched CF
        beside ours in the flow-decomposition toggle."""
        return self.registry / "cf_comparison_by_code.parquet"

    @property
    def scoring_packages_root(self) -> Path:
        return self.cache / "scoring_packages"

    # --- Dashboard / audit ------------------------------------------------

    @property
    def dashboard_run_report(self) -> Path:
        return self.dashboard / "run_report.json"

    @property
    def dashboard_override_log(self) -> Path:
        return self.dashboard / "override_audit.parquet"

    @property
    def dashboard_suppressed_strategy_log(self) -> Path:
        return self.dashboard / "suppressed_strategies.parquet"

    @property
    def dashboard_drop_tally(self) -> Path:
        return self.dashboard / "drop_tally.json"

    @property
    def dashboard_link_log(self) -> Path:
        return self.dashboard / "link_all.log"

    @property
    def dashboard_backtest_dir(self) -> Path:
        return self.dashboard / "backtest"

    @property
    def dashboard_orphan_activities(self) -> Path:
        return self.dashboard / "orphan_activities.parquet"

    @property
    def dashboard_cf_comparison_csv(self) -> Path:
        """Flat CSV of the matched SimaPro-vs-registry per-flow comparisons,
        consumed by the dashboard's CF-comparison tab. Written by
        ``CfComparisonCsvEmitter`` — directly from ``dds-compare-cfs``, or
        re-flattened from :pyattr:`registry_cf_comparison_join` by
        ``dds-build-cf-comparison-csv``."""
        return self.dashboard / "cf_comparison.csv"

    @property
    def dashboard_backtest_pass1_csv(self) -> Path:
        """Per-product × method %-diff matrix the dashboard renders. Written by
        ``BacktestPass1Emitter``."""
        return self.dashboard / "backtest_pass1.csv"

    @property
    def dashboard_decomp_dir(self) -> Path:
        """Per-product flow-decomposition JSONs (``<code>.json``). Written by
        ``dds-build-flow-decomp``."""
        return self.dashboard / "decomp"

    @property
    def dashboard_outlier_reasons(self) -> Path:
        """Per-impact-category outlier explanations (the column-header / cell
        notes). Hand-curated; consumed by the dashboard's backtest tab."""
        return self.dashboard / "outlier_reasons.json"

    @property
    def dashboard_product_reasons(self) -> Path:
        """Per-product × outlier-impact explanations, authored by an LLM pass
        over the flow decomposition. Written by ``dds-build-product-reasons``;
        rendered above the impact-level note in the cell tooltip."""
        return self.dashboard / "product_reasons.json"

    # --- Helpers ----------------------------------------------------------

    def ensure_runtime_dirs(self) -> None:
        """Create runtime directories if missing. ``source/`` is never created here."""
        for d in (
            self.cache,
            self.registry,
            self.dashboard,
            self.to_review,
            self.unlinked,
            self.dashboard_backtest_dir,
            self.scoring_packages_root,
        ):
            d.mkdir(parents=True, exist_ok=True)
