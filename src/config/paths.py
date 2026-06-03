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
    def cache_cf_per_flow_joined_parquet(self) -> Path:
        """Per-(method, ecoinvent biosphere flow) join of SimaPro CFs against
        the scoring registry's CFs. Written by ``FlowLevelCfJoiner`` from the
        ``dds-compare-cfs`` CLI. Not yet consumed by the dashboard; reserved
        for a future drill-down view."""
        return self.cache / "cf_per_flow_joined.parquet"

    # --- Registry parquets ------------------------------------------------

    @property
    def registry_meta(self) -> Path:
        return self.registry / "registry.meta.json"

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
    def dashboard_cf_stats_csv(self) -> Path:
        return self.dashboard / "cf_stats.csv"

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
