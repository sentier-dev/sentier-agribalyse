"""``dds-build-method-cfs-registry`` — emit ``registry/method_cfs/<slug>/cfs.parquet``.

Reads the EF source CSV plus the snapshotted EF v3.1 method definitions
at ``source/ef-v31-methods.json``. No bw2data, no SQLite
(REFACTOR_FINAL F6).
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from ef import EfCfTable, MethodCfRegistryBuilder, SimaProEFCfTable
from ef.cf_simapro_filter import SimaProCfFilter
from ef.regional_cf import RegionalCfRegistryBuilder, RegionalCfTable
from ef.regional_water_cf import RegionalWaterCfRegistryBuilder, RegionalWaterCfTable
from ef.water_use_bio3_bridge import WaterUseBio3Bridge


class BuildMethodCfsRegistryCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_method_cfs_registry"
    DESCRIPTION: ClassVar[str] = (
        "Materialise registry/method_cfs/ — per-method CF parquets + JSON index."
    )

    def execute(self, args: argparse.Namespace) -> None:
        s = self.settings
        cf_table = EfCfTable(path=s.paths.ef_cf_parquet)
        # Wire the SimaPro EF 3.1 (adapted) intersection when the source
        # XLSX is available — see FIX_DATA.md § 1.1 for why this is
        # required to remove inherited stray CFs (Kaolin, Water[air],
        # JRC-pesticides) that ADEME's authoritative method ignores.
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
        # ``WaterResourceCfAugmenter`` is intentionally NOT wired here.
        # Two scopes of :class:`BiosphereResourcePurger` were tested
        # on 2026-05-17 against the augmenter:
        #   * ``EI3CQUNI*`` only (138 365 edges, 545 acts): under-purges
        #     → water-use median **+6 059 %**, max **+144 283 %**.
        #   * ``EI3CQUNI*`` + ``AGRIBALU*`` (921 068 edges, 2 311 acts,
        #     covers ~100 % of AGB-side pre-amortised water): over-purges
        #     → water-use median **-62 %**, 1 952 outliers (vs 246
        #     baseline).
        # Both fail because the pre-amortised water on AGB
        # ``[Dummy]`` stand-ins is a *partial* signal — some volumes
        # already in ADEME's syntheses, some not. Wholesale prefix-based
        # removal can't discriminate. The right next mechanism is
        # per-activity unconditional contribution (``WaterUseInjector``
        # in ``docs/NEXT_SESSION.md`` lines 117-126), not an augmenter
        # + purger pair.
        # The JRC EF v3.1 parquet ships per-country AWARE CFs (208 ISO
        # codes) alongside the global +42.95 / -42.95 rows. The
        # regional builder emits a sidecar ``regional_cfs.parquet``
        # under the water-use slug; downstream ``ScoringPackageBuilder``
        # turns each row into a per-activity-column correction so
        # tropical and arid regions get their actual AWARE CF instead
        # of the global mean (closes the ~+1000 % outliers on mango /
        # citrus / banana products).
        regional_water_cf_builder = RegionalWaterCfRegistryBuilder(
            regional_table=RegionalWaterCfTable(cf_table=cf_table),
        )
        # Generic per-(bio3 db, bio3 code, location) regional CFs for
        # non-water methods on the audited opt-in allowlist below. JRC
        # ships per-location CFs for ~10 methods but ADEME's AGRIBALYSE
        # reference only regionalises a subset. Enabling regional CFs on
        # a method where ADEME uses the site-generic value (e.g. acid,
        # eutroph terrestrial) diverges our backtest from reference
        # (acid median went from +0.27% to -19.7% with 2 149 outliers
        # vs 145 baseline on 2026-05-21). The allowlist below contains
        # only methods where ADEME's methodology matches per-location
        # regional application:
        #
        # * Ecotoxicity freshwater: slight backtest improvement
        #   confirmed (-45 outliers, median -0.13 pp).
        #
        # Water-use is handled by its own builder above; not listed
        # here. Methods OUTSIDE the allowlist get no regional sidecar
        # and the scoring path keeps the global CF for them.
        REGIONAL_METHOD_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
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
            enabled_methods=REGIONAL_METHOD_ALLOWLIST,
        )
        # Curated EF v3.1 (adapted) bio3 CFs for water-use. Replaces the
        # disabled SimaPro augment_rows path which over-counted by 337x due
        # to the per-kg vs per-m3 name-match mismatch. The bridge adds
        # positive CFs on natural-resource water inputs and negative CFs
        # on water-compartment releases so balanced flows cancel correctly
        # and only true net consumption contributes -- exactly mirroring
        # ADEME's SimaPro reference. Pairs with disabling the per-activity
        # AwareConsumptionCorrectionBuilder, which becomes a double-count
        # once the base CFs are complete on the Q row.
        water_use_bio3_bridge = WaterUseBio3Bridge.load(s.paths.water_use_bio3_bridge_json)

        # ``SpRegionalWaterCfLoader`` wiring is DISABLED — neither
        # variant survives the inventory mismatch:
        # * Bilateral ±CF emission (2026-05-23 run #4): water-use
        #   mean |Δ| jumped from baseline 15.98 % to **69.76 %**.
        #   The ~1-5 % imbalance on every nominally balanced
        #   cooling/turbine activity amplifies through ±6.98 (FR),
        #   ±42.95 (global), ±49.7 (CN) CFs.
        # * Resource-only emission (2026-05-23 run #6): water-use
        #   mean |Δ| jumped to **9 816 %**. ~1 M AGB resource-side
        #   flows now contribute +CF without ANY release-side
        #   cancellation — every balanced cooling pair becomes a
        #   net positive contribution.
        # The synthetic ``@<region>`` matrix rows from
        # :class:`matching.RegionalSuffixApplier` are still emitted —
        # they are needed for a future per-activity-aware approach
        # (resource * release asymmetry per (activity, region) pair) -
        # but the CF assignment is parked here pending that work.
        # See [docs/superpowers/specs/2026-05-22-regional-flow-mapping-design.md](docs/superpowers/specs/2026-05-22-regional-flow-mapping-design.md)
        # risk #1 and the empty-JSON [source/water-use-bio3-bridge.json](../../source/water-use-bio3-bridge.json)
        # for the same failure mode at the global-CF scale.

        MethodCfRegistryBuilder(
            settings=s,
            cf_table=cf_table,
            simapro_filter=simapro_filter,
            water_use_bio3_bridge=water_use_bio3_bridge,
            regional_water_cf_builder=regional_water_cf_builder,
            regional_cf_builder=regional_cf_builder,
            sp_regional_water_cf_loader=None,
        ).build()


def main() -> int:
    return BuildMethodCfsRegistryCli().run()


if __name__ == "__main__":
    sys.exit(main())
