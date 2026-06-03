"""``RegionalWaterCfTable`` / ``RegionalWaterCfRegistryBuilder`` — JRC AWARE
regional water CFs as a per-location correction layer on top of the global
``cfs.parquet``.

The bw2io snapshot characterises 5 bio3 ``Water → air`` codes at the global
AWARE CF (+42.95 m³ deprivation / m³). ADEME's reference applies the *regional*
AWARE CF for each activity's location (e.g. Brazil: 2.43; Cyprus: 74.3;
Argentina: 47.1). That regional split is the dominant driver of the
backtest's water-use tail (464 outliers, median +107 %, top: Nectar de mangue
+1011 %).

The JRC source parquet (``EF-LCIAMethod_CF(EF-v3.1)``) carries the regional
factors at the ``LCIAMethod_location`` column (ISO 3166-1 alpha-2) for 208
locations, mirrored across 11 water FLOW_uuids with identical magnitudes —
sign differs only between *consumption-side* (positive) and
*emission-side* (negative) rows. We collapse those 11 flows into a single
``|CF|`` per location and reapply the sign of the global CF on the bio3 row
we're correcting.

The companion builder emits
``registry/method_cfs/<water-slug>/regional_cfs.parquet`` keyed by
``(database, code, location, amount)``. Downstream
``ScoringPackageBuilder`` resolves each row's bio3 flow row index +
joins activity locations from ``registry/ecoinvent_catalog.parquet`` to
pre-compute a per-activity-column correction vector, so scoring stays at
``q @ B @ supply + delta_vec @ supply`` — no location lookup at score time.

We deliberately do NOT re-enable ``WaterResourceCfAugmenter`` here. That
class adds resource-flow CFs (``Water [natural resource, in ground]`` …)
which the bw2io snapshot omitted; wiring it triggered the 2026-05-08
double-count regression that pushed water-use median to 34 100 %. The
regional correction we apply here scales the existing
``Water → air`` proxies that the inherited CF snapshot already attached
to bio3 — operationally aligned with how ADEME's published scores were
computed against the same proxy set.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import pandas as pd

from ef.cf_table import EfCfTable


@dataclass(frozen=True)
class RegionalWaterCfTable:
    """Per-location |AWARE CF| extracted from the JRC EF v3.1 parquet.

    ``cf_by_location`` returns ``{ISO_code: |regional_cf|}`` where the value
    is the mean of ``|CF EF3.1|`` over the 11 JRC water FLOW_uuids that
    share the same regional grid. The 2026-05-13 audit confirmed those
    magnitudes are identical per location across all 11 flows (only sign
    differs); the mean is a defensive collapse — should the JRC source
    ever differentiate magnitudes between flow types we'd want to surface
    it via a per-flow lookup instead.
    """

    cf_table: EfCfTable

    # JRC's authoritative method-name string for "user deprivation
    # potential (deprivation-weighted water consumption)" — same key
    # ``MethodCfRegistryBuilder.EF_METHOD_MAP`` uses on the source side.
    WATER_USE_METHOD_NAME: ClassVar[str] = "Water use"

    @cached_property
    def cf_by_location(self) -> dict[str, float]:
        df = self.cf_table.raw
        wu = df[df["LCIAMethod_name"] == self.WATER_USE_METHOD_NAME]
        regional = wu[wu["LCIAMethod_location"].notna()].copy()
        if regional.empty:
            return {}
        regional["abs_cf"] = regional["CF EF3.1"].astype(float).abs()
        agg = regional.groupby("LCIAMethod_location")["abs_cf"].mean()
        # Drop any locations where every flow has zero magnitude (would
        # otherwise emit a no-op correction row).
        return {str(loc): float(val) for loc, val in agg.items() if val > 0.0}


@dataclass(frozen=True)
class RegionalWaterCfRegistryBuilder:
    """Emit ``regional_cfs.parquet`` rows for the water-use method.

    Inputs:

    * ``water_global_cfs`` — the per-(database, code, amount) rows
      ``MethodCfRegistryBuilder._cfs_for_method`` already builds for the
      water-use slug.
    * ``regional_table`` — the per-location |CF| lookup.

    Outputs: a list of dicts with columns
    ``(database, code, location, amount)``. The ``amount`` is
    ``sign(global_cf) × |regional_cf[location]|`` — same convention the
    bw2io snapshot used for the global value, but per-location. Rows
    where ``|regional - global| < EPS`` are skipped to avoid emitting
    no-op corrections.
    """

    regional_table: RegionalWaterCfTable

    # Below this magnitude difference (m³ deprivation / m³) the
    # correction is too small to matter against ADEME's noise floor.
    # 0.01 is one-thousandth of the global CF and well below ADEME's
    # quoted precision.
    DIFF_EPS: ClassVar[float] = 0.01

    # Codes carrying the ``@<region>`` separator are synthetic regional
    # rows emitted by ``BiosphereMatcher``. Currently they carry NO
    # Q-vector CF (per-region CF emission is parked — see the cf_registry
    # comment block); the synthetic rows therefore still need the
    # per-activity-location correction this builder generates, same as
    # the base codes. Keep them in.
    SYNTHETIC_CODE_SEPARATOR: ClassVar[str] = "@"

    def build_rows(self, water_global_cfs: list[dict]) -> list[dict]:
        regional = self.regional_table.cf_by_location
        if not regional:
            return []
        out: list[dict] = []
        for cf_row in water_global_cfs:
            try:
                global_amount = float(cf_row.get("amount", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if global_amount == 0.0:
                continue
            db = str(cf_row.get("database", "") or "")
            code = str(cf_row.get("code", "") or "")
            if not db or not code:
                continue
            global_abs = abs(global_amount)
            sign = 1.0 if global_amount >= 0 else -1.0
            for location, regional_abs in regional.items():
                if abs(regional_abs - global_abs) < self.DIFF_EPS:
                    continue
                out.append(
                    {
                        "database": db,
                        "code": code,
                        "location": location,
                        "amount": sign * regional_abs,
                    }
                )
        out.sort(key=lambda r: (r["database"], r["code"], r["location"]))
        return out

    @staticmethod
    def to_dataframe(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["database", "code", "location", "amount"])
        return df.astype(
            {
                "database": "string",
                "code": "string",
                "location": "string",
                "amount": "float64",
            }
        )
