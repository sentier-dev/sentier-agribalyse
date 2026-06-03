"""``WaterResourceCfAugmenter`` — fill the bw2io snapshot's water-use gap.

The bw2io ``LCIA Implementation 3.9.1.xlsx`` mapping table inherits CFs
onto bio3 codes by name — it works fine for emission flows but skips
the *resource* side of EF v3.1's Water use method. Result: bio3
``Water, river [natural resource, in water]`` and friends carry **no**
water-use CF in our matrix, so AGB activities consuming water from
freshwater / groundwater / lakes / wells score 0 against ADEME's
non-zero references.

This class fills the gap with JRC's null-region (global) CF for each
mappable bio3 resource flow. It does NOT import SimaPro CFs (the
project pins to JRC EF v3.1 only) and it does NOT widen the CF on
flows JRC chose not to characterise (sea water, water in air).

The mapping is deliberately narrow — only six unambiguous JRC names
land on bio3 resource flows. Wider mappings would risk double-counting
against the inherited ``Water [air]`` rows (which the
``SimaProCfFilter.SKIP_FILTER_METHODS`` set keeps in place for the
water-use method).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ef.cf_table import EfCfTable

# Mapping from JRC FLOW_name (lowercased + stripped) → list of bio3
# (name, top_compartment) targets. Both bio3 and ecoinvent-3.9.1-biosphere
# copies of each (name, top_compartment) pair receive the same CF. We
# match on (name, top compartment) so multi-sub-compartment bio3 codes
# (e.g. ``Water, unspecified natural origin`` exists with [in water],
# [in ground], [fossil well]) all inherit the same generic ``water``
# CF.
_JRC_TO_BIO3_TARGETS: dict[str, list[tuple[str, str]]] = {
    "lake water": [("water, lake", "natural resource")],
    "river water": [("water, river", "natural resource")],
    "ground water": [("water, well, in ground", "natural resource")],
    "water": [("water, unspecified natural origin", "natural resource")],
    "water to cooling": [("water, cooling, unspecified natural origin", "natural resource")],
    "water to turbine": [("water, turbine use, unspecified natural origin", "natural resource")],
}

_JRC_WATER_USE_METHOD: str = "Water use"


@dataclass(frozen=True)
class WaterResourceCfAugmenter:
    """Emit JRC water-use CFs onto bio3 water resource codes."""

    ef_cf_table: EfCfTable
    biosphere_catalog_path: Path

    JRC_TO_BIO3_TARGETS: ClassVar[dict[str, list[tuple[str, str]]]] = _JRC_TO_BIO3_TARGETS
    JRC_METHOD_NAME: ClassVar[str] = _JRC_WATER_USE_METHOD
    SCORED_DATABASES: ClassVar[tuple[str, ...]] = (
        "ecoinvent-3.9.1-biosphere",
        "biosphere3",
    )

    def cf_rows(self) -> list[dict]:
        """Return ``[{"database", "code", "amount"}, ...]`` for water-use augmentation.

        One row per (database, code) target. Output is sorted by
        ``(database, code)`` for deterministic registry builds.
        """
        cf_by_jrc_name = self._jrc_global_cfs()
        if not cf_by_jrc_name:
            return []
        bio_index = self._bio_index_by_name_top()
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for jrc_name, targets in self.JRC_TO_BIO3_TARGETS.items():
            cf = cf_by_jrc_name.get(jrc_name)
            if cf is None:
                continue
            for bio_name, bio_top in targets:
                for db, code in bio_index.get((bio_name, bio_top), ()):
                    if (db, code) in seen:
                        continue
                    out.append({"database": db, "code": code, "amount": cf})
                    seen.add((db, code))
        return sorted(out, key=lambda r: (r["database"], r["code"]))

    # ------------------------------------------------------------------
    # JRC side.

    def _jrc_global_cfs(self) -> dict[str, float]:
        """``jrc_flow_name_lower → global CF`` for the Water use method.

        Prefers JRC's null-``LCIAMethod_location`` row (the global /
        regionally-unspecified value). Falls back to the unweighted
        mean across regional rows when no null row exists. Only
        flows on the resource side (``FLOW_class0 == "Resources"``)
        are returned — emission-side CFs are negative and the bio3
        codes for water emissions are characterised separately by
        the inherited ``Water [air]`` rows preserved by
        ``SimaProCfFilter.SKIP_FILTER_METHODS``.
        """
        df = self.ef_cf_table.raw
        df = df[df["LCIAMethod_name"] == self.JRC_METHOD_NAME]
        df = df[df["FLOW_class0"] == "Resources"]
        if df.empty:
            return {}
        out: dict[str, float] = {}
        for name, group in df.groupby("FLOW_name"):
            key = str(name).strip().lower()
            nulls = group[group["LCIAMethod_location"].isna()]
            if not nulls.empty:
                out[key] = float(nulls["CF EF3.1"].iloc[0])
            else:
                out[key] = float(group["CF EF3.1"].astype(float).mean())
        return out

    # ------------------------------------------------------------------
    # bio3 side.

    @cached_property
    def _bio_catalog(self) -> pd.DataFrame:
        df = pd.read_parquet(self.biosphere_catalog_path)
        df = df[df["database"].isin(self.SCORED_DATABASES)].copy()
        df["_name_l"] = df["name"].astype(str).str.lower().str.strip()
        df["_top"] = df["categories"].apply(self._top_compartment)
        return df

    def _bio_index_by_name_top(self) -> dict[tuple[str, str], list[tuple[str, str]]]:
        df = self._bio_catalog
        out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for _, r in df.iterrows():
            out.setdefault((r["_name_l"], r["_top"]), []).append(
                (str(r["database"]), str(r["code"]))
            )
        return out

    @staticmethod
    def _top_compartment(cats: object) -> str:
        if cats is None:
            return ""
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if isinstance(cats, (list, tuple)) and cats:
            return str(cats[0]).strip().lower()
        return ""
