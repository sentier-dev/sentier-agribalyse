"""``SpRegionalWaterCfLoader`` — load per-region AWARE deprivation CFs.

Reads ``cache/simapro-EF31-adapted-cfs.parquet`` (the parsed SimaPro
EF v3.1 (adapted) export ADEME uses to compute the AGRIBALYSE 3.2
reference scores) and emits one row per ``(base_name, compartment,
sub_compartment, region, cf, flow_unit)`` tuple for the **Water use**
method.

Only the ``flow_unit = "m3"`` rows are kept. SimaPro carries two
parallel CF tables for every water flow — one in cubic metres, one in
kilograms (with CF re-scaled by 1/1000) — and our matrix-side
amounts are uniformly in m³, so the kg variant would 1000x under-count
on every linked edge. The kg variant was the root cause of the
``WaterUseBio3Bridge`` 337x over-count documented in the spec. This
loader hard-rejects it.

Regional suffixes are detected via
:class:`matching.regional_suffix.RegionalSuffixParser`, the same
parser the matcher uses, so the (base_name, region) tuples that come
out align with the synthetic codes the augmented biosphere catalog
carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pandas as pd

from matching.regional_suffix import RegionalSuffixParser


@dataclass(frozen=True)
class SpRegionalWaterCfLoader:
    """Per-region water-use CFs from the SimaPro EF v3.1 (adapted) parquet."""

    cache_path: Path
    parser: RegionalSuffixParser = field(default_factory=RegionalSuffixParser)

    SP_METHOD: ClassVar[str] = "Water use"
    FLOW_UNIT_M3: ClassVar[str] = "m3"

    # Suffixes SimaPro appends to disambiguate kg vs m³ variants on the
    # same flow (``Water, cooling, unspecified natural origin/kg`` vs
    # ``/m3``). Stripped from ``base_name`` before parsing the region.
    _UNIT_SUFFIX_PATTERN: ClassVar[str] = r"/(?:kg|m3)\s*$"

    @cached_property
    def df(self) -> pd.DataFrame:
        """Return cols: ``base_name``, ``compartment``, ``sub_compartment``,
        ``region``, ``cf``, ``flow_unit``.

        One row per emitted (base_name, compartment, sub_compartment,
        region) tuple. ``region`` is the empty string for the global /
        non-regional rows that SimaPro keeps alongside the per-country
        variants — callers can use those as the global fallback CF.
        """
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"SimaPro adapted CF parquet not found: {self.cache_path}. "
                f"Build via the ``EfCfTable`` parse path or re-run the "
                f"`dds-build-method-cfs-registry` CLI to materialise it."
            )
        raw = pd.read_parquet(self.cache_path)
        water = raw[raw["simapro_method"] == self.SP_METHOD].copy()
        if water.empty:
            return pd.DataFrame(
                columns=[
                    "base_name",
                    "compartment",
                    "sub_compartment",
                    "region",
                    "cf",
                    "flow_unit",
                ]
            )
        # Hard filter to m³ — kg variants are exactly 1/1000 the m³ CF
        # and our matrix carries amounts in m³, so emitting kg rows
        # would silently under-count by 1000x on linked edges.
        water = water[water["flow_unit"].astype(str) == self.FLOW_UNIT_M3].copy()
        # Strip the SimaPro unit suffix (``Water, cooling, unspecified
        # natural origin/m3`` → ``Water, cooling, unspecified natural
        # origin``). Strip even though we only keep m³ rows so the base
        # name stays clean for downstream catalog joins.
        water["name_clean"] = (
            water["name"].astype(str).str.replace(self._UNIT_SUFFIX_PATTERN, "", regex=True)
        )
        # Split (base_name, region) deterministically through the shared
        # parser. Rows whose trailing token is not a recognised region
        # (e.g. ``BR-Mid-western grid``, ``Canada without Quebec``) keep
        # the full name as base and get region="" so they are usable as
        # global fallbacks but never as regional CFs.
        parsed = water["name_clean"].apply(self.parser.parse)
        water["base_name"] = parsed.map(lambda t: t[0])
        water["region"] = parsed.map(lambda t: t[1])
        water["sub_compartment"] = water["sub_compartment"].astype(str)
        water["compartment"] = water["compartment"].astype(str)
        water["cf"] = water["cf"].astype(float)
        cols = [
            "base_name",
            "compartment",
            "sub_compartment",
            "region",
            "cf",
            "flow_unit",
        ]
        out = water[cols].copy()
        # Deterministic ordering for testability.
        out = out.sort_values(by=cols, kind="mergesort").reset_index(drop=True)
        return out

    def cf_for(
        self,
        *,
        base_name: str,
        region: str,
        compartment: str,
        sub_compartment: str,
    ) -> float | None:
        """Look up the CF for one (base, region, compartment, sub) tuple.

        Returns ``None`` when no SimaPro row exists. Callers can fall
        back to the global / inherited CF in that case.
        """
        m = self._index.get((base_name, region, compartment, sub_compartment))
        return m

    @cached_property
    def _index(self) -> dict[tuple[str, str, str, str], float]:
        return {
            (row.base_name, row.region, row.compartment, row.sub_compartment): row.cf
            for row in self.df.itertuples(index=False)
        }
