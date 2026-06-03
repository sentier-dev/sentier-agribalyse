"""``SimaProEFCfTable`` — parse the SimaPro EF 3.1 (adapted) XLSX export.

The XLSX file is the SimaPro 10.x export of the Environmental Footprint 3.1
method as adapted for the SimaPro substance database. ADEME computes its
AGRIBALYSE 3.2 reference scores against this exact CF table — so when our
backtest disagrees with the ADEME synthesis, the first thing to compare is
*these* CFs against ours.

The XLSX layout (one sheet ``Sheet1``):

* Rows 0..149 are header / metadata text.
* Each impact category begins with a header row whose first cell is
  ``Impact category`` and whose second cell is the SimaPro method name
  (e.g. ``Ionising radiation``); the third cell is the reference unit.
* Subsequent rows are CFs, with this column convention::

      A: compartment       (Air, Water, Soil, Raw, …)
      B: sub-compartment   ((unspecified), low. pop., ground-, …)
      C: flow name         (e.g. ``Radon-222``)
      D: CAS number        (zero-padded 9-digit string or empty)
      E: CF value          (float, in the method's reference unit)
      F: flow unit         (kg, kBq, m2a, …)
      G: CF unit           (e.g. ``kBq U-235 eq / kBq``)

  Blank rows separate categories.

This module reads that sheet into a flat DataFrame with one row per CF,
keyed by ``(simapro_method, compartment, sub_compartment, name)``.

No bw2 dependency. Lazily parsed and cached as a parquet next to the
source XLSX so subsequent reads are O(parquet) rather than O(XLSX).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class SimaProEFCfTable:
    """Parsed view of the SimaPro EF 3.1 (adapted) method export.

    Attributes:
        xlsx_path: Path to the SimaPro export file.
        cache_path: Where to write/read the parquet cache.
    """

    xlsx_path: Path
    cache_path: Path

    HEADER_SKIP: ClassVar[int] = 150
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "simapro_method",
        "simapro_method_unit",
        "compartment",
        "sub_compartment",
        "name",
        "cas",
        "cf",
        "flow_unit",
        "cf_unit",
    )

    @cached_property
    def df(self) -> pd.DataFrame:
        """Return the parsed CFs as a DataFrame, materialising the cache once."""
        if (
            self.cache_path.exists()
            and self.cache_path.stat().st_mtime >= self.xlsx_path.stat().st_mtime
        ):
            return pd.read_parquet(self.cache_path)
        df = self._parse_xlsx()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path)
        return df

    def _parse_xlsx(self) -> pd.DataFrame:
        # openpyxl is part of the runtime deps; import locally because this
        # parser is build-time only and we don't want to pay the cost on
        # every script that imports the ef package.
        import openpyxl

        wb = openpyxl.load_workbook(self.xlsx_path, read_only=True, data_only=True)
        ws = wb["Sheet1"]
        rows: list[dict] = []
        current_method: str | None = None
        current_unit: str | None = None
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i < self.HEADER_SKIP:
                continue
            if not row:
                continue
            a, b, c, d, e, f, g = row[:7] + (None,) * (7 - len(row))
            if a == "Impact category" and b:
                current_method = str(b)
                current_unit = str(c) if c is not None else None
                continue
            if not current_method:
                continue
            if a is None and b is None:
                continue  # blank separator
            if not isinstance(e, (int, float)):
                continue
            if a is None:
                continue
            rows.append(
                {
                    "simapro_method": current_method,
                    "simapro_method_unit": current_unit,
                    "compartment": str(a),
                    "sub_compartment": str(b) if b is not None else "",
                    "name": str(c) if c is not None else "",
                    "cas": str(d) if d is not None else "",
                    "cf": float(e),
                    "flow_unit": str(f) if f is not None else "",
                    "cf_unit": str(g) if g is not None else "",
                }
            )
        return pd.DataFrame(rows, columns=list(self.COLUMNS))

    # ------------------------------------------------------------------
    # Per-method slicing.

    METHOD_TO_OUR_KEY: ClassVar[dict[str, tuple[str, str]]] = {
        "Acidification": ("acidification", "accumulated exceedance (AE)"),
        "Climate change": ("climate change", "global warming potential (GWP100)"),
        "Climate change - Biogenic": (
            "climate change: biogenic",
            "global warming potential (GWP100)",
        ),
        "Climate change - Fossil": (
            "climate change: fossil",
            "global warming potential (GWP100)",
        ),
        "Climate change - Land use and LU change": (
            "climate change: land use and land use change",
            "global warming potential (GWP100)",
        ),
        "Ecotoxicity, freshwater": (
            "ecotoxicity: freshwater",
            "comparative toxic unit for ecosystems (CTUe)",
        ),
        "Resource use, fossils": (
            "energy resources: non-renewable",
            "abiotic depletion potential (ADP): fossil fuels",
        ),
        "Eutrophication, freshwater": (
            "eutrophication: freshwater",
            "fraction of nutrients reaching freshwater end compartment (P)",
        ),
        "Eutrophication, marine": (
            "eutrophication: marine",
            "fraction of nutrients reaching marine end compartment (N)",
        ),
        "Eutrophication, terrestrial": (
            "eutrophication: terrestrial",
            "accumulated exceedance (AE)",
        ),
        "Human toxicity, cancer": (
            "human toxicity: carcinogenic",
            "comparative toxic unit for human (CTUh)",
        ),
        "Human toxicity, non-cancer": (
            "human toxicity: non-carcinogenic",
            "comparative toxic unit for human (CTUh)",
        ),
        "Ionising radiation": (
            "ionising radiation: human health",
            "human exposure efficiency relative to u235",
        ),
        "Land use": ("land use", "soil quality index"),
        "Resource use, minerals and metals": (
            "material resources: metals/minerals",
            "abiotic depletion potential (ADP): elements (ultimate reserves)",
        ),
        "Ozone depletion": ("ozone depletion", "ozone depletion potential (ODP)"),
        "Particulate matter": (
            "particulate matter formation",
            "impact on human health",
        ),
        "Photochemical ozone formation": (
            "photochemical oxidant formation: human health",
            "tropospheric ozone concentration increase",
        ),
        "Water use": (
            "water use",
            "user deprivation potential (deprivation-weighted water consumption)",
        ),
    }

    def for_our_method(self, our_category: str, our_indicator: str) -> pd.DataFrame:
        """Return SimaPro CF rows for a method given our (category, indicator) tuple.

        SimaPro splits ``Ecotoxicity, freshwater`` and ``Human toxicity, ...``
        into ``- inorganics`` / ``- organics`` sub-methods. We aggregate
        all three (main + inorganics + organics) when the user asks for
        the parent method, because that's what gets compared against the
        single ADEME reference number.
        """
        sp_root = self._our_to_simapro(our_category, our_indicator)
        if sp_root is None:
            return self.df.iloc[0:0]
        wanted = {sp_root, f"{sp_root} - inorganics", f"{sp_root} - organics"}
        return self.df[self.df["simapro_method"].isin(wanted)].copy()

    @classmethod
    def _our_to_simapro(cls, category: str, indicator: str) -> str | None:
        for sp, key in cls.METHOD_TO_OUR_KEY.items():
            if key == (category, indicator):
                return sp
        return None
