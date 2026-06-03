"""``BacktestPass1Emitter`` — write ``dashboard/backtest_pass1.csv``.

Translates the long-form method columns produced by ``BacktestPipeline``
(``"climate change"``, ``"climate change: biogenic"``, ...) into the 19
short IDs the dashboard HTML expects (``"climate"``, ``"cc_bio"``, ...).
Replaces the previously manual regeneration of
``dashboard/backtest_pass1.csv`` from ``dashboard/backtest/diff_pct.parquet``.

Only rows where ``scores_df["mapped"]`` is True are emitted — the
dashboard's "resolution=mapped" filter excludes unmapped reference rows
that carry no computed diff.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class BacktestPass1Emitter:
    out_path: Path

    LONG_TO_SHORT: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "climate change": "climate",
            "ozone depletion": "ozone",
            "ionising radiation: human health": "radiation",
            "photochemical oxidant formation: human health": "photo_ox",
            "particulate matter formation": "pm",
            "human toxicity: non-carcinogenic": "ht_nc",
            "human toxicity: carcinogenic": "ht_c",
            "acidification": "acid",
            "eutrophication: freshwater": "e_fw",
            "eutrophication: marine": "e_m",
            "eutrophication: terrestrial": "e_t",
            "ecotoxicity: freshwater": "ecotox",
            "land use": "land",
            "water use": "water",
            "energy resources: non-renewable": "energy",
            "material resources: metals/minerals": "mater",
            "climate change: biogenic": "cc_bio",
            "climate change: fossil": "cc_fos",
            "climate change: land use and land use change": "cc_luc",
        }
    )

    SHORT_ORDER: ClassVar[tuple[str, ...]] = (
        "climate",
        "ozone",
        "radiation",
        "photo_ox",
        "pm",
        "ht_nc",
        "ht_c",
        "acid",
        "e_fw",
        "e_m",
        "e_t",
        "ecotox",
        "land",
        "water",
        "energy",
        "mater",
        "cc_bio",
        "cc_fos",
        "cc_luc",
    )

    HEADER: ClassVar[tuple[str, ...]] = (
        "code",
        "name",
        "mapped_to",
        "type",
        "resolution",
    )

    def write(self, scores_df: pd.DataFrame, diff_pct_df: pd.DataFrame) -> Path:
        if "mapped" in scores_df.columns:
            mapped_codes = scores_df.loc[scores_df["mapped"].astype(bool), "Code AGB"]
            mask = diff_pct_df["Code AGB"].isin(set(mapped_codes.astype(str)))
            df = diff_pct_df.loc[mask].copy()
        else:
            df = diff_pct_df.copy()

        out_cols: list[str] = list(self.HEADER) + list(self.SHORT_ORDER)
        out_rows: list[dict[str, object]] = []
        for _, row in df.iterrows():
            record: dict[str, object] = {
                "code": row.get("Code AGB", ""),
                "name": row.get("Nom du Produit", ""),
                "mapped_to": row.get("LCI Name", ""),
                "type": "",
                "resolution": "mapped",
            }
            for long, short in self.LONG_TO_SHORT.items():
                val = row.get(long) if long in df.columns else None
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    record[short] = ""
                else:
                    record[short] = val
            out_rows.append(record)
        out_df = pd.DataFrame(out_rows, columns=out_cols)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(self.out_path, index=False)
        return self.out_path
