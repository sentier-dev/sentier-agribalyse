"""``EfCfTargetIndexSource`` — feeds ``target_index_ef.parquet``.

Reads the EF v3.1 CF parquet and emits one row per EF flow with: code,
name, unit, top bucket, categories, CAS. The index is consumed by the
matcher for tier-7 generic EF lookup and tier-5 CAS disambiguation
(fix 1.o).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from domain import Bucket, EfTargetIndexRow
from readers import ParquetReader


@dataclass(frozen=True)
class EfCfTargetIndexSource:
    """Read the EF CF parquet and produce one row per distinct flow UUID."""

    path: Path
    parquet: ParquetReader = None  # type: ignore[assignment]
    default_unit: str = "kilogram"
    provenance: str = "ef.cf-parquet.target-index"

    def __post_init__(self) -> None:
        if self.parquet is None:
            object.__setattr__(self, "parquet", ParquetReader())

    def read(self) -> list[EfTargetIndexRow]:
        if not Path(self.path).exists():
            return []
        cf = self.parquet.read(self.path)
        # One row per distinct UUID. The parquet has per-(method, location)
        # rows; we collapse by FLOW_uuid for the target index.
        flows = cf.drop_duplicates(subset=["FLOW_uuid"])[
            ["FLOW_uuid", "FLOW_name", "FLOW_class0", "FLOW_class1", "FLOW_class2"]
        ]
        out: list[EfTargetIndexRow] = []
        for _, r in flows.iterrows():
            cats = tuple(
                str(c).strip()
                for c in (r["FLOW_class0"], r["FLOW_class1"], r["FLOW_class2"])
                if pd.notna(c)
            )
            out.append(
                EfTargetIndexRow(
                    code=str(r["FLOW_uuid"]),
                    name=str(r["FLOW_name"]),
                    unit=self.default_unit,
                    bucket=Bucket.from_categories(cats),
                    categories=cats,
                    cas=None,
                )
            )
        return out
