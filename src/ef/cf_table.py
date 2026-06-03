"""``EfCfTable`` — load + collapse the EF v3.1 CF parquet."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from readers import ParquetReader


@dataclass(frozen=True)
class EfCfTable:
    """Loaded view of ``EF-LCIAMethod_CF(EF-v3.1)__lciamethods_CF.parquet``."""

    path: Path
    parquet: ParquetReader = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.parquet is None:
            object.__setattr__(self, "parquet", ParquetReader())

    @cached_property
    def raw(self) -> pd.DataFrame:
        return self.parquet.read(self.path)

    @cached_property
    def global_cfs(self) -> pd.DataFrame:
        """One CF per (LCIAMethod_name, FLOW_uuid) — prefer NULL-location else mean."""
        cf = self.raw
        null_rows = (
            cf[cf["LCIAMethod_location"].isna()]
            .drop_duplicates(subset=["LCIAMethod_name", "FLOW_uuid"])
            .copy()
        )
        null_rows["CF_global"] = null_rows["CF EF3.1"].astype(float)

        null_keys = set(zip(null_rows["LCIAMethod_name"], null_rows["FLOW_uuid"], strict=False))
        regional = cf[cf["LCIAMethod_location"].notna()].copy()
        regional["_key"] = list(
            zip(regional["LCIAMethod_name"], regional["FLOW_uuid"], strict=False)
        )
        regional = regional[~regional["_key"].isin(null_keys)]

        regional_means = regional.groupby(["LCIAMethod_name", "FLOW_uuid"], as_index=False).agg(
            FLOW_name=("FLOW_name", "first"),
            FLOW_class0=("FLOW_class0", "first"),
            FLOW_class1=("FLOW_class1", "first"),
            FLOW_class2=("FLOW_class2", "first"),
            LCIAMethod_direction=("LCIAMethod_direction", "first"),
            CF_global=("CF EF3.1", "mean"),
        )

        cols = [
            "LCIAMethod_name",
            "FLOW_uuid",
            "FLOW_name",
            "FLOW_class0",
            "FLOW_class1",
            "FLOW_class2",
            "LCIAMethod_direction",
            "CF_global",
        ]
        return pd.concat([null_rows[cols], regional_means[cols]], ignore_index=True)
