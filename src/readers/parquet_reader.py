"""Parquet reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ParquetReader:
    """Read a parquet file into a DataFrame."""

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(Path(path))
