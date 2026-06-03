"""Cached xlsx reader. Composes ParquetCache from core/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.parquet_cache import ParquetCache


@dataclass(frozen=True)
class XlsxReader:
    """Read xlsx via parquet cache. Pass an instance of ``ParquetCache``.

    A reader bound to one cache directory is reusable across many xlsx
    files. The cache key is ``(xlsx_path, sheet_name)``.
    """

    cache: ParquetCache

    def read(
        self,
        path: Path,
        sheet_name: str | None = None,
        *,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        return self.cache.read_excel(
            Path(path),
            sheet_name=sheet_name,
            force_rebuild=force_rebuild,
        )
