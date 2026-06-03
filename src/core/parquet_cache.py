"""Parquet-cached xlsx reader. Class-based, takes ``cache_dir`` in constructor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.logging import Logging


@dataclass(frozen=True)
class ParquetCache:
    """Read xlsx files via a sibling parquet cache.

    ``cache_dir=None`` → caches sit next to the source xlsx (legacy behaviour).
    """

    cache_dir: Path | None = None

    @property
    def _log(self):
        return Logging.get(__name__)

    def parquet_path_for(self, xlsx_path: Path, sheet_name: str | None = None) -> Path:
        stem = xlsx_path.stem
        if sheet_name and sheet_name != "Sheet1":
            stem = f"{stem}__{sheet_name}"
        parent = Path(self.cache_dir) if self.cache_dir is not None else xlsx_path.parent
        return parent / f"{stem}.parquet"

    def read_excel(
        self,
        xlsx_path: Path,
        *,
        sheet_name: str | None = None,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        xlsx_path = Path(xlsx_path)
        parquet_path = self.parquet_path_for(xlsx_path, sheet_name)

        if (
            not force_rebuild
            and parquet_path.exists()
            and parquet_path.stat().st_mtime >= xlsx_path.stat().st_mtime
        ):
            t0 = time.time()
            df = pd.read_parquet(parquet_path)
            self._log.info(
                "parquet_cache.hit",
                path=parquet_path.name,
                rows=len(df),
                elapsed_s=round(time.time() - t0, 3),
            )
            return df

        self._log.info(
            "parquet_cache.miss",
            xlsx=xlsx_path.name,
            sheet=sheet_name,
            msg="parsing xlsx (one-time conversion to parquet)",
        )
        t0 = time.time()
        kwargs: dict = {}
        if sheet_name is not None:
            kwargs["sheet_name"] = sheet_name
        df = pd.read_excel(xlsx_path, **kwargs)
        parse_elapsed = time.time() - t0

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        self._log.info(
            "parquet_cache.written",
            parquet=parquet_path.name,
            rows=len(df),
            parse_s=round(parse_elapsed, 1),
            cache_mb=round(parquet_path.stat().st_size / 1e6, 1),
        )
        return df
