"""Unit tests for ``core.parquet_cache.ParquetCache``."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from core import ParquetCache


def _write_xlsx(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


class TestParquetPathDerivation:
    def test_default_cache_dir_is_xlsx_sibling(self, tmp_path: Path):
        cache = ParquetCache(cache_dir=None)
        target = cache.parquet_path_for(tmp_path / "wb.xlsx")
        assert target == tmp_path / "wb.parquet"

    def test_explicit_cache_dir_redirects_output(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache = ParquetCache(cache_dir=cache_dir)
        target = cache.parquet_path_for(tmp_path / "wb.xlsx")
        assert target == cache_dir / "wb.parquet"

    def test_default_sheet_name_does_not_alter_stem(self, tmp_path: Path):
        cache = ParquetCache(cache_dir=tmp_path)
        assert cache.parquet_path_for(tmp_path / "wb.xlsx", sheet_name="Sheet1") == (
            tmp_path / "wb.parquet"
        )

    def test_named_sheet_name_appended_to_stem(self, tmp_path: Path):
        cache = ParquetCache(cache_dir=tmp_path)
        assert cache.parquet_path_for(tmp_path / "wb.xlsx", sheet_name="ExtraSheet") == (
            tmp_path / "wb__ExtraSheet.parquet"
        )


class TestReadExcelHitMissBehavior:
    @pytest.fixture
    def workbook(self, tmp_path: Path) -> Path:
        path = tmp_path / "src" / "wb.xlsx"
        _write_xlsx(path, pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        return path

    def test_first_read_writes_parquet_cache(self, workbook: Path, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache = ParquetCache(cache_dir=cache_dir)
        df = cache.read_excel(workbook)
        assert list(df.columns) == ["a", "b"]
        assert (cache_dir / "wb.parquet").exists()

    def test_second_read_serves_from_cache_when_xlsx_unchanged(
        self, workbook: Path, tmp_path: Path, monkeypatch
    ):
        cache_dir = tmp_path / "cache"
        cache = ParquetCache(cache_dir=cache_dir)
        cache.read_excel(workbook)
        # Tamper with the parquet so we can detect a re-parse vs. a reuse.
        parquet = cache_dir / "wb.parquet"
        pd.DataFrame({"a": [42], "b": ["sentinel"]}).to_parquet(parquet)
        # bump parquet mtime above xlsx mtime so it is a hit
        future = workbook.stat().st_mtime + 60
        os.utime(parquet, (future, future))

        df = cache.read_excel(workbook)
        assert df["b"].tolist() == ["sentinel"]  # served from parquet, not xlsx

    def test_cache_invalidates_when_xlsx_is_newer(self, workbook: Path, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache = ParquetCache(cache_dir=cache_dir)
        cache.read_excel(workbook)

        parquet = cache_dir / "wb.parquet"
        pd.DataFrame({"a": [42], "b": ["should be replaced"]}).to_parquet(parquet)
        # Make parquet older than xlsx — this should force a re-parse.
        past = workbook.stat().st_mtime - 60
        os.utime(parquet, (past, past))

        df = cache.read_excel(workbook)
        assert "sentinel" not in df["b"].tolist()
        assert df["a"].tolist() == [1, 2, 3]

    def test_force_rebuild_bypasses_cache_even_if_fresh(self, workbook: Path, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache = ParquetCache(cache_dir=cache_dir)
        cache.read_excel(workbook)
        parquet = cache_dir / "wb.parquet"
        pd.DataFrame({"a": [42], "b": ["old"]}).to_parquet(parquet)
        future = workbook.stat().st_mtime + 60
        os.utime(parquet, (future, future))

        df = cache.read_excel(workbook, force_rebuild=True)
        assert df["a"].tolist() == [1, 2, 3]
        # Cache was overwritten with re-parsed content.
        assert pd.read_parquet(parquet)["a"].tolist() == [1, 2, 3]

    def test_named_sheet_creates_distinct_parquet(self, tmp_path: Path):
        path = tmp_path / "wb.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path) as w:
            pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="One", index=False)
            pd.DataFrame({"b": [2]}).to_excel(w, sheet_name="Two", index=False)

        cache = ParquetCache(cache_dir=tmp_path / "cache")
        df_one = cache.read_excel(path, sheet_name="One")
        df_two = cache.read_excel(path, sheet_name="Two")
        assert list(df_one.columns) == ["a"]
        assert list(df_two.columns) == ["b"]
        assert (tmp_path / "cache" / "wb__One.parquet").exists()
        assert (tmp_path / "cache" / "wb__Two.parquet").exists()

    def test_read_excel_creates_cache_parent_dir(self, workbook: Path, tmp_path: Path):
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        cache = ParquetCache(cache_dir=cache_dir)
        cache.read_excel(workbook)
        assert (cache_dir / "wb.parquet").exists()

    def test_missing_xlsx_raises_filesystem_error(self, tmp_path: Path):
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        with pytest.raises(FileNotFoundError):
            cache.read_excel(tmp_path / "ghost.xlsx")


class TestParquetCacheIsFrozen:
    def test_cannot_mutate_cache_dir(self, tmp_path: Path):
        cache = ParquetCache(cache_dir=tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cache.cache_dir = tmp_path / "other"  # type: ignore[misc]
