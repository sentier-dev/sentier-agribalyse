"""Unit tests for the file-format readers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from core import ParquetCache
from readers import (
    GzJsonReader,
    JsonOrGzJsonReader,
    JsonReader,
    ParquetReader,
    RandonneurDataLoader,
    XlsxReader,
)


class TestJsonReader:
    def test_reads_utf8_json(self, tmp_path: Path):
        path = tmp_path / "data.json"
        path.write_text('{"a": 1, "b": "héllo"}', encoding="utf-8")
        assert JsonReader().read(path) == {"a": 1, "b": "héllo"}

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            JsonReader().read(tmp_path / "ghost.json")

    def test_accepts_pathlike(self, tmp_path: Path):
        path = tmp_path / "d.json"
        path.write_text("[1, 2, 3]")
        assert JsonReader().read(str(path)) == [1, 2, 3]


class TestGzJsonReader:
    def test_reads_gzipped_json(self, tmp_path: Path):
        path = tmp_path / "data.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"x": [1, 2, 3]}, f)
        assert GzJsonReader().read(path) == {"x": [1, 2, 3]}


class TestJsonOrGzJsonReader:
    @pytest.fixture
    def reader(self):
        return JsonOrGzJsonReader()

    def test_dispatches_plain_json(self, tmp_path: Path, reader):
        path = tmp_path / "d.json"
        path.write_text('{"a": 1}')
        assert reader.read(path) == {"a": 1}

    def test_dispatches_gzipped(self, tmp_path: Path, reader):
        path = tmp_path / "d.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"a": 1}, f)
        assert reader.read(path) == {"a": 1}


class TestParquetReader:
    def test_round_trips_a_dataframe(self, tmp_path: Path):
        path = tmp_path / "df.parquet"
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(path, index=False)
        df = ParquetReader().read(path)
        assert df["a"].tolist() == [1, 2]
        assert df["b"].tolist() == ["x", "y"]


class TestXlsxReader:
    def test_xlsx_reader_uses_supplied_cache(self, tmp_path: Path):
        path = tmp_path / "wb.xlsx"
        pd.DataFrame({"a": [1, 2]}).to_excel(path, index=False)
        cache = ParquetCache(cache_dir=tmp_path / "cache")
        reader = XlsxReader(cache=cache)

        df = reader.read(path)
        assert df["a"].tolist() == [1, 2]
        assert (tmp_path / "cache" / "wb.parquet").exists()


class TestRandonneurDataLoader:
    def test_labels_proxies_registry_keys(self, monkeypatch):
        loader = RandonneurDataLoader()
        fake_registry = {
            "label-A": {"name": "label-A", "filename": "a.json"},
            "label-B": {"name": "label-B", "filename": "b.json.gz"},
        }
        # Inject a registry-shaped object via cached_property override.
        object.__setattr__(loader, "_registry", _FakeRegistry(fake_registry))
        assert sorted(loader.labels()) == ["label-A", "label-B"]
        assert loader.has("label-A") is True
        assert loader.has("missing") is False
        assert loader.metadata("label-A")["filename"] == "a.json"

    def test_load_returns_get_file_payload(self, monkeypatch):
        loader = RandonneurDataLoader()
        payload = {"data": {"update": [{"source": {"name": "x"}, "target": {"name": "y"}}]}}
        registry = _FakeRegistry({"pkg": {"name": "pkg"}}, payloads={"pkg": payload})
        object.__setattr__(loader, "_registry", registry)
        assert loader.load("pkg") == payload


class _FakeRegistry(dict):
    """Minimal stand-in for ``randonneur_data.Registry``."""

    def __init__(self, items: dict, payloads: dict | None = None):
        super().__init__(items)
        self._payloads = payloads or {}
        self.data_dir = "."

    def get_file(self, label):
        return self._payloads.get(label, {})
