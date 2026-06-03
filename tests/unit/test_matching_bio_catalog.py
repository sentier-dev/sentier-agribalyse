"""Unit tests for ``matching.BiosphereCatalog`` — parquet-backed biosphere index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from domain import Bucket
from matching.bio_catalog import BioFlowRef, BiosphereCatalog


def _ref(**kw):
    base = dict(
        db="biosphere3",
        code="abc",
        name="Carbon dioxide",
        unit="kg",
        bucket=Bucket.AIR,
        cas=None,
    )
    base.update(kw)
    return BioFlowRef(**base)


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


class TestBiosphereCatalogIndexes:
    def test_lookup_name_bucket_returns_matching_flows(self):
        cat = BiosphereCatalog(
            db_names=("biosphere3",),
            flows=(
                _ref(code="a", name="Carbon dioxide"),
                _ref(code="b", name="methane", bucket=Bucket.AIR),
            ),
        )
        hits = cat.lookup_name_bucket("biosphere3", "carbon dioxide", Bucket.AIR)
        assert [h.code for h in hits] == ["a"]

    def test_lookup_name_bucket_is_case_insensitive(self):
        cat = BiosphereCatalog(
            db_names=("biosphere3",),
            flows=(_ref(name="Carbon Dioxide"),),
        )
        assert cat.lookup_name_bucket("biosphere3", "carbon dioxide", Bucket.AIR)

    def test_get_returns_none_for_unknown_keys(self):
        cat = BiosphereCatalog(
            db_names=("biosphere3",),
            flows=(_ref(code="abc"),),
        )
        assert cat.get("biosphere3", "abc").code == "abc"
        assert cat.get("biosphere3", "missing") is None
        assert cat.get("ghost-db", "abc") is None

    def test_lookup_cas_returns_flows_keyed_on_cas_within_db(self):
        cat = BiosphereCatalog(
            db_names=("biosphere3", "ef"),
            flows=(
                _ref(db="biosphere3", code="a", cas="124-38-9"),
                _ref(db="ef", code="ef-a", cas="124-38-9"),
                _ref(db="biosphere3", code="b", cas=None),
            ),
        )
        bio_hits = cat.lookup_cas("biosphere3", "124-38-9")
        assert [h.code for h in bio_hits] == ["a"]
        assert cat.lookup_cas("biosphere3", "  124-38-9  ")  # whitespace-tolerant


class TestBiosphereCatalogLoad:
    def test_load_reads_parquet_into_flows(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "cat.parquet",
            [
                {
                    "database": "biosphere3",
                    "code": "a",
                    "name": "Carbon dioxide",
                    "categories": ["air"],
                    "unit": "kg",
                    "cas": "124-38-9",
                    "synonyms": [],
                }
            ],
        )
        cat = BiosphereCatalog.load(path)
        assert cat.db_names == ("biosphere3",)
        assert len(cat.flows) == 1
        assert cat.flows[0].cas == "124-38-9"
        assert cat.flows[0].bucket == Bucket.AIR

    def test_load_filters_by_db_names(self, tmp_path: Path):
        path = _write_parquet(
            tmp_path / "cat.parquet",
            [
                {
                    "database": "biosphere3",
                    "code": "a",
                    "name": "X",
                    "categories": ["air"],
                    "unit": "kg",
                    "cas": None,
                    "synonyms": [],
                },
                {
                    "database": "ef",
                    "code": "b",
                    "name": "Y",
                    "categories": ["water"],
                    "unit": "kg",
                    "cas": None,
                    "synonyms": [],
                },
            ],
        )
        cat = BiosphereCatalog.load(path, db_names=("biosphere3",))
        assert cat.db_names == ("biosphere3",)
        assert {f.code for f in cat.flows} == {"a"}

    def test_load_raises_when_path_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="biosphere catalog parquet"):
            BiosphereCatalog.load(tmp_path / "ghost.parquet")
