"""Unit tests for ``BiosphereCatalogAugmenter``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from matching.bio_catalog_augmenter import BiosphereCatalogAugmenter


def _write_base_catalog(path: Path) -> Path:
    rows = [
        {
            "database": "ecoinvent-3.9.1-biosphere",
            "code": "water-well-uuid",
            "name": "Water, well",
            "categories": ["natural resource", "in water"],
            "unit": "m3",
            "cas": "7732-18-5",
            "synonyms": [],
        },
        {
            "database": "ecoinvent-3.9.1-biosphere",
            "code": "water-lake-uuid",
            "name": "Water, lake",
            "categories": ["natural resource", "in water"],
            "unit": "m3",
            "cas": "7732-18-5",
            "synonyms": [],
        },
        {
            "database": "biosphere3",
            "code": "co2-uuid",
            "name": "Carbon dioxide",
            "categories": ["air"],
            "unit": "kg",
            "cas": "124-38-9",
            "synonyms": [],
        },
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def base_catalog_path(tmp_path: Path) -> Path:
    return _write_base_catalog(tmp_path / "biosphere_catalog.parquet")


class TestBiosphereCatalogAugmenter:
    def test_augment_appends_one_synthetic_row_per_unique_region(self, base_catalog_path: Path):
        aug = BiosphereCatalogAugmenter(catalog_path=base_catalog_path)
        n = aug.augment(
            [
                ("ecoinvent-3.9.1-biosphere", "water-well-uuid", "CN"),
                ("ecoinvent-3.9.1-biosphere", "water-well-uuid", "FR"),
                ("ecoinvent-3.9.1-biosphere", "water-lake-uuid", "ES"),
            ]
        )
        assert n == 3
        df = pd.read_parquet(base_catalog_path)
        # base + 3 synthetic
        assert len(df) == 6
        synth = df[df["region"] != ""].sort_values(["code"]).reset_index(drop=True)
        assert list(synth["code"]) == [
            "water-lake-uuid@ES",
            "water-well-uuid@CN",
            "water-well-uuid@FR",
        ]
        assert list(synth["base_code"]) == [
            "water-lake-uuid",
            "water-well-uuid",
            "water-well-uuid",
        ]
        # Synthetic rows inherit name from the base row.
        assert list(synth["name"]) == [
            "Water, lake",
            "Water, well",
            "Water, well",
        ]

    def test_augment_is_idempotent(self, base_catalog_path: Path):
        aug = BiosphereCatalogAugmenter(catalog_path=base_catalog_path)
        first = aug.augment([("ecoinvent-3.9.1-biosphere", "water-well-uuid", "CN")])
        assert first == 1
        second = aug.augment([("ecoinvent-3.9.1-biosphere", "water-well-uuid", "CN")])
        assert second == 0

    def test_augment_skips_non_allowlisted_databases(self, base_catalog_path: Path):
        aug = BiosphereCatalogAugmenter(catalog_path=base_catalog_path)
        n = aug.augment([("ef", "ef-flow-uuid", "FR")])
        assert n == 0

    def test_augment_skips_unknown_base_codes(self, base_catalog_path: Path):
        aug = BiosphereCatalogAugmenter(catalog_path=base_catalog_path)
        n = aug.augment([("ecoinvent-3.9.1-biosphere", "no-such-base", "FR")])
        assert n == 0

    def test_augment_from_sp_data_collects_synthetic_codes(self, base_catalog_path: Path):
        aug = BiosphereCatalogAugmenter(catalog_path=base_catalog_path)
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Water, well, CN",
                        "input": ("ecoinvent-3.9.1-biosphere", "water-well-uuid@CN"),
                    },
                    {
                        "type": "biosphere",
                        "name": "Carbon dioxide",
                        "input": ("biosphere3", "co2-uuid"),  # no suffix
                    },
                    {
                        "type": "technosphere",  # ignored
                        "input": ("x", "y@FR"),
                    },
                ],
            }
        ]
        n = aug.augment_from_sp_data(sp_data)
        assert n == 1
        df = pd.read_parquet(base_catalog_path)
        synth = df[df["region"] != ""]
        assert list(synth["code"]) == ["water-well-uuid@CN"]
