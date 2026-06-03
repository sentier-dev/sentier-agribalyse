"""Tests for ``WaterResourceCfAugmenter`` — emit JRC water-use CFs onto
bio3 water-resource flow codes that the bw2io snapshot leaves
uncharacterised.

Background: JRC's EF v3.1 Water use method ships per-region CFs for
``freshwater``, ``ground water``, ``lake water``, ``river water``,
``Water to Cooling``, ``Water to turbine`` (1 470 region rows per uuid).
The bw2io snapshot drops them when collapsing to bio3 codes — our matrix
inherits **zero** resource-side water-use CFs, so AGB-side activities
that consume ``Water, river`` etc. score 0 on water use even when the
ADEME reference is non-zero.

This augmenter fills the gap with the JRC null-region (global) CF.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ef.cf_table import EfCfTable
from ef.water_resource_augmenter import WaterResourceCfAugmenter


def _write_jrc_cf(tmp_path: Path, rows: list[dict]) -> EfCfTable:
    cols = [
        "FLOW_uuid",
        "FLOW_name",
        "LCIAMethod_uuid EF3.1",
        "LCIAMethod_name",
        "CF EF3.1",
        "LCIAMethod_location",
        "FLOW_class0",
        "FLOW_class1",
        "FLOW_class2",
        "LCIAMethod_derivation",
        "LCIAMethod_direction",
    ]
    df = pd.DataFrame(rows, columns=cols)
    p = tmp_path / "jrc_cf.parquet"
    df.to_parquet(p)
    return EfCfTable(path=p)


def _write_bio_catalog(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(
        rows,
        columns=["database", "code", "name", "categories", "unit", "cas", "synonyms"],
    )
    p = tmp_path / "biosphere_catalog.parquet"
    df.to_parquet(p)
    return p


def _jrc_water_row(name: str, uuid: str, cf: float, location=None) -> dict:
    return {
        "FLOW_uuid": uuid,
        "FLOW_name": name,
        "LCIAMethod_uuid EF3.1": "method-water-use",
        "LCIAMethod_name": "Water use",
        "CF EF3.1": cf,
        "LCIAMethod_location": location,
        "FLOW_class0": "Resources",
        "FLOW_class1": "Resources from water",
        "FLOW_class2": "Renewable material resources from water",
        "LCIAMethod_derivation": None,
        "LCIAMethod_direction": "input",
    }


def test_augment_emits_river_water_cf_onto_bio3_water_river(tmp_path: Path):
    """JRC ``river water`` at null-region CF=42.95 must land on bio3
    ``Water, river [natural resource, in water]`` for both bio3 + ecoinvent
    DBs. This is the canonical case — AGB activities emit ``Water, river``
    and ADEME's water-use score uses the same global 42.95 CF.
    """
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _jrc_water_row("river water", "uuid-river", 42.95, location=None),
            _jrc_water_row("river water", "uuid-river", 6.98, location="FR"),
        ],
    )
    bio = _write_bio_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "8c75e7ab-8ab8-41e4-b394-c166ff5b050d",
                "name": "Water, river",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "ecoinvent-3.9.1-biosphere",
                "code": "8c75e7ab-8ab8-41e4-b394-c166ff5b050d",
                "name": "Water, river",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    aug = WaterResourceCfAugmenter(ef_cf_table=jrc, biosphere_catalog_path=bio)

    rows = aug.cf_rows()
    assert len(rows) == 2, rows
    assert all(r["amount"] == pytest.approx(42.95) for r in rows)
    dbs = sorted(r["database"] for r in rows)
    assert dbs == ["biosphere3", "ecoinvent-3.9.1-biosphere"]
    assert all(r["code"] == "8c75e7ab-8ab8-41e4-b394-c166ff5b050d" for r in rows)


def test_augment_falls_back_to_mean_when_no_null_region(tmp_path: Path):
    """If JRC has no null-region row, the global CF is the mean across
    all regional CFs — the production-weighted approximation.
    """
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _jrc_water_row("river water", "uuid-r", 10.0, location="FR"),
            _jrc_water_row("river water", "uuid-r", 30.0, location="US"),
        ],
    )
    bio = _write_bio_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "8c75e7ab-8ab8-41e4-b394-c166ff5b050d",
                "name": "Water, river",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    aug = WaterResourceCfAugmenter(ef_cf_table=jrc, biosphere_catalog_path=bio)

    rows = aug.cf_rows()
    assert len(rows) == 1
    assert rows[0]["amount"] == pytest.approx(20.0)


def test_augment_emits_lake_ground_unspecified_and_cooling_turbine(tmp_path: Path):
    """All six unambiguous JRC names land on their bio3 counterparts:
    ``lake water`` → ``Water, lake``,
    ``ground water`` → ``Water, well, in ground``,
    ``water`` → ``Water, unspecified natural origin`` (every sub-comp),
    ``Water to Cooling`` → ``Water, cooling, unspecified natural origin``,
    ``Water to turbine`` → ``Water, turbine use, unspecified natural origin``.
    """
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _jrc_water_row("lake water", "uuid-l", 42.95),
            _jrc_water_row("ground water", "uuid-g", 42.95),
            _jrc_water_row("water", "uuid-w", 37.8),
            _jrc_water_row("Water to Cooling", "uuid-wc", 37.8),
            _jrc_water_row("Water to turbine", "uuid-wt", 42.95),
        ],
    )
    bio = _write_bio_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "1acb026e-9de6-48fe-9e0d-be4d24125bbc",
                "name": "Water, lake",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "67c40aae-d403-464d-9649-c12695e43ad8",
                "name": "Water, well, in ground",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "831f249e-53f2-49cf-a93c-7cee105f048e",
                "name": "Water, unspecified natural origin",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "478e8437-1c21-4032-8438-872a6b5ddcdf",
                "name": "Water, unspecified natural origin",
                "categories": ["natural resource", "in ground"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "fc1c42ce-a759-49fa-b987-f1ec5e503db1",
                "name": "Water, cooling, unspecified natural origin",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "8c1494a5-4987-4715-aa2d-1908c495f4eb",
                "name": "Water, turbine use, unspecified natural origin",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    aug = WaterResourceCfAugmenter(ef_cf_table=jrc, biosphere_catalog_path=bio)

    rows = aug.cf_rows()
    by_code = {r["code"]: r["amount"] for r in rows}
    assert by_code["1acb026e-9de6-48fe-9e0d-be4d24125bbc"] == pytest.approx(42.95)
    assert by_code["67c40aae-d403-464d-9649-c12695e43ad8"] == pytest.approx(42.95)
    # Both sub-comps of "Water, unspecified natural origin" emit the
    # generic JRC ``water`` CF.
    assert by_code["831f249e-53f2-49cf-a93c-7cee105f048e"] == pytest.approx(37.8)
    assert by_code["478e8437-1c21-4032-8438-872a6b5ddcdf"] == pytest.approx(37.8)
    assert by_code["fc1c42ce-a759-49fa-b987-f1ec5e503db1"] == pytest.approx(37.8)
    assert by_code["8c1494a5-4987-4715-aa2d-1908c495f4eb"] == pytest.approx(42.95)


def test_augment_skips_unmapped_jrc_flow_names(tmp_path: Path):
    """JRC carries ``freshwater`` and ``Water`` (emission side) which do
    not have unambiguous bio3 resource-flow targets. The augmenter must
    skip them rather than guess — risk of double-counting against the
    existing inherited ``Water [air]`` rows is too high.
    """
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _jrc_water_row("freshwater", "uuid-fw", 42.95),
            {
                "FLOW_uuid": "uuid-water-em",
                "FLOW_name": "Water",
                "LCIAMethod_uuid EF3.1": "method-water-use",
                "LCIAMethod_name": "Water use",
                "CF EF3.1": -42.95,
                "LCIAMethod_location": None,
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to water",
                "FLOW_class2": "Emissions to fresh water",
                "LCIAMethod_derivation": None,
                "LCIAMethod_direction": "output",
            },
        ],
    )
    bio = _write_bio_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "8c75e7ab-8ab8-41e4-b394-c166ff5b050d",
                "name": "Water, river",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    aug = WaterResourceCfAugmenter(ef_cf_table=jrc, biosphere_catalog_path=bio)

    assert aug.cf_rows() == []


def test_augment_skips_when_no_matching_bio_code(tmp_path: Path):
    """JRC has the CF but bio3 lacks the code → silently skip."""
    jrc = _write_jrc_cf(
        tmp_path,
        [_jrc_water_row("lake water", "uuid-l", 42.95)],
    )
    bio = _write_bio_catalog(tmp_path, [])
    aug = WaterResourceCfAugmenter(ef_cf_table=jrc, biosphere_catalog_path=bio)

    assert aug.cf_rows() == []


def test_augment_returns_deterministic_order(tmp_path: Path):
    """Output rows sort by (database, code) so registry parquets are
    reproducible across builds.
    """
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _jrc_water_row("river water", "uuid-r", 42.95),
            _jrc_water_row("lake water", "uuid-l", 42.95),
        ],
    )
    bio = _write_bio_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "z-river-uuid",
                "name": "Water, river",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "a-lake-uuid",
                "name": "Water, lake",
                "categories": ["natural resource", "in water"],
                "unit": "m3",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    aug = WaterResourceCfAugmenter(ef_cf_table=jrc, biosphere_catalog_path=bio)

    rows = aug.cf_rows()
    keys = [(r["database"], r["code"]) for r in rows]
    assert keys == sorted(keys)
