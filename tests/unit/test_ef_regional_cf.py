"""Tests for ``RegionalCfTable`` + ``RegionalCfRegistryBuilder``.

Pins the bio3-keyed per-(flow, location) regional CF emission used to
populate ``registry/method_cfs/<non-water-slug>/regional_cfs.parquet``.
The builder joins JRC's per-FLOW_uuid regional rows to bio3 /
ecoinvent-biosphere codes via ``(name_lower, top_compartment)`` so the
sidecar references rows the matrix actually carries.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from ef.cf_table import EfCfTable
from ef.regional_cf import RegionalCfRegistryBuilder, RegionalCfTable


def _write_cf_parquet(tmp_path: Path, rows: list[dict]) -> EfCfTable:
    """Materialise a stub JRC EF v3.1 CF parquet and return an ``EfCfTable``."""
    p = tmp_path / "ef_cf.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(p)
    later = time.time() + 10
    os.utime(p, (later, later))
    return EfCfTable(path=p)


_BIO_COLUMNS = ("database", "code", "name", "categories", "unit", "cas")


def _write_bio_catalog(tmp_path: Path, rows: list[dict]) -> Path:
    """Stub biosphere catalog parquet with the columns the builder reads."""
    p = tmp_path / "biosphere_catalog.parquet"
    df = pd.DataFrame(rows, columns=list(_BIO_COLUMNS))
    df.to_parquet(p)
    return p


# ----------------------------------------------------------------------------
# RegionalCfTable


def test_regional_rows_returns_tidy_frame_for_method(tmp_path: Path):
    """``regional_rows`` strips the JRC parquet down to one row per
    (FLOW_uuid, location) for the requested method and renames columns
    to the builder's canonical schema."""
    table = _write_cf_parquet(
        tmp_path,
        [
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "DE",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.13,
            },
            # Global row — excluded by location notna filter
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": None,
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 3.02,
            },
            # Other method — excluded
            {
                "LCIAMethod_name": "Land use",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "land-uuid",
                "FLOW_name": "land transformation",
                "FLOW_class0": "Land use",
                "FLOW_class1": None,
                "CF EF3.1": 1.0,
            },
        ],
    )
    out = RegionalCfTable(cf_table=table).regional_rows("Acidification")
    assert list(out.columns) == ["flow_uuid", "flow_name", "top_class", "location", "cf"]
    assert len(out) == 2
    by_loc = dict(zip(out["location"], out["cf"], strict=False))
    assert by_loc == {"FR": 2.78, "DE": 2.13}


def test_regional_rows_empty_when_method_has_no_regional(tmp_path: Path):
    table = _write_cf_parquet(
        tmp_path,
        [
            {
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": None,
                "FLOW_uuid": "co2-uuid",
                "FLOW_name": "carbon dioxide",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 1.0,
            },
        ],
    )
    out = RegionalCfTable(cf_table=table).regional_rows("Climate change")
    assert out.empty
    assert list(out.columns) == ["flow_uuid", "flow_name", "top_class", "location", "cf"]


# ----------------------------------------------------------------------------
# RegionalCfRegistryBuilder


_ACID_KEY = ("acidification", "accumulated exceedance (AE)")


def _make_builder(
    tmp_path: Path,
    regional_jrc_rows: list[dict],
    bio_catalog_rows: list[dict],
    enabled_methods: frozenset[tuple[str, str]] = frozenset({_ACID_KEY}),
) -> RegionalCfRegistryBuilder:
    table = _write_cf_parquet(tmp_path, regional_jrc_rows)
    cat_path = _write_bio_catalog(tmp_path, bio_catalog_rows)
    return RegionalCfRegistryBuilder(
        regional_table=RegionalCfTable(cf_table=table),
        biosphere_catalog_path=cat_path,
        enabled_methods=enabled_methods,
    )


def test_build_rows_emits_bio3_keyed_rows_preserving_global_sign(tmp_path: Path):
    """A bio3 code with global CF=+3.02 should pick up regional rows
    keyed by THAT bio3 (db, code) — not by the JRC FLOW_uuid — with the
    sign preserved (positive) so the bw2io snapshot's emission/consumption
    convention survives the correction."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "DE",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.13,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "bio-nh3-air-unspec",
                "name": "Ammonia",
                "categories": ["air", "unspecified"],
                "unit": "kilogram",
                "cas": "007664-41-7",
            },
        ],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[
            {"database": "biosphere3", "code": "bio-nh3-air-unspec", "amount": 3.02},
        ],
    )
    by_loc = {r["location"]: r for r in rows}
    assert set(by_loc) == {"FR", "DE"}
    assert all(r["database"] == "biosphere3" for r in rows)
    assert all(r["code"] == "bio-nh3-air-unspec" for r in rows)
    assert abs(by_loc["FR"]["amount"] - 2.78) < 1e-9
    assert abs(by_loc["DE"]["amount"] - 2.13) < 1e-9


def test_build_rows_applies_sign_of_global_to_regional_magnitude(tmp_path: Path):
    """A bio3 row with a negative global CF (consumption/emission sign
    flipped by bw2io) must keep that sign — the regional row uses
    ``sign(global) × |regional|``."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "bio-nh3-flipped",
                "name": "Ammonia",
                "categories": ["air"],
                "unit": "kilogram",
                "cas": "007664-41-7",
            },
        ],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[{"database": "biosphere3", "code": "bio-nh3-flipped", "amount": -3.02}],
    )
    assert len(rows) == 1
    assert abs(rows[0]["amount"] - (-2.78)) < 1e-9


def test_build_rows_collapses_multiple_jrc_uuids_per_name_top(tmp_path: Path):
    """JRC ships 5 ammonia FLOW_uuids that all carry the same |CF| per
    location. Two bio3 codes with name="Ammonia" + top="air" must each
    receive the regional correction, taking the mean |CF| across the
    matching JRC uuids (defensive against future divergence)."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid-a",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid-b",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "bio-nh3-1",
                "name": "Ammonia",
                "categories": ["air", "unspecified"],
                "unit": "kilogram",
                "cas": "007664-41-7",
            },
            {
                "database": "biosphere3",
                "code": "bio-nh3-2",
                "name": "Ammonia",
                "categories": ["air", "low population density"],
                "unit": "kilogram",
                "cas": "007664-41-7",
            },
        ],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[
            {"database": "biosphere3", "code": "bio-nh3-1", "amount": 3.02},
            {"database": "biosphere3", "code": "bio-nh3-2", "amount": 3.02},
        ],
    )
    by_code = {r["code"]: r for r in rows}
    assert set(by_code) == {"bio-nh3-1", "bio-nh3-2"}
    for r in rows:
        assert r["location"] == "FR"
        assert abs(r["amount"] - 2.78) < 1e-9


def test_build_rows_ignores_ef_db_global_rows(tmp_path: Path):
    """EF-coded global CFs are inert in the matrix for non-water methods
    (no AGB/ecoinvent emission routes onto an EF FLOW_uuid for acid
    etc.), so the builder must not emit regional siblings for them."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
        ],
        bio_catalog_rows=[],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[{"database": "ef", "code": "nh3-uuid", "amount": 3.02}],
    )
    assert rows == []


def test_build_rows_drops_bio3_rows_without_matching_jrc_name(tmp_path: Path):
    """A bio3 code whose name doesn't appear in the JRC regional set
    (e.g. NOX → "Nitrogen oxides" vs JRC "Nitrogen dioxide") gets no
    regional correction — the global CF stays unchanged at score time."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "no2-uuid",
                "FLOW_name": "nitrogen dioxide",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 0.74,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "bio-nox",
                "name": "Nitrogen oxides",
                "categories": ["air", "unspecified"],
                "unit": "kilogram",
                "cas": None,
            },
        ],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[{"database": "biosphere3", "code": "bio-nox", "amount": 0.56}],
    )
    assert rows == []


def test_build_rows_skips_locations_within_diff_eps(tmp_path: Path):
    """A regional |CF| within DIFF_EPS of |global| is dropped — no
    point storing rows that would be no-ops at score time."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "ZZ",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 3.020,  # exact match
            },
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "ZW",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 3.025,  # within DIFF_EPS
            },
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "bio-nh3",
                "name": "Ammonia",
                "categories": ["air"],
                "unit": "kilogram",
                "cas": "007664-41-7",
            },
        ],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[{"database": "biosphere3", "code": "bio-nh3", "amount": 3.02}],
    )
    assert {r["location"] for r in rows} == {"FR"}


def test_build_rows_stable_ordering(tmp_path: Path):
    """Output sorted by (database, code, location) — required for
    deterministic parquet bytes + scoring package content hash."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "DE",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.13,
            },
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "code-b",
                "name": "Ammonia",
                "categories": ["air"],
                "unit": "kilogram",
                "cas": None,
            },
            {
                "database": "biosphere3",
                "code": "code-a",
                "name": "Ammonia",
                "categories": ["air"],
                "unit": "kilogram",
                "cas": None,
            },
        ],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[
            {"database": "biosphere3", "code": "code-b", "amount": 3.02},
            {"database": "biosphere3", "code": "code-a", "amount": 3.02},
        ],
    )
    triples = [(r["database"], r["code"], r["location"]) for r in rows]
    assert triples == sorted(triples)


def test_build_rows_returns_empty_when_method_not_in_allowlist(tmp_path: Path):
    """A method excluded from ``enabled_methods`` must produce no rows
    even when JRC has per-location data and bio3 names match. The gate
    is how we suppress regional application on methods (acid,
    eutroph-terr) where ADEME's reference uses the JRC site-generic
    CF — applying regional there would diverge from the backtest
    target."""
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "bio-nh3",
                "name": "Ammonia",
                "categories": ["air"],
                "unit": "kilogram",
                "cas": "007664-41-7",
            },
        ],
        enabled_methods=frozenset(),  # explicitly empty allowlist
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[{"database": "biosphere3", "code": "bio-nh3", "amount": 3.02}],
    )
    assert rows == []


def test_to_dataframe_schema(tmp_path: Path):
    builder = _make_builder(
        tmp_path,
        regional_jrc_rows=[
            {
                "LCIAMethod_name": "Acidification",
                "LCIAMethod_location": "FR",
                "FLOW_uuid": "nh3-uuid",
                "FLOW_name": "ammonia",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to air",
                "CF EF3.1": 2.78,
            },
        ],
        bio_catalog_rows=[
            {
                "database": "biosphere3",
                "code": "bio-nh3",
                "name": "Ammonia",
                "categories": ["air"],
                "unit": "kilogram",
                "cas": None,
            },
        ],
    )
    rows = builder.build_rows(
        method_key=_ACID_KEY,
        method_name="Acidification",
        global_cfs=[{"database": "biosphere3", "code": "bio-nh3", "amount": 3.02}],
    )
    df = RegionalCfRegistryBuilder.to_dataframe(rows)
    assert list(df.columns) == ["database", "code", "location", "amount"]
    assert str(df["database"].dtype) == "string"
    assert str(df["code"].dtype) == "string"
    assert str(df["location"].dtype) == "string"
    assert str(df["amount"].dtype) == "float64"
