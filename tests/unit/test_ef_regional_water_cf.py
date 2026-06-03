"""Tests for ``RegionalWaterCfTable`` + ``RegionalWaterCfRegistryBuilder``.

These pin the per-location magnitude collapse (mean over the 11 JRC water
flow UUIDs) and the sign-preserving row emission used to populate
``registry/method_cfs/<water-slug>/regional_cfs.parquet``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from ef.cf_table import EfCfTable
from ef.regional_water_cf import RegionalWaterCfRegistryBuilder, RegionalWaterCfTable


def _write_cf_parquet(tmp_path: Path, rows: list[dict]) -> EfCfTable:
    """Materialise a stub JRC EF v3.1 CF parquet and return an ``EfCfTable``."""
    p = tmp_path / "ef_cf.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(p)
    later = time.time() + 10
    os.utime(p, (later, later))
    return EfCfTable(path=p)


# ----------------------------------------------------------------------------
# RegionalWaterCfTable


def test_cf_by_location_collapses_11_water_flows_to_mean_abs(tmp_path: Path):
    """JRC parquet ships the same regional CF magnitude across 11 water
    FLOW_uuids per location, with sign flipping between resource and
    emission rows. The collapse should take the mean of |CF| so positive
    and negative rows contribute symmetrically and the per-location
    magnitude is preserved.
    """
    table = _write_cf_parquet(
        tmp_path,
        [
            # BR: +2.43 for resource flow, -2.43 for emission flow
            {
                "LCIAMethod_name": "Water use",
                "LCIAMethod_location": "BR",
                "FLOW_uuid": "freshwater-uuid",
                "FLOW_name": "freshwater",
                "CF EF3.1": 2.43,
            },
            {
                "LCIAMethod_name": "Water use",
                "LCIAMethod_location": "BR",
                "FLOW_uuid": "water-emission-uuid",
                "FLOW_name": "Water",
                "CF EF3.1": -2.43,
            },
            # CY: 74.3
            {
                "LCIAMethod_name": "Water use",
                "LCIAMethod_location": "CY",
                "FLOW_uuid": "freshwater-uuid",
                "FLOW_name": "freshwater",
                "CF EF3.1": 74.3,
            },
            # global row (LCIAMethod_location NaN) — should be ignored
            {
                "LCIAMethod_name": "Water use",
                "LCIAMethod_location": None,
                "FLOW_uuid": "freshwater-uuid",
                "FLOW_name": "freshwater",
                "CF EF3.1": 42.95,
            },
            # Other method — should be ignored
            {
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": "BR",
                "FLOW_uuid": "co2-uuid",
                "FLOW_name": "Carbon dioxide",
                "CF EF3.1": 1.0,
            },
        ],
    )
    rt = RegionalWaterCfTable(cf_table=table)
    out = rt.cf_by_location

    assert set(out) == {"BR", "CY"}
    assert abs(out["BR"] - 2.43) < 1e-9, "BR magnitude is mean(|2.43|,|-2.43|) = 2.43"
    assert abs(out["CY"] - 74.3) < 1e-9


def test_cf_by_location_drops_zero_magnitude(tmp_path: Path):
    """Zero-magnitude location entries are dropped so the builder doesn't
    emit no-op correction rows for them."""
    table = _write_cf_parquet(
        tmp_path,
        [
            {
                "LCIAMethod_name": "Water use",
                "LCIAMethod_location": "AQ",
                "FLOW_uuid": "freshwater-uuid",
                "FLOW_name": "freshwater",
                "CF EF3.1": 0.0,
            },
            {
                "LCIAMethod_name": "Water use",
                "LCIAMethod_location": "BR",
                "FLOW_uuid": "freshwater-uuid",
                "FLOW_name": "freshwater",
                "CF EF3.1": 2.43,
            },
        ],
    )
    out = RegionalWaterCfTable(cf_table=table).cf_by_location
    assert "AQ" not in out
    assert "BR" in out


def test_cf_by_location_empty_when_no_water_use(tmp_path: Path):
    """If the JRC source has no water-use rows we return an empty dict
    rather than raising — keeps the registry build robust."""
    table = _write_cf_parquet(
        tmp_path,
        [
            {
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": "BR",
                "FLOW_uuid": "co2-uuid",
                "FLOW_name": "Carbon dioxide",
                "CF EF3.1": 1.0,
            },
        ],
    )
    out = RegionalWaterCfTable(cf_table=table).cf_by_location
    assert out == {}


# ----------------------------------------------------------------------------
# RegionalWaterCfRegistryBuilder


def _make_builder(tmp_path: Path, regional_cfs: dict[str, float]) -> RegionalWaterCfRegistryBuilder:
    """Bypass JRC parquet entirely — inject the regional table directly."""
    table = _write_cf_parquet(
        tmp_path,
        [
            {
                "LCIAMethod_name": "Water use",
                "LCIAMethod_location": loc,
                "FLOW_uuid": "freshwater-uuid",
                "FLOW_name": "freshwater",
                "CF EF3.1": cf,
            }
            for loc, cf in regional_cfs.items()
        ],
    )
    return RegionalWaterCfRegistryBuilder(regional_table=RegionalWaterCfTable(cf_table=table))


def test_build_rows_preserves_global_sign_per_bio3_row(tmp_path: Path):
    """For a bio3 row whose global CF is +42.95, regional corrections
    must also be positive (consumption side). For a -42.95 row (emission
    side), regional corrections must be negative. The bw2io snapshot
    placed both signs on different bio3 codes and they must remain
    distinguishable after the correction.
    """
    builder = _make_builder(tmp_path, {"BR": 2.43, "CY": 74.3})
    rows = builder.build_rows(
        [
            {"database": "biosphere3", "code": "consumption-side", "amount": +42.95},
            {"database": "biosphere3", "code": "emission-side", "amount": -42.95},
        ]
    )
    # Each bio3 row x 2 locations (both differ enough from global) = 4 rows
    assert len(rows) == 4
    by_code = {(r["code"], r["location"]): r["amount"] for r in rows}
    assert abs(by_code[("consumption-side", "BR")] - 2.43) < 1e-9
    assert abs(by_code[("consumption-side", "CY")] - 74.3) < 1e-9
    assert abs(by_code[("emission-side", "BR")] - (-2.43)) < 1e-9
    assert abs(by_code[("emission-side", "CY")] - (-74.3)) < 1e-9


def test_build_rows_skips_locations_near_global(tmp_path: Path):
    """A regional CF within ``DIFF_EPS`` (0.01) of the global is dropped —
    avoids emitting no-op correction rows that would just inflate the
    parquet without changing the score."""
    builder = _make_builder(
        tmp_path,
        {
            "ZZ": 42.950,  # exact global — drop
            "ZW": 42.955,  # within 0.01 — drop
            "BR": 2.43,  # well below global — keep
        },
    )
    rows = builder.build_rows(
        [{"database": "biosphere3", "code": "consumption-side", "amount": +42.95}]
    )
    locations = {r["location"] for r in rows}
    assert locations == {"BR"}


def test_build_rows_skips_zero_amount_global_cfs(tmp_path: Path):
    """A bio3 row that carries amount=0 in the global parquet (CF leaked
    through filter) wouldn't change scoring — emitting regional
    corrections for it would be inert and wasteful."""
    builder = _make_builder(tmp_path, {"BR": 2.43})
    rows = builder.build_rows([{"database": "biosphere3", "code": "zeroed-flow", "amount": 0.0}])
    assert rows == []


def test_build_rows_stable_ordering(tmp_path: Path):
    """Output is sorted by (database, code, location) so the resulting
    parquet bytes are deterministic across runs — important for
    ScoringPackage content_hash stability."""
    builder = _make_builder(tmp_path, {"CY": 74.3, "BR": 2.43, "FR": 6.98})
    rows = builder.build_rows(
        [
            {"database": "biosphere3", "code": "code-b", "amount": 42.95},
            {"database": "biosphere3", "code": "code-a", "amount": 42.95},
        ]
    )
    triples = [(r["database"], r["code"], r["location"]) for r in rows]
    assert triples == sorted(triples)


def test_to_dataframe_schema(tmp_path: Path):
    """DataFrame columns + dtypes match the parquet schema downstream
    consumers expect."""
    builder = _make_builder(tmp_path, {"BR": 2.43})
    rows = builder.build_rows([{"database": "biosphere3", "code": "code-a", "amount": +42.95}])
    df = RegionalWaterCfRegistryBuilder.to_dataframe(rows)
    assert list(df.columns) == ["database", "code", "location", "amount"]
    assert str(df["database"].dtype) == "string"
    assert str(df["code"].dtype) == "string"
    assert str(df["location"].dtype) == "string"
    assert str(df["amount"].dtype) == "float64"
