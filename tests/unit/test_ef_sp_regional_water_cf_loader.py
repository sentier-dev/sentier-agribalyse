"""Unit tests for ``SpRegionalWaterCfLoader``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ef.sp_regional_water_cf_loader import SpRegionalWaterCfLoader


def _write_fixture(path: Path, rows: list[dict]) -> Path:
    cols = (
        "simapro_method",
        "simapro_method_unit",
        "compartment",
        "sub_compartment",
        "name",
        "cas",
        "cf",
        "flow_unit",
        "cf_unit",
    )
    df = pd.DataFrame(rows, columns=list(cols))
    df.to_parquet(path, index=False)
    return path


def _row(name, cf, flow_unit="m3", compartment="Raw", sub_compartment="(unspecified)"):
    return {
        "simapro_method": "Water use",
        "simapro_method_unit": "m3 depriv.",
        "compartment": compartment,
        "sub_compartment": sub_compartment,
        "name": name,
        "cas": "007732-18-5",
        "cf": float(cf),
        "flow_unit": flow_unit,
        "cf_unit": "m3 depriv. / m3",
    }


class TestSpRegionalWaterCfLoader:
    def test_load_emits_one_row_per_regional_variant(self, tmp_path: Path):
        cache = _write_fixture(
            tmp_path / "sp.parquet",
            [
                _row("Water, lake", 42.95),
                _row("Water, lake, FR", 6.98),
                _row("Water, lake, CN", 49.7),
                _row("Water, well, FR", 6.98),
                _row("Water, well, CN", 49.7),
            ],
        )
        loader = SpRegionalWaterCfLoader(cache_path=cache)
        df = loader.df
        # 5 input rows -> 5 output rows; base names normalised, regions parsed.
        assert set(zip(df["base_name"], df["region"], strict=False)) == {
            ("Water, lake", ""),
            ("Water, lake", "FR"),
            ("Water, lake", "CN"),
            ("Water, well", "FR"),
            ("Water, well", "CN"),
        }

    def test_load_strips_kg_variant(self, tmp_path: Path):
        """The kg variant is exactly 1/1000 of the m³ value. Our matrix
        carries amounts in m³, so emitting kg rows would 1000× under-count.
        The loader must reject the kg flow_unit."""
        cache = _write_fixture(
            tmp_path / "sp.parquet",
            [
                _row("Water, unspecified natural origin/kg", 0.04295, flow_unit="kg"),
                _row("Water, unspecified natural origin/m3", 42.95, flow_unit="m3"),
            ],
        )
        loader = SpRegionalWaterCfLoader(cache_path=cache)
        df = loader.df
        assert len(df) == 1
        row = df.iloc[0]
        # Trailing ``/m3`` is stripped from the base name.
        assert row["base_name"] == "Water, unspecified natural origin"
        assert row["flow_unit"] == "m3"
        assert row["cf"] == pytest.approx(42.95)

    def test_release_side_picks_negative_cf(self, tmp_path: Path):
        """SimaPro's release-side CFs are negative — return them verbatim."""
        cache = _write_fixture(
            tmp_path / "sp.parquet",
            [
                _row("Water, FR", -6.98, compartment="Water"),
                _row(
                    "Water, FR",
                    0.0,
                    compartment="Water",
                    sub_compartment="ocean",
                ),
            ],
        )
        loader = SpRegionalWaterCfLoader(cache_path=cache)
        assert loader.cf_for(
            base_name="Water",
            region="FR",
            compartment="Water",
            sub_compartment="(unspecified)",
        ) == pytest.approx(-6.98)
        assert (
            loader.cf_for(
                base_name="Water",
                region="FR",
                compartment="Water",
                sub_compartment="ocean",
            )
            == 0.0
        )
        assert (
            loader.cf_for(
                base_name="Water",
                region="ZZ",  # unknown
                compartment="Water",
                sub_compartment="(unspecified)",
            )
            is None
        )

    def test_non_iso_regions_keep_full_base_name(self, tmp_path: Path):
        """SimaPro carries grid-region rows like ``BR-Mid-western grid``
        and ``Canada without Quebec`` whose trailing tokens aren't
        recognised ISO/aggregate codes. The loader keeps the full name
        as the base and emits ``region = ""`` so the row is usable as a
        non-regional CF fallback but never as a regional override."""
        cache = _write_fixture(
            tmp_path / "sp.parquet",
            [
                _row("Water, lake, BR-Mid-western grid", 2.17),
            ],
        )
        loader = SpRegionalWaterCfLoader(cache_path=cache)
        df = loader.df
        assert len(df) == 1
        row = df.iloc[0]
        assert row["base_name"] == "Water, lake, BR-Mid-western grid"
        assert row["region"] == ""

    def test_empty_method_returns_empty_frame(self, tmp_path: Path):
        cache = _write_fixture(
            tmp_path / "sp.parquet",
            [
                {
                    "simapro_method": "Climate change",
                    "simapro_method_unit": "kg CO2 eq",
                    "compartment": "Air",
                    "sub_compartment": "(unspecified)",
                    "name": "Carbon dioxide",
                    "cas": "124-38-9",
                    "cf": 1.0,
                    "flow_unit": "kg",
                    "cf_unit": "kg CO2 eq / kg",
                },
            ],
        )
        loader = SpRegionalWaterCfLoader(cache_path=cache)
        assert loader.df.empty

    def test_missing_file_raises(self, tmp_path: Path):
        loader = SpRegionalWaterCfLoader(cache_path=tmp_path / "nope.parquet")
        with pytest.raises(FileNotFoundError, match="SimaPro adapted CF parquet"):
            _ = loader.df
