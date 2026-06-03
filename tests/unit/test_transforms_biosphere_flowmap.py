"""Unit tests for ``BiosphereFlowmapApplier`` patches.

Covers the inverted-energy-conversion patcher introduced to fix the
``Energy, from uranium → Uranium`` cf=560000 (and friends) bug that
inflated non-renewable energy / fossil-fuel resource scores by up to
3.1 × 10^11 for any product whose supply chain touched AGB ``Sea cage``
or other activities emitting ``Energy, from <fuel>`` biosphere flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from transforms.biosphere_flowmap import BiosphereFlowmapApplier


@dataclass
class _Datapackage:
    data: dict[str, Any]


def _entry(src_name: str, src_unit: str, tgt_unit: str, cf: float) -> dict:
    return {
        "source": {"name": src_name, "unit": src_unit, "context": ["Resources", "in ground"]},
        "target": {
            "name": "<target>",
            "unit": tgt_unit,
            "context": ["natural resource", "in ground"],
        },
        "conversion_factor": cf,
    }


class TestInvertedEnergyConversionPatch:
    def test_patches_energy_from_uranium(self):
        dp = _Datapackage(data={"update": [_entry("Energy, from uranium", "MJ", "kg", 560000.0)]})
        n = BiosphereFlowmapApplier._patch_inverted_energy_cfs(
            BiosphereFlowmapApplier.__new__(BiosphereFlowmapApplier), dp
        )
        assert n == 1
        assert dp.data["update"][0]["conversion_factor"] == pytest.approx(1.0 / 560000.0)
        assert "patched: inverted heat-content" in dp.data["update"][0]["comment"]

    def test_patches_energy_from_coal_brown(self):
        dp = _Datapackage(data={"update": [_entry("Energy, from coal, brown", "MJ", "kg", 9.9)]})
        n = BiosphereFlowmapApplier._patch_inverted_energy_cfs(
            BiosphereFlowmapApplier.__new__(BiosphereFlowmapApplier), dp
        )
        assert n == 1
        assert dp.data["update"][0]["conversion_factor"] == pytest.approx(1.0 / 9.9)

    def test_patches_energy_from_peat(self):
        dp = _Datapackage(data={"update": [_entry("Energy, from peat", "MJ", "kg", 9.9)]})
        n = BiosphereFlowmapApplier._patch_inverted_energy_cfs(
            BiosphereFlowmapApplier.__new__(BiosphereFlowmapApplier), dp
        )
        assert n == 1
        assert dp.data["update"][0]["conversion_factor"] == pytest.approx(1.0 / 9.9)

    def test_patches_energy_from_gas_natural_sm3(self):
        dp = _Datapackage(data={"update": [_entry("Energy, from gas, natural", "MJ", "Sm3", 40.3)]})
        n = BiosphereFlowmapApplier._patch_inverted_energy_cfs(
            BiosphereFlowmapApplier.__new__(BiosphereFlowmapApplier), dp
        )
        assert n == 1
        assert dp.data["update"][0]["conversion_factor"] == pytest.approx(1.0 / 40.3)

    def test_patches_energy_from_hydro_power_mj_to_mj(self):
        # MJ → MJ should be the identity (1.0). The upstream flowmap ships
        # cf=40.3 (the natural-gas heat content, plainly a copy-paste
        # artefact). Reset to 1.0 to match the convention used by the
        # other renewable energy entries (wind / solar / geothermal /
        # biomass all ship MJ → MJ at cf=1.0).
        dp = _Datapackage(
            data={
                "update": [
                    {
                        "source": {
                            "name": "Energy, from hydro power",
                            "unit": "MJ",
                            "context": ["Resources", "in water"],
                        },
                        "target": {
                            "name": "Energy, potential (in hydropower reservoir), converted",
                            "unit": "MJ",
                            "context": ["natural resource", "in water"],
                        },
                        "conversion_factor": 40.3,
                    }
                ]
            }
        )
        n = BiosphereFlowmapApplier._patch_inverted_energy_cfs(
            BiosphereFlowmapApplier.__new__(BiosphereFlowmapApplier), dp
        )
        assert n == 1
        assert dp.data["update"][0]["conversion_factor"] == pytest.approx(1.0)
        assert "patched: inverted heat-content" in dp.data["update"][0]["comment"]

    def test_unrelated_entries_left_alone(self):
        # Bq → kBq at cf=0.001 is correct; should NOT be touched.
        bq_entry = {
            "source": {"name": "Caesium-137", "unit": "Bq"},
            "target": {"name": "Caesium-137", "unit": "kBq"},
            "conversion_factor": 0.001,
        }
        dp = _Datapackage(data={"update": [dict(bq_entry)]})
        n = BiosphereFlowmapApplier._patch_inverted_energy_cfs(
            BiosphereFlowmapApplier.__new__(BiosphereFlowmapApplier), dp
        )
        assert n == 0
        assert dp.data["update"][0]["conversion_factor"] == 0.001

    def test_handles_multiple_entries(self):
        dp = _Datapackage(
            data={
                "update": [
                    _entry("Energy, from uranium", "MJ", "kg", 560000.0),
                    _entry("Energy, from coal, brown", "MJ", "kg", 9.9),
                    {
                        "source": {"name": "Other", "unit": "kg"},
                        "target": {"unit": "kg"},
                        "conversion_factor": 1.0,
                    },
                ]
            }
        )
        n = BiosphereFlowmapApplier._patch_inverted_energy_cfs(
            BiosphereFlowmapApplier.__new__(BiosphereFlowmapApplier), dp
        )
        assert n == 2
