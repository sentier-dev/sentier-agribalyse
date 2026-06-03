"""``BiosphereFlowmapApplier`` — applies ``agribalyse-3.2-ecoinvent-3.10-biosphere.json``.

Includes the upstream NaN-cf patch lifted out of the legacy linker
(``_patch_biosphere_randonneur_nan_cfs``). The randonneur datapackage
ships 7 entries with ``conversion_factor=NaN`` that break
``Database.process()`` downstream.

Also patches a small set of *inverted* ``conversion_factor`` entries on
``Energy, from <source> (MJ) → <fuel> (<mass-or-volume>)`` rows. Those
factors are stored as the fuel's heat content (MJ per kg or per Sm³),
when the convention everywhere else in the file is
``target = source × multiplier`` — i.e., the reciprocal. Without the
patch, 1 MJ of "Energy, from uranium" gets mapped to 560 000 kg of
"Uranium, in ground" instead of 1.79 × 10⁻⁶ kg, blowing up
non-renewable energy scores 3.1 × 10¹¹× for any product whose supply
chain touches AGB ``Sea cage``, fisheries, or other activities that
emit those energy-equivalent biosphere flows. (Backtest outliers like
"Bar rayé" 188 million× over were traced here.)"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import randonneur as rn

from config import Settings
from core.logging import Logging


@dataclass(frozen=True)
class BiosphereFlowmapApplier:
    settings: Settings

    # Mn-54 water target lifted from upstream entry [1556] — used to fix
    # the broken Manganese-55/Bq entry whose target/cf is NaN.
    MN54_WATER_TARGET: ClassVar[dict] = {
        "name": "Manganese-54",
        "unit": "kBq",
        "identifier": "729eac4f-0339-4d0f-8956-a71ebe20a527",
        "context": ["water", "unspecified"],
        "CAS number": "13966-31-9",
    }

    # Upstream "Energy, from <fuel>" → fuel mapping rows that store the
    # fuel's heat content (MJ / kg-or-Sm3) instead of the conversion
    # multiplier the rest of the file uses. Each pair (source_name,
    # source_unit, target_unit) has a single corrected multiplier — the
    # reciprocal of the energy density. We match by tuple to avoid
    # re-patching unrelated entries (the file does carry sane Bq → kBq
    # rows at cf 0.001 etc.).
    ENERGY_TO_MASS_INVERSIONS: ClassVar[dict] = {
        # 1 MJ of uranium energy ≈ 1 / 560 000 kg of uranium.
        ("Energy, from uranium", "MJ", "kg"): 1.0 / 560_000.0,
        # 1 MJ of brown-coal energy ≈ 1 / 9.9 kg of brown coal.
        ("Energy, from coal, brown", "MJ", "kg"): 1.0 / 9.9,
        # 1 MJ of peat energy ≈ 1 / 9.9 kg of peat.
        ("Energy, from peat", "MJ", "kg"): 1.0 / 9.9,
        # 1 MJ of natural-gas energy ≈ 1 / 40.3 Sm³ of natural gas.
        ("Energy, from gas, natural", "MJ", "Sm3"): 1.0 / 40.3,
        ("Energy, from gas, natural", "MJ", "cubic meter"): 1.0 / 40.3,
        # MJ → MJ identity. Upstream flowmap ships cf=40.3 here (a
        # natural-gas heat-content copy-paste). The bio3 target
        # ``Energy, potential (in hydropower reservoir), converted``
        # carries no JRC EF v3.1 CF on any method, so the bug has no
        # current scoring impact — but the convention everywhere else
        # for renewable-energy MJ→MJ rows (wind, solar, geothermal,
        # biomass) is cf=1.0. Reset to 1.0 to keep the inventory
        # accounting honest in case the flow becomes characterised.
        ("Energy, from hydro power", "MJ", "MJ"): 1.0,
    }

    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(self, sp) -> dict[str, int]:
        path = self.settings.paths.biosphere_flowmap_json
        if not path.exists():
            return {"applied": 0, "patched_nan_cfs": 0, "patched_inverted_cfs": 0}

        dp = rn.Datapackage.from_json(path)
        n_patched = self._patch_nan_cfs(dp)
        n_inverted = self._patch_inverted_energy_cfs(dp)
        sp.randonneur(datapackage=dp)
        self._log.info(
            "transforms.biosphere_flowmap.applied",
            datapackage=path.name,
            patched_nan_cfs=n_patched,
            patched_inverted_cfs=n_inverted,
        )
        return {
            "applied": 1,
            "patched_nan_cfs": n_patched,
            "patched_inverted_cfs": n_inverted,
        }

    def _patch_nan_cfs(self, dp) -> int:
        """Fix the 7 NaN ``conversion_factor`` entries in the upstream flowmap.

        - 6 × Water (kg → m³): density ≈ 1000 kg/m³ ⇒ cf = 0.001.
        - 1 × Manganese-55 (Bq): stable isotope, treat as Mn-54 typo and
          redirect to the Mn-54 / kBq water target with cf = 0.001.
        """
        n = 0
        for entry in dp.data["update"]:
            cf = entry.get("conversion_factor")
            if not (isinstance(cf, float) and math.isnan(cf)):
                continue
            src_name = entry.get("source", {}).get("name", "")
            src_unit = entry.get("source", {}).get("unit", "")
            if src_name == "Water" and src_unit == "kg":
                entry["conversion_factor"] = 0.001
            elif src_name == "Manganese-55" and src_unit == "Bq":
                entry["target"] = dict(self.MN54_WATER_TARGET)
                entry["conversion_factor"] = 0.001
                entry["comment"] = (
                    (entry.get("comment") or "") + " [patched: Mn-55/Bq treated as Mn-54 typo]"
                ).strip()
            else:
                raise ValueError(f"Unexpected NaN conversion_factor in upstream flowmap: {entry}")
            n += 1
        return n

    def _patch_inverted_energy_cfs(self, dp) -> int:
        """Replace ``conversion_factor`` on the Energy → fuel rows with
        the reciprocal so the convention matches the rest of the file
        (target = source * multiplier)."""
        n = 0
        for entry in dp.data["update"]:
            src = entry.get("source", {})
            tgt = entry.get("target", {})
            key = (src.get("name", ""), src.get("unit", ""), tgt.get("unit", ""))
            corrected = self.ENERGY_TO_MASS_INVERSIONS.get(key)
            if corrected is None:
                continue
            entry["conversion_factor"] = corrected
            entry["comment"] = (
                (entry.get("comment") or "")
                + f" [patched: inverted heat-content conversion to {corrected:.6e}]"
            ).strip()
            n += 1
        return n
