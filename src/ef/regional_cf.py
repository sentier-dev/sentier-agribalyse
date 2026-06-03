"""``RegionalCfTable`` / ``RegionalCfRegistryBuilder`` — per-bio3-row, per-location
regional CFs for non-water LCIA methods.

Generalises the water-use regional pattern (:mod:`ef.regional_water_cf`) to
every method that ships per-``LCIAMethod_location`` rows in the JRC EF v3.1
parquet. The matrix carries one row per ``(database, code)`` from the AGB +
ecoinvent biosphere link, which means **bio3 / ecoinvent-3.9.1-biosphere**
codes — not the JRC EF ``FLOW_uuid`` the regional data is keyed against.
A regional sidecar must therefore also be keyed by bio3 (db, code) to
have any effect at scoring time; rows keyed against the EF FLOW_uuid would
map to rows the matrix doesn't carry and contribute nothing.

The build path:

1. :class:`RegionalCfTable.regional_rows` returns a tidy DataFrame of every
   per-location row for the method (``flow_uuid, flow_name, top_class,
   location, cf``).
2. :class:`RegionalCfRegistryBuilder.build_rows`:

   a. Folds JRC rows by ``(flow_name_lower, top_class_bio3, location)``
      taking the mean of ``|cf|``. Multiple JRC FLOW_uuids that share a
      name + top compartment (e.g. the 11 water uuids for water-use, or
      the 5 ammonia sub-compartment uuids for acidification) collapse to
      a single magnitude per location, matching the convention water-use
      already established.
   b. Iterates the bio3 / ecoinvent-3.9.1-biosphere rows in the post-SimaPro
      global CF set and looks up each one's ``(name, top)`` in the
      biosphere catalog.
   c. For each matched location, emits ``(bio3_db, bio3_code, location,
      sign(global) × |regional_cf|)`` so the bw2io snapshot's sign
      convention on bio3 is preserved. Rows within :attr:`DIFF_EPS` of
      ``|global|`` are dropped.

EF-coded global CFs (``database == ef``) are ignored: those rows are inert
in the matrix (no AGB/ecoinvent activity emits onto them for non-water
methods), so emitting regional siblings would be wasted bytes.

Downstream consumers (:class:`MethodCfRegistryLoader.load_regional_all`
→ :class:`scoring.scoring_package.ScoringPackageBuilder` →
:class:`scoring.regional_correction.RegionalCorrectionBuilder`) are
already method-agnostic; existence of ``regional_cfs.parquet`` next to
``cfs.parquet`` is all they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ef.cf_table import EfCfTable

# Top compartment in the JRC source lives in ``FLOW_class1``
# ("Emissions to air" / "Emissions to water" / "Emissions to soil" /
# "Resources from water"); ``FLOW_class0`` is the parent kingdom ("Emissions"
# / "Resources" / "Land use") that doesn't discriminate compartments.
# bio3 / ecoinvent use the shorter labels "air" / "water" / "soil" /
# "natural resource" — the table joins against the bio3 vocabulary.
_JRC_TO_BIO3_TOP: dict[str, str] = {
    "emissions to air": "air",
    "emissions to water": "water",
    "emissions to soil": "soil",
    "resources from water": "natural resource",
    "resources from ground": "natural resource",
    "resources": "natural resource",
    "raw": "natural resource",
}


@dataclass(frozen=True)
class RegionalCfTable:
    """Tidy DataFrame view of per-location CF rows for a method."""

    cf_table: EfCfTable

    REGIONAL_COLUMNS: ClassVar[tuple[str, ...]] = (
        "flow_uuid",
        "flow_name",
        "top_class",
        "location",
        "cf",
    )

    def regional_rows(self, method_name: str) -> pd.DataFrame:
        """Return regional rows for ``method_name`` with normalised columns.

        ``top_class`` is read from ``FLOW_class1`` because that's where
        the actual compartment label lives ("Emissions to air"); ``FLOW_class0``
        is the parent kingdom that doesn't discriminate compartments.

        Empty DataFrame (with the right schema) when the method has no
        per-location rows.
        """
        df = self.cf_table.raw
        method_rows = df[df["LCIAMethod_name"] == method_name]
        regional = method_rows[method_rows["LCIAMethod_location"].notna()]
        if regional.empty:
            return pd.DataFrame({col: pd.Series(dtype="object") for col in self.REGIONAL_COLUMNS})
        out = regional[
            ["FLOW_uuid", "FLOW_name", "FLOW_class1", "LCIAMethod_location", "CF EF3.1"]
        ].copy()
        out.columns = list(self.REGIONAL_COLUMNS)
        out["cf"] = out["cf"].astype(float)
        return out.reset_index(drop=True)


@dataclass(frozen=True)
class RegionalCfRegistryBuilder:
    """Emit per-(bio3 db, bio3 code, location, amount) rows for any non-water method.

    See module docstring for the algorithm. The builder owns one fixed
    dependency — the biosphere catalog path — and is called per method
    via :meth:`build_rows`.

    Opt-in allowlist (``enabled_methods``): a method receives regional
    sidecars **only** if its ``(category, indicator)`` tuple is in this
    set. JRC ships per-location CFs for ~10 methods, but ADEME's
    AGRIBALYSE reference applies regional CFs only for a subset (water-use
    via AWARE; ecotoxicity freshwater). Emitting regional sidecars for
    methods ADEME treats as site-generic (e.g. acidification, where the
    JRC global is much higher than national averages) makes our scores
    diverge from the reference even though they're scientifically
    defensible — see backtest 2026-05-21 for the +20%-median bias on
    acidification + eutrophication terrestrial when those were enabled.
    Default is an empty set; the build CLI passes the audited allowlist.
    """

    regional_table: RegionalCfTable
    biosphere_catalog_path: Path
    enabled_methods: frozenset[tuple[str, str]] = frozenset()

    # Magnitude difference below which a regional row is treated as a
    # no-op against the global. Matches :attr:`RegionalWaterCfRegistryBuilder.DIFF_EPS`.
    DIFF_EPS: ClassVar[float] = 0.01

    # The two biosphere databases the matrix actually carries rows for.
    BIO_DATABASES: ClassVar[frozenset[str]] = frozenset({"biosphere3", "ecoinvent-3.9.1-biosphere"})

    def build_rows(
        self,
        *,
        method_key: tuple[str, str],
        method_name: str,
        global_cfs: list[dict],
    ) -> list[dict]:
        if method_key not in self.enabled_methods:
            return []
        regional_df = self.regional_table.regional_rows(method_name)
        if regional_df.empty:
            return []

        per_name_top_location = self._collapse_regional_by_name_top(regional_df)
        if not per_name_top_location:
            return []

        bio_lookup = self._bio_lookup
        out: list[dict] = []
        for row in global_cfs:
            db = str(row.get("database", "") or "")
            if db not in self.BIO_DATABASES:
                continue
            code = str(row.get("code", "") or "")
            if not code:
                continue
            try:
                amount = float(row.get("amount", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if amount == 0.0:
                continue
            meta = bio_lookup.get((db, code))
            if meta is None:
                continue
            per_loc = per_name_top_location.get(meta)
            if not per_loc:
                continue
            global_abs = abs(amount)
            sign = 1.0 if amount >= 0 else -1.0
            for location, regional_abs in per_loc.items():
                if abs(regional_abs - global_abs) < self.DIFF_EPS:
                    continue
                out.append(
                    {
                        "database": db,
                        "code": code,
                        "location": location,
                        "amount": sign * regional_abs,
                    }
                )
        out.sort(key=lambda r: (r["database"], r["code"], r["location"]))
        return out

    @staticmethod
    def _collapse_regional_by_name_top(
        regional_df: pd.DataFrame,
    ) -> dict[tuple[str, str], dict[str, float]]:
        """``(flow_name_lower, top_class_bio3) → {location: mean(|cf|)}``.

        FLOW_uuids that share a flow name + top compartment in the JRC
        source agree on ``|cf|`` per location (verified for water-use's
        11 uuids on 2026-05-13; same pattern for ammonia / NOX / SO2
        clusters in acidification). The mean is a defensive collapse —
        it would surface any future divergence as a slightly different
        magnitude rather than silently picking one row.
        """
        df = regional_df.copy()
        df["name_lower"] = df["flow_name"].astype(str).str.lower().str.strip()
        df["top_lower"] = df["top_class"].astype(str).str.lower().str.strip()
        df["top_bio3"] = df["top_lower"].map(_JRC_TO_BIO3_TOP).fillna(df["top_lower"])
        df["cf_abs"] = df["cf"].abs()
        agg = (
            df.groupby(["name_lower", "top_bio3", "location"], sort=False)["cf_abs"]
            .mean()
            .reset_index()
        )
        out: dict[tuple[str, str], dict[str, float]] = {}
        for r in agg.itertuples(index=False):
            cf_val = float(r.cf_abs)
            if cf_val <= 0.0:
                continue
            out.setdefault((r.name_lower, r.top_bio3), {})[str(r.location)] = cf_val
        return out

    @cached_property
    def _bio_lookup(self) -> dict[tuple[str, str], tuple[str, str]]:
        """``(db, code) → (name_lower, top_bio3)`` for bio3 / ecoinvent-biosphere rows."""
        cat = pd.read_parquet(self.biosphere_catalog_path)
        cat = cat[cat["database"].isin(self.BIO_DATABASES)]
        out: dict[tuple[str, str], tuple[str, str]] = {}
        for db, code, name, cats in zip(
            cat["database"], cat["code"], cat["name"], cat["categories"], strict=False
        ):
            name_lower = str(name).lower().strip() if name is not None else ""
            out[(str(db), str(code))] = (name_lower, self._top_of(cats))
        return out

    @staticmethod
    def _top_of(cats: object) -> str:
        if cats is None:
            return ""
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if isinstance(cats, (list, tuple)) and cats:
            return str(cats[0]).strip().lower()
        return ""

    @staticmethod
    def to_dataframe(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["database", "code", "location", "amount"])
        return df.astype(
            {
                "database": "string",
                "code": "string",
                "location": "string",
                "amount": "float64",
            }
        )
