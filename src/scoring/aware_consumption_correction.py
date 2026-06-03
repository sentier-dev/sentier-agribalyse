"""``AwareConsumptionCorrectionBuilder`` — per-activity AWARE net-consumption
correction for the water-use method.

The bw2io snapshot characterises 5 bio3 ``Water → air`` codes at the global
AWARE CF (+42.95 m³ deprivation/m³). That captures evaporative consumption
that AGB's foreground encodes via a Water-to-air proxy. But the underlying
ecoinvent unit-process activities for water-intensive products (irrigation,
fish farming, tropical-fruit production) record their consumption as
resource extraction (``Water, river [natural resource]`` etc.) with no
matching air emission — so the AGB → Water[air] path under-counts those
products by 50–100x against ADEME's reference.

This builder fills that gap **per-activity, only where the LCI signal is
unambiguous**:

* Apply ``regional_AWARE × (res − ret)`` per activity column j when:

  1. ``asymmetry = |res − ret| / max(res, ret) ≥ 0.95`` — i.e. the
     activity carries an essentially one-sided water flow rather than
     a closed-loop withdrawal/return pair.
  2. ``net = res − ret ≥ 0.05 m³`` per reference unit — filters
     numerical noise.
  3. ``res > ret`` — only POSITIVE net consumption qualifies. The
     opposite direction (return without matching withdrawal) is the
     signature of dehydration / drying / wastewater outputs whose
     physical reality is evaporation (already captured by the
     Water[air] proxy); characterising it would double-count.

* ``res`` and ``ret`` sum over the SimaPro EF v3.1 (adapted) flow set:

  - Resource side (positive): ``Water, lake / river / turbine use``
  - Return side (negative-signed in AWARE; positive in inventory):
    ``Water [water, *]`` for unspecified / fossil well / ground- /
    ground- long-term / surface water sub-compartments. Ocean is
    excluded (saltwater not characterised).

  These are the same flows SimaPro's adapted EF v3.1 export
  characterises (3 resource + 5 emission bio3 codes; see
  ``SimaProCfFilter`` and ``cache/simapro-EF31-adapted-cfs.parquet``).

* The CF is the activity's *regional* AWARE value (per ISO country from
  the JRC parquet) when the activity location is known, falling back
  to ``7.0 m³/m³`` (≈ FR / European median) for activities at
  aggregate locations (GLO / RoW / RER). The bidirectional bio-row
  approach (``Q @ B``) blows up fleet-wide because residual ~2 %
  imbalances across thousands of nominally balanced activities (e.g.
  ``market for electricity FR``) accumulate into ±300 m³eq cumulative
  errors per kg. The asymmetric gate isolates the consumption signal
  from the noise.

Output: a ``(1, n_activities)`` sparse correction row added to
``ScoringPackage.corrections`` for the water-use method, summed with the
existing :class:`RegionalCorrectionBuilder` output. At score time the
sum is ``q @ B @ supply + correction @ supply`` — same shape, same fast
path, no per-activity logic at scoring time.

The asymmetry threshold (0.95), minimum net (0.05 m³), and fallback CF
(7.0) are the configuration triple proven against the 2026-05-14 fleet
backtest: median ``|Δ|`` 29 % → 18 %, outliers 324 → 202, median signed
−10.8 % → −0.2 %. See docs/FIX_WATER_USE_AUGMENTATION.md (revised) for
the derivation.

Known residual: dehydrated soups (~50 products) over-count because
their LCI records process water as ``Water [water, surface water]``
emission when it physically evaporates. That's an inventory bookkeeping
issue addressed in a separate workstream — see
docs/IMPROVEMENT_ROADMAP.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy import sparse as sp

from scoring.matrix_builder import BuiltMatrix


@dataclass(frozen=True)
class AwareConsumptionCorrectionBuilder:
    """Compose the water-use AWARE net-consumption correction row."""

    # Resource-side bio3 (name_lower, top_compartment_lower).
    # Aligned with SimaPro EF v3.1 (adapted)'s 3 positive resource flows.
    RESOURCE_TARGETS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("water, lake", "natural resource"),
        ("water, river", "natural resource"),
        ("water, turbine use, unspecified natural origin", "natural resource"),
    )

    # Return-side bio3 (name_lower, top_compartment_lower, sub_compartment_lower).
    # Empty sub means the catalog row has no second category element.
    # Ocean is intentionally excluded (saltwater not characterised).
    RETURN_TARGETS: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("water", "water", ""),
        ("water", "water", "fossil well"),
        ("water", "water", "ground-"),
        ("water", "water", "ground-, long-term"),
        ("water", "water", "surface water"),
    )

    # Tuning constants (validated 2026-05-14 against fleet backtest).
    ASYMMETRY_THRESHOLD: ClassVar[float] = 0.95
    MIN_NET_M3: ClassVar[float] = 0.05
    FALLBACK_CF: ClassVar[float] = 7.0

    # Codes carrying the ``@<region>`` separator are synthetic regional
    # rows emitted by :class:`matching.biosphere.BiosphereMatcher`. They
    # already carry a per-region SimaPro CF on the Q vector, so summing
    # them into this per-activity AWARE correction would double-count.
    # We exclude them at the catalog-resolution stage so AwareConsumption
    # only sees the base (un-regionalised) bio3 rows that still need a
    # location-based estimate.
    SYNTHETIC_CODE_SEPARATOR: ClassVar[str] = "@"

    def build(
        self,
        *,
        biosphere: BuiltMatrix,
        technosphere: BuiltMatrix,
        biosphere_catalog: pd.DataFrame,
        regional_cf_by_location: dict[str, float],
        col_id_to_location: dict[int, str],
    ) -> sp.csr_matrix:
        """Return a ``(1, n_activities)`` sparse correction row.

        ``biosphere_catalog`` must carry the columns ``database``,
        ``code``, ``name``, ``categories`` (list of two strings: top
        then sub). ``regional_cf_by_location`` maps ISO-3166 alpha-2 to
        the regional AWARE CF magnitude. ``col_id_to_location`` maps
        technosphere column ids (the ``output_id`` hash) to the
        activity's location string.

        Empty (no nonzero entries) if no bio rows resolve or no
        activities pass the inclusion gate.
        """
        n_act = technosphere.matrix.shape[1]
        if n_act == 0:
            return sp.csr_matrix((1, 0))

        res_rows = self._resolve_resource_rows(biosphere_catalog, biosphere.row_id_to_idx)
        ret_rows = self._resolve_return_rows(biosphere_catalog, biosphere.row_id_to_idx)
        if not res_rows or not ret_rows:
            return sp.csr_matrix((1, n_act))

        b_csr = biosphere.matrix.tocsr()
        res_per_col = np.asarray(b_csr[res_rows, :].sum(axis=0)).ravel()
        ret_per_col = np.asarray(b_csr[ret_rows, :].sum(axis=0)).ravel()

        max_side = np.maximum(res_per_col, ret_per_col)
        min_side = np.minimum(res_per_col, ret_per_col)
        with np.errstate(divide="ignore", invalid="ignore"):
            asym = np.where(max_side > 0, (max_side - min_side) / max_side, 0.0)
        net = res_per_col - ret_per_col

        include = (
            (asym >= self.ASYMMETRY_THRESHOLD)
            & (net >= self.MIN_NET_M3)
            & (res_per_col > ret_per_col)
        )
        if not include.any():
            return sp.csr_matrix((1, n_act))

        cf_per_col = self._regional_cf_vector(
            n_act=n_act,
            col_id_to_location=col_id_to_location,
            col_id_to_idx=technosphere.col_id_to_idx,
            regional_cf_by_location=regional_cf_by_location,
        )

        delta = np.zeros(n_act, dtype="float64")
        delta[include] = cf_per_col[include] * net[include]
        if not np.any(delta):
            return sp.csr_matrix((1, n_act))
        return sp.csr_matrix(delta.reshape(1, -1))

    # ------------------------------------------------------------------

    @classmethod
    def _resolve_resource_rows(
        cls,
        biosphere_catalog: pd.DataFrame,
        row_id_to_idx: dict[int, int],
    ) -> list[int]:
        return cls._collect_rows(
            biosphere_catalog=biosphere_catalog,
            row_id_to_idx=row_id_to_idx,
            targets=cls.RESOURCE_TARGETS,
            with_sub=False,
        )

    @classmethod
    def _resolve_return_rows(
        cls,
        biosphere_catalog: pd.DataFrame,
        row_id_to_idx: dict[int, int],
    ) -> list[int]:
        return cls._collect_rows(
            biosphere_catalog=biosphere_catalog,
            row_id_to_idx=row_id_to_idx,
            targets=cls.RETURN_TARGETS,
            with_sub=True,
        )

    @classmethod
    def _collect_rows(
        cls,
        *,
        biosphere_catalog: pd.DataFrame,
        row_id_to_idx: dict[int, int],
        targets: tuple,
        with_sub: bool,
    ) -> list[int]:
        """Resolve target flow descriptors → matrix row indices.

        Delegates flow-id hashing to ``ExchangeFrameBuilder.flow_id_for``
        so the keys we produce match the ones the matrix builder used.
        """
        # Local import to avoid the cyclic chain at module load time.
        from scoring.exchange_frame_builder import ExchangeFrameBuilder

        if biosphere_catalog.empty:
            return []
        bc = biosphere_catalog
        # We scope to the two databases the matrix actually carries.
        bc = bc[bc["database"].isin(("ecoinvent-3.9.1-biosphere", "biosphere3"))].copy()
        if bc.empty:
            return []
        bc["_name_l"] = bc["name"].astype(str).str.lower().str.strip()
        bc["_top"] = bc["categories"].apply(cls._top)
        bc["_sub"] = bc["categories"].apply(cls._sub)

        out: list[int] = []
        for target in targets:
            if with_sub:
                name_l, top_l, sub_l = target
                mask = (
                    (bc["_name_l"] == name_l.lower())
                    & (bc["_top"].str.lower() == top_l.lower())
                    & (bc["_sub"].str.lower() == sub_l.lower())
                )
            else:
                name_l, top_l = target
                mask = (bc["_name_l"] == name_l.lower()) & (bc["_top"].str.lower() == top_l.lower())
            for row in bc[mask].itertuples(index=False):
                db = str(getattr(row, "database", ""))
                code = str(getattr(row, "code", ""))
                if not db or not code:
                    continue
                # Synthetic ``<base>@<region>`` rows are kept in the
                # sweep. The flow-level Q-vector CF emission for
                # synthetic codes is DISABLED (see the
                # ``SpRegionalWaterCfLoader`` block in the build CLI
                # for the failure modes), so synthetic rows carry no
                # CF of their own and the activity-location estimate
                # in this builder remains the only source of regional
                # signal. Excluding them would silently zero-out every
                # regionally-tagged water flow.
                fid = ExchangeFrameBuilder.flow_id_for((db, code))
                idx = row_id_to_idx.get(int(fid))
                if idx is not None:
                    out.append(int(idx))
        # Dedup while preserving order; CSR row-slice is fine on dup
        # indices but cheaper without them.
        seen: set[int] = set()
        dedup: list[int] = []
        for i in out:
            if i not in seen:
                seen.add(i)
                dedup.append(i)
        return dedup

    @classmethod
    def _regional_cf_vector(
        cls,
        *,
        n_act: int,
        col_id_to_location: dict[int, str],
        col_id_to_idx: dict[int, int],
        regional_cf_by_location: dict[str, float],
    ) -> np.ndarray:
        """``cf_per_col[j] = regional_AWARE(location(j))`` with fallback."""
        cf_per_col = np.full(n_act, cls.FALLBACK_CF, dtype="float64")
        if col_id_to_location and regional_cf_by_location:
            for col_id, loc in col_id_to_location.items():
                if not isinstance(loc, str) or not loc:
                    continue
                idx = col_id_to_idx.get(int(col_id))
                if idx is None:
                    continue
                # Collapse sub-regional codes (CA-QC, US-WECC) to country
                # prefix; aggregate codes (RoW, GLO, ...) would be empty
                # strings out of ActivityLocationParser already.
                head = loc.split("-", 1)[0]
                cf = regional_cf_by_location.get(head)
                if cf is not None:
                    cf_per_col[idx] = float(cf)
        return cf_per_col

    @staticmethod
    def _top(cats: object) -> str:
        if cats is None:
            return ""
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if isinstance(cats, (list, tuple)) and len(cats) >= 1:
            return str(cats[0])
        return ""

    @staticmethod
    def _sub(cats: object) -> str:
        if cats is None:
            return ""
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if isinstance(cats, (list, tuple)) and len(cats) >= 2:
            return str(cats[1])
        return ""
