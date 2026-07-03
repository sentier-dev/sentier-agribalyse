"""CF-comparison data for the dashboard, built on the deterministic 1:1 join.

The CF-comparison tab compares the SimaPro adapted EF 3.1 CFs against the built
per-method registry CFs the scoring pipeline uses. The match is the
**per-flow 1:1 join** produced by :class:`ef.cf_flow_join.FlowLevelCfJoiner`:
every registry biosphere flow ``code`` is matched to *exactly one* SimaPro CF
(via full ``(method, name, compartment, sub_compartment)`` identity with
synonym / CAS / short-name fallbacks), so there are no "candidates" to choose
between — the multiplicity the old bucketed comparison produced was an artefact
of collapsing flows to a coarse compartment bucket.

Three classes, all OOP per ``CLAUDE.md`` (frozen dataclasses, DI, no
module-level behaviour):

* :class:`CfComparisonJoinBuilder` — turn the per-method ``JoinedFlowFrame``s
  into one flat join DataFrame (one row per ``(method, code)``), computing the
  SimaPro/registry compartment paths, the agree/differ status, and the
  disagreement metrics. This is the complete join (matched + registry-only),
  persisted to ``registry/cf_comparison_join.parquet`` for auditing.
* :class:`CfComparisonCsvEmitter` — project the join onto the dashboard CSV
  schema and write ``dashboard/cf_comparison.csv``. By default only the
  **matched** rows are written (the genuine molecule-to-molecule comparisons);
  registry-only rows live in the parquet.
* :class:`CfComparisonByCodeBuilder` — the per-``code`` SimaPro CF sidecar
  (``registry/cf_comparison_by_code.parquet``) consumed by
  :class:`reporting.SimaProCfLookup` for the flow-decomposition toggle.

How the match was made is surfaced as ``match_provenance`` (``exact_name`` /
``synonym`` / ``cas`` / ``short_name``) — the reconciliation signal showing
*which* SimaPro flow a registry flow resolved against.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from ef.cf_flow_join import ContextNormaliser, JoinedFlowFrame

# Two CFs agree when within this absolute or relative tolerance. Mirrors the
# thresholds the prior comparison used so the agree/differ split is stable.
AGREE_ABS = 1e-9
AGREE_REL = 1e-4

# Matched-comparison statuses (a SimaPro CF was found for the registry flow).
STATUS_AGREE = "both_agree"
STATUS_DIFFER = "both_differ"
STATUS_REGISTRY_ONLY = "registry_only"
_MATCHED_STATUSES = frozenset({STATUS_AGREE, STATUS_DIFFER})

# Registry top-bucket (via ContextNormaliser) → coarse comparison bucket used by
# the dashboard's compartment filter.
_BUCKET_BY_SP_COMPARTMENT = {"Air": "air", "Water": "water", "Soil": "soil", "Raw": "resource"}

# Sub-compartment values treated as "no sub" and dropped from the display path.
_UNSPECIFIED_SUBS = frozenset({"", "(unspecified)", "unspecified", "nan", "none"})


@dataclass(frozen=True)
class CfComparisonJoinBuilder:
    """Flatten per-method :class:`JoinedFlowFrame`s into the comparison join.

    One row per ``(method, code)``. ``normaliser`` derives the coarse
    compartment bucket from each registry flow's ``categories`` (reusing the
    same context logic the joiner used to match), so the bucket the dashboard
    filters on is consistent with how the flow was matched.
    """

    normaliser: ContextNormaliser

    # Full join schema (also the parquet schema). The CSV emitter projects a
    # comparison-first subset of these in a stable order.
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "method",
        "code",
        "database",
        "compartment",
        "compartment_simapro",
        "compartment_registry",
        "cas",
        "name_simapro",
        "name_registry",
        "cf_simapro",
        "cf_registry",
        "sp_reg_ratio",
        "rel_diff",
        "abs_diff",
        "status",
        "match_provenance",
    )

    def build(self, frames: Iterable[JoinedFlowFrame]) -> pd.DataFrame:
        rows: list[dict] = []
        for frame in frames:
            method = frame.method_key[2]
            for r in frame.df.itertuples(index=False):
                rows.append(self._row(method, r))
        return pd.DataFrame(rows, columns=list(self.COLUMNS))

    def _row(self, method: str, r) -> dict:
        ef_cf = float(r.ef_cf)
        sp_raw = r.sp_cf
        matched = sp_raw is not None and not pd.isna(sp_raw)
        categories = r.categories
        bucket = self._bucket(categories)
        comp_registry = self._format_parts(self._as_list(categories))
        if matched:
            sp_cf = float(sp_raw)
            abs_diff = abs(sp_cf - ef_cf)
            # ``has_ref`` asks "is the registry CF a usable denominator?",
            # which means *non-zero* — NOT "bigger than the agree tolerance".
            # Toxicity CFs are routinely far below ``AGREE_ABS`` (down to
            # ~1e-45) yet are perfectly good references. Gating on
            # ``AGREE_ABS`` here was the bug behind "agree despite a >50%
            # gap": tiny CFs fell through to the absolute shortcut and any
            # sub-1e-9 difference read as agreement, hiding relative gaps of
            # many orders of magnitude.
            has_ref = ef_cf != 0.0
            rel_diff = abs_diff / abs(ef_cf) if has_ref else np.nan
            ratio = sp_cf / ef_cf if has_ref else np.nan
            # With a reference, agreement is purely the *relative* gap. The
            # absolute floor only resolves the no-reference case (registry CF
            # is exactly zero — agree iff SimaPro is essentially zero too).
            agree = rel_diff < AGREE_REL if has_ref else abs_diff < AGREE_ABS
            status = STATUS_AGREE if agree else STATUS_DIFFER
            comp_simapro = self._format_parts([r.sp_compartment, r.sp_sub_compartment])
            name_simapro = r.sp_name or None
            provenance = r.sp_match_provenance
        else:
            sp_cf = np.nan
            abs_diff = rel_diff = ratio = np.nan
            status = STATUS_REGISTRY_ONLY
            comp_simapro = ""
            name_simapro = None
            provenance = r.sp_match_provenance  # "unmatched"
        return {
            "method": method,
            "code": r.code,
            "database": r.database,
            "compartment": bucket,
            "compartment_simapro": comp_simapro,
            "compartment_registry": comp_registry,
            "cas": (r.cas or None) if isinstance(r.cas, str) else None,
            "name_simapro": name_simapro,
            "name_registry": r.name or None,
            "cf_simapro": sp_cf,
            "cf_registry": ef_cf,
            "sp_reg_ratio": ratio,
            "rel_diff": rel_diff,
            "abs_diff": abs_diff,
            "status": status,
            "match_provenance": provenance,
        }

    def _bucket(self, categories) -> str:
        comp, _sub = self.normaliser.normalise(self._as_list(categories))
        return _BUCKET_BY_SP_COMPARTMENT.get(comp, comp.lower())

    @staticmethod
    def _as_list(categories) -> list[str]:
        if categories is None:
            return []
        if isinstance(categories, (list, tuple, np.ndarray)):
            return [str(c) for c in categories]
        return [str(categories)]

    @staticmethod
    def _format_parts(parts: Sequence[str]) -> str:
        """Join a compartment path, dropping blank / ``(unspecified)`` tails.

        ``["air", "(unspecified)"]`` → ``"air"``; ``["Air", "indoor"]`` →
        ``"Air / indoor"``. A leading part is always kept even if unspecified.
        """
        cleaned = [str(p).strip() for p in parts if p is not None and str(p).strip() != ""]
        while len(cleaned) > 1 and cleaned[-1].lower() in _UNSPECIFIED_SUBS:
            cleaned.pop()
        return " / ".join(cleaned)


@dataclass(frozen=True)
class UsedFlowFilter:
    """Restrict a comparison join to flows the scoring pipeline actually uses.

    The CF comparison enumerates the full CF *reference* table (every flow with a
    CF, across the ``ecoinvent-3.9.1-biosphere`` and ``ef`` namespaces). Scoring
    only ever touches the biosphere flows the linked Agribalyse inventory emits —
    the rows of the scoring package's biosphere matrix. Those rows are keyed by
    the deterministic ``(database, code)`` hash
    (:meth:`scoring.exchange_frame_builder.ExchangeFrameBuilder.flow_id_for`), so
    a join row is "used" iff ``flow_id_for((database, code))`` is one of the
    package's biosphere row ids.

    Build with :meth:`from_biosphere_row_ids` from a loaded
    ``ScoringPackage.biosphere.row_id_to_idx``; then :meth:`filter` a join frame.
    """

    used_flow_ids: frozenset[int]

    @classmethod
    def from_biosphere_row_ids(cls, row_id_to_idx: dict) -> UsedFlowFilter:
        return cls(used_flow_ids=frozenset(int(k) for k in row_id_to_idx))

    def mask(self, df: pd.DataFrame) -> pd.Series:
        from scoring.exchange_frame_builder import ExchangeFrameBuilder

        used = self.used_flow_ids
        flags = [
            ExchangeFrameBuilder.flow_id_for((str(db), str(code))) in used
            for db, code in zip(df["database"], df["code"], strict=False)
        ]
        return pd.Series(flags, index=df.index)

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.mask(df)]


@dataclass(frozen=True)
class CfComparisonJoinLoader:
    """Read ``registry/cf_comparison_join.parquet`` into a DataFrame."""

    path: Path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"CF comparison join parquet not found: {self.path}. "
                "Run dds-compare-cfs to regenerate."
            )
        return pd.read_parquet(self.path)


@dataclass(frozen=True)
class CfComparisonCsvEmitter:
    """Project the CF comparison join onto the dashboard CSV schema.

    By default only matched rows (``both_agree`` / ``both_differ``) are written —
    the genuine molecule-to-molecule comparisons the tab is about. The complete
    join (including ``registry_only`` flows) is kept in the parquet.
    """

    out_path: Path
    matched_only: bool = True

    # Comparison-first column order: every SimaPro field beside its registry
    # counterpart so each (name, cf, compartment) pair reads as one comparison.
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "method",
        "compartment",
        "compartment_simapro",
        "compartment_registry",
        "cas",
        "name_simapro",
        "name_registry",
        "cf_simapro",
        "cf_registry",
        "rel_diff",
        "abs_diff",
        "status",
        "match_provenance",
    )

    def write(self, df: pd.DataFrame) -> Path:
        missing = [c for c in self.COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CF comparison join is missing expected columns: {missing}. "
                f"Got: {sorted(df.columns)}"
            )
        out = df
        if self.matched_only:
            out = out[out["status"].isin(_MATCHED_STATUSES)]
        projected = out.loc[:, list(self.COLUMNS)]
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # ``%.6g`` keeps 6 significant figures — ample for CF display — and
        # avoids pandas' full float repr (``1.0482100000000001``), which would
        # otherwise bloat the runtime-fetched CSV by several MB.
        projected.to_csv(self.out_path, index=False, float_format="%.6g")
        return self.out_path


@dataclass(frozen=True)
class CfComparisonByCodeBuilder:
    """Per-``code`` SimaPro CF sidecar for the flow-decomposition toggle.

    One row per matched ``(method category, registry code)`` carrying the
    SimaPro CF and how it was matched, in the schema
    :class:`reporting.SimaProCfLookup` expects (``code, method, cf_simapro,
    match_basis, name_simapro, name_registry``). Spans every method present in
    ``frames`` regardless of any display filter, so the toggle has full coverage.
    Deduped on ``(method, code)`` keeping the first occurrence.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "code",
        "method",
        "cf_simapro",
        "match_basis",
        "name_simapro",
        "name_registry",
    )

    def build(self, frames: Iterable[JoinedFlowFrame]) -> pd.DataFrame:
        rows: list[dict] = []
        for frame in frames:
            method = frame.method_key[2]
            for r in frame.df.itertuples(index=False):
                if r.sp_cf is None or pd.isna(r.sp_cf):
                    continue
                rows.append(
                    {
                        "code": str(r.code),
                        "method": method,
                        "cf_simapro": float(r.sp_cf),
                        "match_basis": r.sp_match_provenance,
                        "name_simapro": r.sp_name or "",
                        "name_registry": r.name or "",
                    }
                )
        df = pd.DataFrame(rows, columns=list(self.COLUMNS))
        return df.drop_duplicates(["method", "code"], keep="first").reset_index(drop=True)
