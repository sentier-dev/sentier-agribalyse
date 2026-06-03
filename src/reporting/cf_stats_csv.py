"""``CfStatsEmitter`` — write ``dashboard/cf_stats.csv``.

Translates the ``list[MethodRow]`` produced by
:class:`cli.compare_cfs.CfComparator` into a flat CSV consumed by the
CF-stats tab in ``dashboard/backtest_dashboard.html``.

Columns per row:

* ``method`` — display name (registry category, with SimaPro
  ``- inorganics`` / ``- organics`` already folded into the root by the
  comparator).
* ``alignment`` ∈ ``{"both", "sp_only", "ef_only"}``.
* ``sp_<stat>`` / ``ef_<stat>`` — raw stat values (blank when that side
  is missing).
* ``diff_<stat>`` — symmetric percent difference
  ``(ef − sp) / max(|sp|, |ef|)`` as a fraction. Bounded in ``[-2, +2]``:
  ``[-1, +1]`` when ``sp`` and ``ef`` share a sign, beyond that when they
  flip sign (e.g. ``sp=-1, ef=+1`` → ``+2.0``). Returns ``0`` when both
  sides are exactly ``0``; blank only when one side is missing.

The symmetric form replaces the historical ``(ef − sp) / |sp|`` because the
unbounded variant produced misleading values like ``+9900%`` when ``sp`` was
near zero. The dashboard multiplies ``diff_*`` by 100 for display, so the
unit stays consistent with the existing ``backtest_pass1.csv`` %-diff
columns. Values past ``±100%`` flag a sign mismatch worth investigating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

if TYPE_CHECKING:
    from cli.compare_cfs import MethodRow


@dataclass(frozen=True)
class CfStatsEmitter:
    out_path: Path

    STAT_KEYS: ClassVar[tuple[str, ...]] = (
        "count",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "sum",
    )

    HEADER: ClassVar[tuple[str, ...]] = (
        "method",
        "alignment",
        "sp_count",
        "ef_count",
        "sp_min",
        "ef_min",
        "sp_max",
        "ef_max",
        "sp_mean",
        "ef_mean",
        "sp_median",
        "ef_median",
        "sp_std",
        "ef_std",
        "sp_sum",
        "ef_sum",
        "diff_count",
        "diff_min",
        "diff_max",
        "diff_mean",
        "diff_median",
        "diff_std",
        "diff_sum",
    )

    def write(self, rows: list[MethodRow]) -> Path:
        records: list[dict[str, object]] = [self._row_to_record(r) for r in rows]
        df = pd.DataFrame(records, columns=list(self.HEADER))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_path, index=False)
        return self.out_path

    def _row_to_record(self, row: MethodRow) -> dict[str, object]:
        sp = row.simapro_stats
        ef = row.ef31_stats
        pf = row.per_flow_diff_stats
        alignment = self._alignment(sp, ef)
        record: dict[str, object] = {
            "method": row.display_name,
            "alignment": alignment,
        }
        for stat in self.STAT_KEYS:
            record[f"sp_{stat}"] = sp[stat] if sp is not None else ""
            record[f"ef_{stat}"] = ef[stat] if ef is not None else ""
            record[f"diff_{stat}"] = self._diff(sp, ef, stat, pf)
        return record

    @staticmethod
    def _alignment(sp: dict | None, ef: dict | None) -> str:
        if sp is not None and ef is not None:
            return "both"
        if sp is not None:
            return "sp_only"
        return "ef_only"

    # Stats for which the joined-mode diff stays as the historical
    # ``(ef_<stat> - sp_<stat>) / max(|sp|, |ef|)`` formula:
    #
    # * ``count`` — diff_count is the coverage gap (sp_count = matched
    #   flows; ef_count = total registry flows; diff_count tells the
    #   customer "how much of the registry SP covers").
    # * ``sum``   — diff_sum is total CF-mass agreement; aggregating per-
    #   flow diffs into a sum is unbounded and uninterpretable.
    _PER_FLOW_SKIP: ClassVar[frozenset[str]] = frozenset({"count", "sum"})

    @classmethod
    def _diff(
        cls,
        sp: dict | None,
        ef: dict | None,
        stat: str,
        per_flow: dict | None = None,
    ) -> object:
        if sp is None or ef is None:
            return ""
        # Joined-mode (per-flow stats available): diff_<stat> = <stat> of
        # per-flow relative diffs for min/max/mean/median/std. Each per-
        # flow diff is bounded in [-2, +2], so aggregating gives a
        # customer-meaningful "% per-flow disagreement" without the
        # blow-up of "diff of summary stats" when one side's stat is
        # near zero. ``count`` and ``sum`` keep the historical formula.
        if per_flow is not None and stat not in cls._PER_FLOW_SKIP:
            return per_flow[stat]
        sp_v = sp[stat]
        ef_v = ef[stat]
        denom = max(abs(sp_v), abs(ef_v))
        if denom == 0:
            return 0.0
        return (ef_v - sp_v) / denom
