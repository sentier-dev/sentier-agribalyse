"""``PerennialLifecycleFilter`` — drop acid-relevant biosphere edges on perennial non-prod / grubbing-up.

ADEME's published synthese excludes acidification emissions (NH3) of
plantation establishment and grubbing-up lifecycle stages — a common
LCA capital-goods convention applied to *combustion-style* emissions
but **not** to biogenic CO2 or land use, which ADEME's reference
*does* include from those same activities.

AGB's SimaPro recipes include all biosphere emissions of the
non-productive / grubbing-up stages and amortise them across the
productive output. For the 14 spices that share the
``Black pepper, dried, consumption mix {FR}`` proxy, the NH3 piece
drives every SKU to +55.5 % over ADEME's acidification reference.

This filter drops only the **ammonia** biosphere edges on the
non-productive / grubbing-up activities, leaving the carbon / land /
water flows intact so biogenic CC and land use methods stay aligned.
First attempt (2026-05-17) cut the aggregator → non-productive
technosphere edges entirely; closed acid 14 → 1 but introduced 14
new biogenic-CC and 14 new land-use outliers at −60 % / −53 % (the
spec's capital-goods premise was wrong for those methods). The
biosphere-level filter is the surgical fix.

Scope
-----

Black pepper (Vietnam) only. The same naming pattern exists for
``Kiwifruit FR`` but its SKU
(``Kiwi, pulpe et graines, cru``) passes baseline acidification at
+10.2 %; dropping its NH3 would regress it. Expand scope after
measuring per-SKU effect.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from core.logging import Logging
from scoring.exchange_frame import ExchangeFrame
from scoring.exchange_frame_builder import ExchangeFrameBuilder


@dataclass(frozen=True)
class PerennialLifecycleFilter:
    """Stateless: ``purge(frame, sp_data=..., biosphere_catalog=...)``
    returns a frame with the acid-relevant biosphere edges on
    non-productive / grubbing-up activities dropped, plus a stats
    dict summarising what was removed."""

    biosphere_catalog: pd.DataFrame
    target_db_name: str = "agribalyse-3.2"

    # AGB-side activity names whose biosphere emissions the filter
    # surgically reduces. Matches by name (resilient to code refresh)
    # and resolves codes via ``ExchangeFrameBuilder.flow_id_for``.
    NON_PRODUCTIVE_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "Black pepper berries, fresh, non productive years {VN}",
            "Black pepper berries, fresh, non productive years {VN} U",
            "Black pepper berries, fresh, grubbing up {VN}",
            "Black pepper berries, fresh, grubbing up {VN} U",
            "Black Pepper bells, fresh, non productive years {VN}",
            "Black Pepper bells, fresh, non productive years {VN} U",
            "Black Pepper bells, fresh, grubbing up {VN}",
            "Black Pepper bells, fresh, grubbing up {VN} U",
        }
    )

    # Lowercased biosphere flow names the filter drops on those
    # activities. NH3 is the dominant acidification contributor in
    # the spice cluster decomposition; NOx and SOx are sub-1 % and
    # are left untouched to keep the filter minimal. Expand only if
    # measurement justifies it.
    ACID_FLOW_NAMES: ClassVar[frozenset[str]] = frozenset({"ammonia"})

    SCORED_DATABASES: ClassVar[tuple[str, ...]] = (
        "ecoinvent-3.9.1-biosphere",
        "biosphere3",
    )

    def purge(
        self, frame: ExchangeFrame, *, sp_data: Iterable[dict]
    ) -> tuple[ExchangeFrame, dict[str, Any]]:
        target_activity_ids = self._resolve_target_activity_ids(sp_data)
        target_flow_ids = self._resolve_target_flow_ids()
        if not target_activity_ids or not target_flow_ids:
            return frame, {
                "dropped": 0,
                "n_distinct_activities": 0,
                "n_distinct_flows": 0,
            }

        df = frame.df
        mask = (
            df["is_biosphere"]
            & df["output_id"].isin(target_activity_ids)
            & df["input_id"].isin(target_flow_ids)
        )
        if not mask.any():
            return frame, {
                "dropped": 0,
                "n_distinct_activities": 0,
                "n_distinct_flows": 0,
            }

        dropped = df.loc[mask]
        n_dropped = int(mask.sum())
        n_activities = int(dropped["output_id"].nunique())
        n_flows = int(dropped["input_id"].nunique())

        log = Logging.get(__name__)
        log.info(
            "scoring.perennial_lifecycle.dropped",
            n_dropped=n_dropped,
            n_distinct_activities=n_activities,
            n_distinct_flows=n_flows,
            sample=self._sample_first(dropped),
        )

        out_df = df.loc[~mask].reset_index(drop=True)
        return ExchangeFrame.from_long(out_df), {
            "dropped": n_dropped,
            "n_distinct_activities": n_activities,
            "n_distinct_flows": n_flows,
        }

    def _resolve_target_activity_ids(self, sp_data: Iterable[dict]) -> frozenset[int]:
        names = self.NON_PRODUCTIVE_NAMES
        out: set[int] = set()
        for act in sp_data:
            if act.get("database") != self.target_db_name:
                continue
            name = act.get("name")
            if name is None or str(name) not in names:
                continue
            code = act.get("code")
            if code is None:
                continue
            out.add(ExchangeFrameBuilder.flow_id_for((self.target_db_name, str(code))))
        return frozenset(out)

    def _resolve_target_flow_ids(self) -> frozenset[int]:
        bc = self.biosphere_catalog
        if bc is None or bc.empty:
            return frozenset()
        names = self.ACID_FLOW_NAMES
        scored = self.SCORED_DATABASES
        out: set[int] = set()
        for _, row in bc[bc["database"].isin(scored)].iterrows():
            n = str(row["name"]).strip().lower()
            if n not in names:
                continue
            out.add(ExchangeFrameBuilder.flow_id_for((str(row["database"]), str(row["code"]))))
        return frozenset(out)

    @staticmethod
    def _sample_first(rows: pd.DataFrame) -> dict[str, Any]:
        if rows.empty:
            return {}
        first = rows.iloc[0]
        return {
            "output_id": int(first["output_id"]),
            "input_id": int(first["input_id"]),
            "amount": float(first["amount"]),
            "edge_type": str(first["edge_type"]),
        }
