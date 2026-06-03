"""``DanglingEdgeAuditor`` — explain matrix drops one row at a time.

The matrix-emit path in ``LinkAllPipeline._emit_scoring_package`` has
three destruction points after the long-form frame is concatenated:

1. ``Allocator`` rewrites every ``output_id`` to a synthetic id derived
   from ``(activity_id, product_id)`` and drops activities that have no
   production row. The original (parent) ``output_id`` disappears from
   the column space.
2. ``ProductDeduplicator`` keeps only the smallest ``output_id``
   producer per product — every other producer is dropped, taking its
   non-production edges with it.
3. ``DanglingEdgePruner`` drops zero-amount production rows, self-loops,
   and consumer edges whose input has no producer.

This class snapshots the (activity, product) universe before each step,
attributes every missing producer / dangling product to the first stage
that dropped it, and writes ``dashboard/dangling_edges.parquet``. One
row per dropped activity or dangling-consumed product, with stage,
database, code, name, location, and (for products) the count of
consumer edges that referenced it before the pruner ran.

The auditor is read-only: it does not modify the frames it inspects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from core.logging import Logging
from scoring.exchange_frame import ExchangeFrame
from scoring.exchange_frame_builder import ExchangeFrameBuilder

_PRODUCTION_TYPES = frozenset({"production", "generic production"})
_TECHNOSPHERE_TYPES = frozenset({"technosphere", "substitution"})


@dataclass(frozen=True)
class IdNameResolver:
    """Reverse map from hashed ``(database, code)`` ids to metadata.

    Built once per run from ``sp.data`` (covers AGB activities + every
    exchange input referenced by an AGB process) and
    ``registry/ecoinvent_catalog.parquet`` (covers the ecoinvent supply
    chain). Lookup returns ``database / code / name / location / type``;
    falls back to ``(unknown)`` with the integer id as code when the
    resolver has no entry — typically a synthetic activity id minted by
    ``Allocator``.
    """

    by_id: Mapping[int, Mapping[str, str]]

    @classmethod
    def from_sources(
        cls,
        sp_data: Iterable[Mapping[str, Any]],
        ei_catalog: pd.DataFrame,
    ) -> IdNameResolver:
        by_id: dict[int, dict[str, str]] = {}
        for ds in sp_data:
            db = str(ds.get("database") or "")
            code = str(ds.get("code") or "")
            if db and code:
                key = ExchangeFrameBuilder.flow_id_for((db, code))
                by_id.setdefault(
                    key,
                    {
                        "database": db,
                        "code": code,
                        "name": str(ds.get("name") or ""),
                        "location": str(ds.get("location") or ""),
                        "type": str(ds.get("type") or "process"),
                    },
                )
            for exc in ds.get("exchanges", ()) or ():
                inp = exc.get("input")
                if inp is None:
                    continue
                db_i = str(inp[0])
                code_i = str(inp[1])
                key = ExchangeFrameBuilder.flow_id_for((db_i, code_i))
                if key in by_id:
                    continue
                by_id[key] = {
                    "database": db_i,
                    "code": code_i,
                    "name": str(exc.get("name") or ""),
                    "location": "",
                    "type": str(exc.get("type") or "product"),
                }
        for row in ei_catalog.itertuples(index=False):
            key = ExchangeFrameBuilder.flow_id_for((row.database, row.code))
            # Ecoinvent catalog is authoritative for ecoinvent flows.
            by_id[key] = {
                "database": str(row.database),
                "code": str(row.code),
                "name": str(row.name or ""),
                "location": str(row.location or ""),
                "type": "process",
                "reference_product": str(row.reference_product or ""),
            }
        return cls(by_id=by_id)

    def lookup(self, id_: int) -> Mapping[str, str]:
        return self.by_id.get(
            int(id_),
            {
                "database": "",
                "code": str(int(id_)),
                "name": "(unknown — likely synthetic id)",
                "location": "",
                "type": "",
            },
        )


@dataclass(frozen=True)
class FrameSnapshot:
    """Activity and product universe at one pipeline stage.

    Only positive-production rows (amount > 0) contribute, matching the
    matrix builder's definition of "really produced"."""

    stage: str
    activity_ids: frozenset[int]
    product_ids: frozenset[int]


@dataclass
class DanglingEdgeAuditor:
    """Stateful: accumulate per-stage snapshots, emit a parquet at the end."""

    resolver: IdNameResolver
    snapshots: list[FrameSnapshot] = field(default_factory=list)
    consumer_edges: pd.DataFrame | None = None

    _PRODUCTION_TYPES: ClassVar[frozenset[str]] = _PRODUCTION_TYPES
    _TECHNOSPHERE_TYPES: ClassVar[frozenset[str]] = _TECHNOSPHERE_TYPES

    def snapshot(self, stage: str, frame: ExchangeFrame) -> None:
        df = frame.df
        prod_pos = df[df["edge_type"].isin(self._PRODUCTION_TYPES) & (df["amount"] > 0)]
        self.snapshots.append(
            FrameSnapshot(
                stage=stage,
                activity_ids=frozenset(prod_pos["output_id"].astype("int64").tolist()),
                product_ids=frozenset(prod_pos["input_id"].astype("int64").tolist()),
            )
        )

    def capture_consumer_edges(self, frame: ExchangeFrame) -> None:
        """Snapshot the technosphere edges right before ``DanglingEdgePruner``."""
        df = frame.df
        tech = df[df["edge_type"].isin(self._TECHNOSPHERE_TYPES)]
        self.consumer_edges = tech[["output_id", "input_id", "amount"]].copy()

    def write(self, out_path: Path) -> dict[str, Any]:
        if len(self.snapshots) < 2:
            return {"snapshots": len(self.snapshots), "rows": 0}
        final = self.snapshots[-1]
        rows: list[dict[str, Any]] = []
        dropped_activities = self._attribute_drops("activity_ids", final.activity_ids)
        for aid, stage in dropped_activities.items():
            info = self.resolver.lookup(aid)
            rows.append(
                {
                    "category": "dropped_activity",
                    "id": int(aid),
                    "database": info.get("database", ""),
                    "code": info.get("code", ""),
                    "name": info.get("name", ""),
                    "location": info.get("location", ""),
                    "drop_stage": stage,
                    "consumer_count": 0,
                }
            )
        # All products consumed (pre-pruner) but absent from the final
        # matrix — the *union* of (initially produced + lost) and
        # (consumed but never produced).
        consumer_counts = self._consumer_counts_for(final.product_ids)
        initial_lost = self.snapshots[0].product_ids - final.product_ids
        all_dangling = set(consumer_counts.keys()) | initial_lost
        for pid in all_dangling:
            info = self.resolver.lookup(pid)
            stage = self._first_stage_where_product_missing(pid)
            rows.append(
                {
                    "category": "dangling_product",
                    "id": int(pid),
                    "database": info.get("database", ""),
                    "code": info.get("code", ""),
                    "name": info.get("name", ""),
                    "location": info.get("location", ""),
                    "drop_stage": stage,
                    "consumer_count": int(consumer_counts.get(pid, 0)),
                }
            )
        df = pd.DataFrame(
            rows,
            columns=[
                "category",
                "id",
                "database",
                "code",
                "name",
                "location",
                "drop_stage",
                "consumer_count",
            ],
        )
        if not df.empty:
            df = df.sort_values(
                ["category", "consumer_count", "name"],
                ascending=[True, False, True],
            ).reset_index(drop=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, compression=None)
        log = Logging.get(__name__)
        stats: dict[str, Any] = {
            "snapshots": [s.stage for s in self.snapshots],
            "dropped_activities": len(dropped_activities),
            "dangling_products": len(all_dangling),
            "max_consumer_count": int(df["consumer_count"].max()) if not df.empty else 0,
            "by_stage": (df.groupby("drop_stage").size().to_dict() if not df.empty else {}),
            "out_path": str(out_path.name),
        }
        log.info("scoring.dangling_edges.audit", **stats)
        return stats

    def _attribute_drops(self, attr: str, final_set: frozenset[int]) -> dict[int, str]:
        if not self.snapshots:
            return {}
        initial = getattr(self.snapshots[0], attr)
        all_dropped = initial - final_set
        if not all_dropped:
            return {}
        attribution: dict[int, str] = {}
        for i in range(1, len(self.snapshots)):
            prev_set = getattr(self.snapshots[i - 1], attr)
            curr_set = getattr(self.snapshots[i], attr)
            dropped_here = (prev_set - curr_set) & all_dropped
            for x in dropped_here:
                if x not in attribution:
                    attribution[x] = self.snapshots[i].stage
        for x in all_dropped:
            if x not in attribution:
                attribution[x] = "before_initial_snapshot"
        return attribution

    def _first_stage_where_product_missing(self, pid: int) -> str:
        """Walk snapshots forward; return the stage where ``pid`` first
        disappears from ``product_ids``. If absent from every snapshot,
        the product was consumed but never produced — return
        ``"never_produced"``."""
        for snap in self.snapshots:
            if pid in snap.product_ids:
                continue
            if snap is self.snapshots[0]:
                return "never_produced"
            return snap.stage
        # Present in all snapshots — shouldn't reach here when caller
        # already filtered to dangling products, but stay safe.
        return "still_present"

    def _consumer_counts_for(self, final_products: frozenset[int]) -> Mapping[int, int]:
        if self.consumer_edges is None or self.consumer_edges.empty:
            return {}
        df = self.consumer_edges
        orphans = df[~df["input_id"].astype("int64").isin(final_products)]
        if orphans.empty:
            return {}
        return orphans.groupby("input_id").size().to_dict()
