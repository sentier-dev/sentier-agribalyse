"""Unit tests for ``DanglingEdgeAuditor`` and ``IdNameResolver``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scoring.dangling_edge_auditor import (
    DanglingEdgeAuditor,
    IdNameResolver,
)
from scoring.exchange_frame import ExchangeFrame
from scoring.exchange_frame_builder import ExchangeFrameBuilder


def _frame(rows: list[tuple]) -> ExchangeFrame:
    df = pd.DataFrame(
        rows,
        columns=["output_id", "input_id", "amount", "edge_type", "is_biosphere"],
    )
    df["output_id"] = df["output_id"].astype("int64")
    df["input_id"] = df["input_id"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["edge_type"] = df["edge_type"].astype("string")
    df["is_biosphere"] = df["is_biosphere"].astype(bool)
    return ExchangeFrame(df=df)


_AID = ExchangeFrameBuilder.flow_id_for(("agb", "act-1"))
_PID = ExchangeFrameBuilder.flow_id_for(("agb", "prod-1"))
_AID2 = ExchangeFrameBuilder.flow_id_for(("agb", "act-2"))
_PID_ORPHAN = ExchangeFrameBuilder.flow_id_for(("agb", "orphan-prod"))


# ============================================================================
# IdNameResolver


class TestIdNameResolver:
    def test_resolves_activity_from_sp_data(self):
        sp_data = [
            {
                "database": "agb",
                "code": "act-1",
                "name": "wheat production FR",
                "location": "FR",
                "type": "process",
                "exchanges": [],
            }
        ]
        ei_catalog = pd.DataFrame(
            columns=["database", "code", "name", "unit", "location", "reference_product"]
        )
        r = IdNameResolver.from_sources(sp_data, ei_catalog)
        info = r.lookup(_AID)
        assert info["name"] == "wheat production FR"
        assert info["location"] == "FR"
        assert info["database"] == "agb"

    def test_resolves_exchange_input_product_name(self):
        sp_data = [
            {
                "database": "agb",
                "code": "act-1",
                "name": "consumer",
                "exchanges": [
                    {
                        "type": "technosphere",
                        "name": "wheat product (kg)",
                        "input": ("agb", "prod-1"),
                    },
                ],
            }
        ]
        ei_catalog = pd.DataFrame(
            columns=["database", "code", "name", "unit", "location", "reference_product"]
        )
        r = IdNameResolver.from_sources(sp_data, ei_catalog)
        info = r.lookup(_PID)
        assert info["name"] == "wheat product (kg)"
        assert info["database"] == "agb"

    def test_ecoinvent_catalog_overrides_sp_data_for_same_key(self):
        sp_data = [
            {
                "database": "ei",
                "code": "abc",
                "name": "fallback name",
                "exchanges": [],
            }
        ]
        ei_catalog = pd.DataFrame(
            [
                {
                    "database": "ei",
                    "code": "abc",
                    "name": "canonical ecoinvent name",
                    "unit": "kg",
                    "location": "CH",
                    "reference_product": "ref-prod",
                }
            ]
        )
        r = IdNameResolver.from_sources(sp_data, ei_catalog)
        key = ExchangeFrameBuilder.flow_id_for(("ei", "abc"))
        info = r.lookup(key)
        assert info["name"] == "canonical ecoinvent name"
        assert info["location"] == "CH"
        assert info["reference_product"] == "ref-prod"

    def test_lookup_unknown_id_returns_placeholder(self):
        r = IdNameResolver.from_sources(
            [],
            pd.DataFrame(
                columns=["database", "code", "name", "unit", "location", "reference_product"]
            ),
        )
        info = r.lookup(12345)
        assert info["code"] == "12345"
        assert "unknown" in info["name"]


# ============================================================================
# DanglingEdgeAuditor


class TestDanglingEdgeAuditor:
    def _resolver(self) -> IdNameResolver:
        sp_data = [
            {"database": "agb", "code": "act-1", "name": "wheat producer", "exchanges": []},
            {"database": "agb", "code": "act-2", "name": "rye producer", "exchanges": []},
            {
                "database": "agb",
                "code": "consumer",
                "name": "consumer",
                "exchanges": [
                    {
                        "type": "technosphere",
                        "name": "orphan product",
                        "input": ("agb", "orphan-prod"),
                    }
                ],
            },
        ]
        return IdNameResolver.from_sources(
            sp_data,
            pd.DataFrame(
                columns=["database", "code", "name", "unit", "location", "reference_product"]
            ),
        )

    def test_attributes_dropped_activity_to_correct_stage(self, tmp_path: Path):
        # act-1 (id _AID) drops between "after_allocator" and "after_dedup".
        initial = _frame(
            [
                (_AID, _PID, 1.0, "production", False),
                (_AID2, _PID, 1.0, "production", False),
            ]
        )
        after_alloc = initial  # no-op
        after_dedup = _frame([(_AID2, _PID, 1.0, "production", False)])
        after_pruner = after_dedup

        auditor = DanglingEdgeAuditor(resolver=self._resolver())
        auditor.snapshot("initial", initial)
        auditor.snapshot("after_allocator", after_alloc)
        auditor.snapshot("after_dedup", after_dedup)
        auditor.capture_consumer_edges(after_dedup)
        auditor.snapshot("after_pruner", after_pruner)

        out_path = tmp_path / "dangling_edges.parquet"
        stats = auditor.write(out_path)

        assert stats["dropped_activities"] == 1
        assert stats["dangling_products"] == 0
        df = pd.read_parquet(out_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["category"] == "dropped_activity"
        assert row["id"] == _AID
        assert row["drop_stage"] == "after_dedup"
        assert row["name"] == "wheat producer"

    def test_consumed_but_never_produced_product_surfaces_with_never_produced_stage(
        self, tmp_path: Path
    ):
        # _PID has a producer; _PID_ORPHAN is consumed by 2 edges but no
        # activity ever produced it. The matcher will have failed to
        # link the consumer, which is exactly the case the auditor must
        # catch — the orphan never made it into the producer universe.
        before_pruner = _frame(
            [
                (_AID, _PID, 1.0, "production", False),
                (_AID, _PID_ORPHAN, 0.5, "technosphere", False),
                (_AID2, _PID_ORPHAN, 0.25, "technosphere", False),
            ]
        )
        after_pruner = _frame([(_AID, _PID, 1.0, "production", False)])

        auditor = DanglingEdgeAuditor(resolver=self._resolver())
        auditor.snapshot("initial", before_pruner)
        auditor.snapshot("after_allocator", before_pruner)
        auditor.snapshot("after_dedup", before_pruner)
        auditor.capture_consumer_edges(before_pruner)
        auditor.snapshot("after_pruner", after_pruner)

        out_path = tmp_path / "dangling_edges.parquet"
        stats = auditor.write(out_path)
        df = pd.read_parquet(out_path)
        assert stats["dangling_products"] == 1
        prod_row = df[df["category"] == "dangling_product"].iloc[0]
        assert prod_row["id"] == _PID_ORPHAN
        assert prod_row["drop_stage"] == "never_produced"
        assert prod_row["consumer_count"] == 2
        assert prod_row["name"] == "orphan product"

    def test_dangling_product_with_consumer_count(self, tmp_path: Path):
        # _PID has a producer initially, but the producer drops at
        # after_dedup. Then _PID is dangling and its 2 consumer edges
        # surface in the parquet.
        initial = _frame(
            [
                (_AID, _PID, 1.0, "production", False),
                (_AID2, _PID_ORPHAN, 1.0, "production", False),  # survives
                (_AID2, _PID, 0.5, "technosphere", False),
                (_AID2, _PID, 0.3, "technosphere", False),
            ]
        )
        after_dedup = _frame(
            [
                (_AID2, _PID_ORPHAN, 1.0, "production", False),
                (_AID2, _PID, 0.5, "technosphere", False),
                (_AID2, _PID, 0.3, "technosphere", False),
            ]
        )
        after_pruner = _frame([(_AID2, _PID_ORPHAN, 1.0, "production", False)])

        auditor = DanglingEdgeAuditor(resolver=self._resolver())
        auditor.snapshot("initial", initial)
        auditor.snapshot("after_allocator", initial)
        auditor.snapshot("after_dedup", after_dedup)
        auditor.capture_consumer_edges(after_dedup)
        auditor.snapshot("after_pruner", after_pruner)

        out_path = tmp_path / "dangling_edges.parquet"
        stats = auditor.write(out_path)
        assert stats["dangling_products"] == 1
        assert stats["dropped_activities"] == 1

        df = pd.read_parquet(out_path)
        prod_row = df[df["category"] == "dangling_product"].iloc[0]
        assert prod_row["id"] == _PID
        assert prod_row["consumer_count"] == 2
        assert prod_row["drop_stage"] == "after_dedup"

    def test_write_handles_no_snapshots_gracefully(self, tmp_path: Path):
        auditor = DanglingEdgeAuditor(resolver=self._resolver())
        stats = auditor.write(tmp_path / "x.parquet")
        assert stats == {"snapshots": 0, "rows": 0}
