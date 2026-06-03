"""``EfFlowsRegistryBuilder`` — emit ``registry/ef_flows.parquet``.

Replaces ``EfDatabase.install()``. Reads the EF flow universe from
``registry/target_index_ef.parquet`` and filters down to the UUIDs
referenced by:

(a) registry biosphere mappings whose ``target_db == ef_db_name``;
(b) EF v3.1 method definitions snapshotted at ``source/ef-v31-methods.json``
    (REFACTOR_FINAL F6 — bw2data is no longer queried at build time).

Determinism: rows sorted by ``code`` before write. Atomic write via
``ParquetAtomicWriter``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from core.logging import Logging
from core.parquet_io import ParquetAtomicWriter
from registry import MappingRegistry


@dataclass(frozen=True)
class EfFlowsRegistryBuilder:
    """Materialise the hygiene-filtered EF flow universe to a parquet."""

    settings: Settings
    registry: MappingRegistry

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Public entry point.

    def build(self) -> Path:
        s = self.settings
        s.paths.ensure_runtime_dirs()

        target_index = self.registry.target_index_ef
        if target_index.empty:
            self._log.warning("ef.flows.registry.no_target_index")
            df = self._frame([])
            ParquetAtomicWriter.write(df, s.paths.registry_ef_flows)
            return s.paths.registry_ef_flows

        wanted = self._needed_uuids()
        kept = target_index[target_index["code"].isin(wanted)]
        rows = [self._row_for(r) for r in kept.itertuples(index=False)]
        df = self._frame(rows)
        target = s.paths.registry_ef_flows
        ParquetAtomicWriter.write(df, target)
        self._log.info(
            "ef.flows.registry.built",
            path=str(target),
            n_rows=len(df),
            n_referenced=len(wanted),
        )
        return target

    # ------------------------------------------------------------------
    # Hygiene filter.

    def _needed_uuids(self) -> set[str]:
        """All EF UUIDs referenced by registry rows or registered EF methods."""
        s = self.settings
        wanted: set[str] = set()

        bio = self.registry.mappings_biosphere
        if not bio.empty:
            mask = (bio["target_db"] == s.ef_db_name) & (bio["target_code"].fillna("") != "")
            wanted.update(bio.loc[mask, "target_code"].tolist())

        # Anything pinned in the EF v3.1 method snapshot. The snapshot is
        # produced once via REFACTOR_FINAL F6; absence is non-fatal so the
        # builder can run before the bootstrap step.
        snap_path = s.paths.ef_methods_snapshot_json
        if snap_path.exists():
            payload = json.loads(snap_path.read_text())
            ef_db = payload.get("ef_db_name", s.ef_db_name)
            for entry in payload.get("methods", {}).values():
                for cf in entry.get("inherited_cfs", []):
                    if cf.get("db") == ef_db and isinstance(cf.get("code"), str):
                        wanted.add(cf["code"])
        else:
            self._log.warning(
                "ef.flows.registry.snapshot_missing",
                path=str(snap_path),
            )
        return wanted

    # ------------------------------------------------------------------
    # Helpers.

    @staticmethod
    def _row_for(row: Any) -> dict:
        cats = list(row.categories) if row.categories is not None else []
        cas = row.cas if isinstance(row.cas, str) and row.cas else None
        return {
            "code": str(row.code),
            "name": str(row.name),
            "categories": [str(c) for c in cats],
            "unit": str(row.unit) if row.unit else "kilogram",
            "cas": cas,
        }

    @staticmethod
    def _frame(rows: list[dict]) -> pd.DataFrame:
        cols = ["code", "name", "categories", "unit", "cas"]
        df = pd.DataFrame(rows, columns=cols)
        df = df.astype({"code": "string", "name": "string", "unit": "string"})
        df = df.sort_values(by="code", kind="mergesort").reset_index(drop=True)
        return df
