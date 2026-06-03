"""``BiosphereRegistryBuilder`` — assemble ``registry/biosphere_catalog.parquet``.

After REFACTOR_FINAL phase F6 the builder reads exclusively from
checked-in source files:

* ``source/biosphere3-flows.json`` — the bw2io-shipped biosphere3
  elementary-flow universe, snapshotted once (4709 flows for 3.9).
* ``source/ecoinvent-3.9.1-biosphere-flows.json`` — the ecoinvent
  biosphere database, snapshotted once after a fresh
  ``bw2io.import_ecoinvent_release`` (~4700 flows for 3.9.1).
* ``registry/ef_flows.parquet`` — the EF v3.1 universe materialised by
  ``EfFlowsRegistryBuilder`` (already SQLite-free since L2).

No bw2data, no SQLite, no `_vacuum`. Determinism + atomic write are
preserved (sorted by ``(database, code)``, ``ParquetAtomicWriter``).
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


@dataclass(frozen=True)
class BiosphereRegistryBuilder:
    """Materialise ``biosphere_catalog.parquet`` from source-file snapshots."""

    settings: Settings

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Public entry point.

    def build(self) -> Path:
        """Read every snapshot + the EF parquet, write the catalog parquet."""
        s = self.settings
        s.paths.ensure_runtime_dirs()

        rows: list[dict] = []

        # biosphere3 (bw2io snapshot)
        rows.extend(
            self._rows_from_json(
                s.paths.biosphere3_flows_json,
                db_name="biosphere3",
            )
        )

        # ecoinvent biosphere — only loaded when the snapshot exists. Skipping
        # is fine: matchers ignore the DB name when it's empty.
        if s.paths.ecoinvent_biosphere_flows_json.exists():
            rows.extend(
                self._rows_from_json(
                    s.paths.ecoinvent_biosphere_flows_json,
                    db_name=s.biosphere_db_name,
                )
            )
        else:
            self._log.warning(
                "biosphere.catalog.skip_ecoinvent",
                reason=f"{s.paths.ecoinvent_biosphere_flows_json} missing",
            )

        # EF half — already a parquet.
        rows.extend(self._ef_rows_from_parquet())

        df = self._frame(rows)
        target = s.paths.registry_biosphere_catalog
        ParquetAtomicWriter.write(df, target)
        self._log.info(
            "biosphere.catalog.built",
            path=str(target),
            n_rows=len(df),
            sources=sorted(set(df["database"].tolist())),
        )
        return target

    # ------------------------------------------------------------------
    # JSON snapshots.

    @staticmethod
    def _rows_from_json(path: Path, *, db_name: str) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(
                f"biosphere snapshot missing: {path}. "
                f"Snapshot via REFACTOR_FINAL phase F6 (one-time bootstrap)."
            )
        payload = json.loads(path.read_text())
        flows = payload.get("flows", payload) if isinstance(payload, dict) else payload
        out: list[dict] = []
        for f in flows:
            cas = f.get("cas") or None
            out.append(
                {
                    "database": db_name,
                    "code": str(f.get("code", "") or ""),
                    "name": str(f.get("name", "") or ""),
                    "categories": [str(c) for c in (f.get("categories") or [])],
                    "unit": str(f.get("unit", "") or ""),
                    "cas": str(cas) if cas else None,
                    "synonyms": [str(s) for s in (f.get("synonyms") or [])],
                }
            )
        return out

    # ------------------------------------------------------------------
    # EF parquet ingestion (unchanged from L2).

    def _ef_rows_from_parquet(self) -> list[dict]:
        s = self.settings
        ef_path = s.paths.registry_ef_flows
        if not ef_path.exists():
            raise FileNotFoundError(
                f"EF flows parquet not found: {ef_path}. "
                f"Run `dds-build-ef-flows-registry` (or `dds-build-data-registries`) "
                f"before building the biosphere catalog."
            )
        ef_df = pd.read_parquet(ef_path)
        out: list[dict] = []
        for r in ef_df.itertuples(index=False):
            cats = list(r.categories) if r.categories is not None else []
            cas = r.cas if isinstance(r.cas, str) and r.cas else None
            out.append(
                {
                    "database": s.ef_db_name,
                    "code": str(r.code) if r.code is not None else "",
                    "name": str(r.name) if r.name is not None else "",
                    "categories": [str(c) for c in cats],
                    "unit": str(r.unit) if r.unit else "",
                    "cas": str(cas) if cas else None,
                    "synonyms": [],
                }
            )
        return out

    # ------------------------------------------------------------------
    # Frame helpers — unchanged.

    @staticmethod
    def _frame(rows: list[dict]) -> pd.DataFrame:
        cols = ["database", "code", "name", "categories", "unit", "cas", "synonyms"]
        df = pd.DataFrame(rows, columns=cols)
        df = df.astype({"database": "string", "code": "string", "name": "string", "unit": "string"})
        df = df.sort_values(by=["database", "code"], kind="mergesort").reset_index(drop=True)
        return df
