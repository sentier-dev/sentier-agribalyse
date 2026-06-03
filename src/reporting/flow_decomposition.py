"""``FlowDecompositionEmitter`` — per-product, per-method flow JSONs.

For every mapped product in the backtest scores frame, decomposes the
score under each registered method into per-biosphere-flow contributions
(pre-CF ``inventory`` × ``cf`` = post-CF ``contribution``), truncates to
``top_n`` rows by ``|contribution|``, joins the biosphere catalog for
per-flow units, and writes one JSON per product to ``out_dir``. The
dashboard fetches these lazily when a user clicks a (product, method)
cell.

By design the emitter is a thin orchestrator on top of
:class:`scoring.decomposer.ScoreDecomposer`: the decomposer carries the
LU factorisation cache so iterating products × 19 methods costs roughly
one solve per product. The biosphere catalog is consumed only for the
``unit`` column.

OOP-only per ``CLAUDE.md``: frozen dataclass, dependencies injected,
no module-level helpers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.logging import Logging
from reporting.simapro_cf_lookup import SimaProCfLookup
from scoring.decomposer import ScoreDecomposer
from scoring.exchange_frame_builder import ExchangeFrameBuilder


@dataclass(frozen=True)
class FlowDecompositionEmitter:
    decomposer: ScoreDecomposer
    biosphere_catalog: pd.DataFrame
    method_short_to_full: Mapping[str, tuple[str, ...]]
    out_dir: Path
    top_n: int = 25
    simapro_cf: SimaProCfLookup | None = None

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------

    def write(self, scores_df: pd.DataFrame) -> Path:
        """Emit one JSON per mapped product in ``scores_df``.

        Expected columns: ``Code AGB`` (product code, str/int),
        ``Nom du Produit`` (display name), ``mapped`` (bool filter).
        Rows with ``mapped=False`` are skipped silently.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        unit_lookup = self._build_unit_lookup()
        code_lookup = self._build_code_lookup()

        mapped = scores_df.loc[scores_df["mapped"].astype(bool)]
        n_written = 0
        n_skipped = 0
        for _, row in mapped.iterrows():
            code = str(row["Code AGB"])
            name = str(row.get("Nom du Produit", code))
            safe_name = Path(code).name
            if not safe_name or safe_name in (".", ".."):
                self._log.warning("flow_decomp.skip", code=code, reason="invalid_code")
                n_skipped += 1
                continue
            db_key = row.get("product_db")
            code_key = row.get("product_code")
            if db_key is None or pd.isna(db_key) or code_key is None or pd.isna(code_key):
                self._log.warning("flow_decomp.skip", code=code, reason="missing_product_key")
                n_skipped += 1
                continue
            try:
                payload = self._build_product_payload(
                    code, name, str(db_key), str(code_key), unit_lookup, code_lookup
                )
            except ValueError as exc:
                self._log.warning("flow_decomp.skip", code=code, reason=str(exc))
                n_skipped += 1
                continue
            (self.out_dir / f"{safe_name}.json").write_text(json.dumps(payload))
            n_written += 1
            if n_written % 100 == 0:
                self._log.info("flow_decomp.progress", written=n_written)

        self._log.info(
            "flow_decomp.done",
            written=n_written,
            skipped=n_skipped,
            out_dir=str(self.out_dir),
        )
        return self.out_dir

    # ------------------------------------------------------------------

    def _build_product_payload(
        self,
        code: str,
        name: str,
        db_key: str,
        code_key: str,
        unit_lookup: dict[int, str],
        code_lookup: dict[int, str],
    ) -> dict:
        wanted = [
            full
            for short, full in self.method_short_to_full.items()
            if full in self.decomposer.package.methods
        ]
        for short, full in self.method_short_to_full.items():
            if full not in self.decomposer.package.methods:
                self._log.warning("flow_decomp.method_missing", short=short, full=full)
        results = self.decomposer.decompose_methods(
            (db_key, code_key), wanted, top_n=None, flow_only=True
        )
        methods_block: dict[str, dict] = {}
        for short, full in self.method_short_to_full.items():
            decomp = results.get(full)
            if decomp is None:
                continue
            # The method tuple is (database, ef_version, category, indicator);
            # the last two key the per-flow SimaPro CF table.
            category, indicator = (full[2], full[3]) if len(full) >= 4 else ("", "")
            flows = decomp.flow_contributions
            total_abs = float(np.abs(flows["contribution"]).sum()) if not flows.empty else 0.0
            top = flows.head(self.top_n)
            flow_rows: list[dict] = []
            for r in top.itertuples(index=False):
                flow_id = int(r.flow_id)
                sp_entry = (
                    self.simapro_cf.get(code_lookup.get(flow_id, ""), category, indicator)
                    if self.simapro_cf is not None
                    else None
                )
                flow_rows.append(
                    {
                        "flow_name": str(r.flow_name),
                        "compartment": str(r.compartment),
                        "sub_compartment": str(r.sub_compartment),
                        "unit": unit_lookup.get(flow_id, ""),
                        "inventory": float(r.inventory_amount),
                        "cf": float(r.cf),
                        "sp_cf": (sp_entry.sp_cf if sp_entry is not None else None),
                        "sp_match_provenance": (
                            sp_entry.provenance if sp_entry is not None else None
                        ),
                        "contribution": float(r.contribution),
                        "share": (float(abs(r.contribution) / total_abs) if total_abs else 0.0),
                    }
                )
            methods_block[short] = {
                "score": decomp.score,
                "global_score": decomp.global_score,
                "correction_score": decomp.correction_score,
                "total_abs_contribution": total_abs,
                "flows": flow_rows,
            }
        return {"code": code, "name": name, "methods": methods_block}

    def _build_unit_lookup(self) -> dict[int, str]:
        lookup: dict[int, str] = {}
        if self.biosphere_catalog.empty:
            return lookup
        for row in self.biosphere_catalog.itertuples(index=False):
            db = str(getattr(row, "database", "") or "")
            code = str(getattr(row, "code", "") or "")
            if not db or not code:
                continue
            unit = str(getattr(row, "unit", "") or "")
            lookup[ExchangeFrameBuilder.flow_id_for((db, code))] = unit
        return lookup

    def _build_code_lookup(self) -> dict[int, str]:
        """``flow_id -> biosphere code`` — the reverse of the integer hash,
        used to join each characterised flow to its SimaPro CF."""
        lookup: dict[int, str] = {}
        if self.biosphere_catalog.empty:
            return lookup
        for row in self.biosphere_catalog.itertuples(index=False):
            db = str(getattr(row, "database", "") or "")
            code = str(getattr(row, "code", "") or "")
            if not db or not code:
                continue
            lookup[ExchangeFrameBuilder.flow_id_for((db, code))] = code
        return lookup
