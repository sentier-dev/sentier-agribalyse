"""``CfComparisonExporter`` — diff our EF v3.1 CFs against the SimaPro export.

For each EF method we ship (one of the 19 keys in
``MethodCfRegistryBuilder.EF_METHOD_MAP``), this exporter:

1. Loads the built per-method CF parquet from ``registry/method_cfs/<slug>/``.
2. Loads the corresponding SimaPro CFs (parsed from the XLSX export) for
   the same method, including the ``- inorganics``/``- organics``
   sub-methods that SimaPro keeps separate.
3. Joins the two by ``(name, top-level compartment)`` and computes the
   per-flow CF ratio.

Output (one parquet per method under
``to_review/cf_comparison/<slug>.parquet``) carries:

    name           : flow name (canonical, lowercased)
    top            : top-level compartment (air, water, soil, raw, …)
    ours_count     : number of CF rows in our parquet for this (name, top)
    ours_mean      : mean CF amount in our parquet
    sp_count       : same for SimaPro
    sp_mean        : same for SimaPro
    ratio          : ours_mean / sp_mean (NaN if either side missing)
    status         : 'match' (ratio ∈ [0.99, 1.01]),
                     'differ' (both sides have a CF but values diverge),
                     'only_ours', 'only_sp'

A single ``_summary.parquet`` carries one row per method with counts of
each status — useful as a top-level FIX_DATA dashboard.

This is a build-time tool. It is not part of ``dds-link-all``; invoke it
explicitly via the ``dds-mappings-comparison`` family of CLIs (a future
``dds-cf-comparison`` should drive this class).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from core.logging import Logging
from core.parquet_io import ParquetAtomicWriter
from ef.cf_registry import MethodCfRegistryLoader
from ef.simapro_cf_table import SimaProEFCfTable
from scoring.method_slug import MethodSlug


@dataclass(frozen=True)
class CfComparisonExporter:
    """Per-method CF diff: our EF v3.1 CF parquet vs SimaPro EF 3.1 (adapted)."""

    settings: Settings

    OUT_SUBDIR: str = "cf_comparison"
    MATCH_TOL: float = 0.01  # ±1% counts as 'match'

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Top-level driver.

    def export(self) -> Path:
        """Materialise per-method comparison parquets + summary."""
        s = self.settings
        out_dir = s.paths.to_review / self.OUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)

        sp_table = SimaProEFCfTable(
            xlsx_path=s.paths.simapro_ef31_xlsx,
            cache_path=s.paths.simapro_ef31_cache,
        )
        bio_catalog = pd.read_parquet(s.paths.registry_biosphere_catalog)
        ours_per_method = MethodCfRegistryLoader(s.paths.registry_method_cfs_dir).load_all()

        summary_rows: list[dict] = []
        for our_key, ours_df in ours_per_method.items():
            slug = MethodSlug.encode(our_key)
            comparison = self._compare_one_method(
                our_key=our_key,
                ours_df=ours_df,
                sp_table=sp_table,
                bio_catalog=bio_catalog,
            )
            target = out_dir / f"{slug}.parquet"
            ParquetAtomicWriter.write(comparison, target)
            counts = comparison["status"].value_counts().to_dict()
            summary_rows.append(
                {
                    "method_key": "|".join(our_key),
                    "slug": slug,
                    "n_match": int(counts.get("match", 0)),
                    "n_differ": int(counts.get("differ", 0)),
                    "n_only_ours": int(counts.get("only_ours", 0)),
                    "n_only_sp": int(counts.get("only_sp", 0)),
                    "n_total": len(comparison),
                }
            )

        summary = pd.DataFrame(summary_rows).sort_values("slug").reset_index(drop=True)
        ParquetAtomicWriter.write(summary, out_dir / "_summary.parquet")
        self._log.info(
            "exports.cf_comparison.written",
            dir=str(out_dir),
            n_methods=len(summary_rows),
        )
        return out_dir

    # ------------------------------------------------------------------
    # Per-method comparison.

    def _compare_one_method(
        self,
        *,
        our_key: tuple[str, ...],
        ours_df: pd.DataFrame,
        sp_table: SimaProEFCfTable,
        bio_catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        category, indicator = our_key[2], our_key[3]
        ours_annotated = ours_df.merge(
            bio_catalog[["database", "code", "name", "categories"]],
            on=["database", "code"],
            how="left",
        )
        ours_annotated["norm_name"] = ours_annotated["name"].astype(str).str.lower().str.strip()
        ours_annotated["top"] = ours_annotated["categories"].apply(self._top_compartment)
        ours_grouped = (
            ours_annotated.groupby(["norm_name", "top"], dropna=False)["amount"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "ours_count", "mean": "ours_mean"})
        )

        sp_rows = sp_table.for_our_method(category, indicator).copy()
        sp_rows["norm_name"] = sp_rows["name"].astype(str).str.lower().str.strip()
        sp_rows["top"] = sp_rows["compartment"].astype(str).str.lower()
        sp_grouped = (
            sp_rows.groupby(["norm_name", "top"], dropna=False)["cf"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "sp_count", "mean": "sp_mean"})
        )

        merged = ours_grouped.merge(sp_grouped, on=["norm_name", "top"], how="outer")
        merged["ratio"] = merged["ours_mean"] / merged["sp_mean"]
        merged["status"] = merged.apply(self._classify, axis=1)
        return merged.sort_values(["status", "norm_name", "top"]).reset_index(drop=True)

    @staticmethod
    def _top_compartment(cats: object) -> str:
        if cats is None:
            return ""
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if isinstance(cats, (list, tuple)) and cats:
            return str(cats[0]).lower()
        return ""

    def _classify(self, row: pd.Series) -> str:
        ours = row["ours_mean"]
        sp = row["sp_mean"]
        if pd.isna(ours) and pd.isna(sp):
            return "empty"
        if pd.isna(ours):
            return "only_sp"
        if pd.isna(sp):
            return "only_ours"
        if sp == 0 and ours == 0:
            return "match"
        if sp == 0 or ours == 0:
            return "differ"
        ratio = ours / sp
        if abs(ratio - 1.0) <= self.MATCH_TOL:
            return "match"
        return "differ"
