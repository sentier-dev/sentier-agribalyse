"""``MappingsComparisonExporter`` — write ``to_review/mappings_comparison.xlsx``.

Single-sheet review workbook splitting every mapping into two color-coded
bands so reviewers see, in one view, what is left to triage:

* RED ("in review") — new mappings produced by sentier_agribalyse for source
  flows whose name does not appear in
  ``placeholder_flow_classification.xlsx`` (ei + ef sheets). Includes
  Neither-sheet rescues and entirely novel flows.
* BLUE ("Already reviewed by Xiaojin and ADEME") — verbatim rows from the
  placeholder ei + ef sheets.

Columns are the unified placeholder schema with one extra ``status`` column.
RED rows render first, BLUE rows below, with a yellow banner separating them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from config import Settings
from core import ParquetCache
from core.logging import Logging
from matching.bio_catalog import BiosphereCatalog
from readers import XlsxReader
from transforms.strategies.internal import ActivityHash


@dataclass(frozen=True)
class _SourceRecord:
    """One AGB biosphere source flow with its dominant linked target."""

    name: str
    cats: tuple[str, ...]
    unit: str
    cas: str
    target: tuple[str, str] | None  # (target_db, target_code), or None if unlinked
    n: int


@dataclass(frozen=True)
class MappingsComparisonExporter:
    """Generate ``mappings_comparison.xlsx`` from in-memory ``sp.data``."""

    settings: Settings
    sp_data: list[dict] | None = None

    COLUMNS = (
        "name",
        "categories",
        "unit",
        "code",
        "cas",
        "target_database",
        "target_name",
        "target_categories",
        "target_cas",
        "match_type",
        "category_proxy_match",
        "status",
    )
    COLUMN_WIDTHS = (46, 42, 12, 34, 14, 32, 46, 42, 14, 22, 22, 38)

    BANNER_FILL = PatternFill(start_color="FFD966", end_color="FFD966", fill_type="solid")
    RED_FILL = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid")
    BLUE_FILL = PatternFill(start_color="CFE2F3", end_color="CFE2F3", fill_type="solid")

    STATUS_REVIEWED = "Already reviewed by Xiaojin and ADEME"
    STATUS_IN_REVIEW = "in review"
    REVIEW_SHEET_NAME = "Mappings Review"

    @property
    def _log(self):
        return Logging.get(__name__)

    @property
    def _ei_bio_db(self) -> str:
        return f"ecoinvent-{self.settings.resolved_ecoinvent_version}-biosphere"

    @property
    def _ef_db(self) -> str:
        return self.settings.ef_db_name

    # ------------------------------------------------------------------
    # Public entrypoint.

    def export(self) -> Path:
        s = self.settings
        out = s.paths.to_review / "mappings_comparison.xlsx"
        out.parent.mkdir(parents=True, exist_ok=True)

        cache = ParquetCache(cache_dir=s.paths.cache)
        xlsx = XlsxReader(cache=cache)
        placeholder = self._load_placeholder_sheets(xlsx, s.paths.placeholder_xlsx)

        target_lookups = self._build_target_lookups()
        src_idx = self._build_source_index()
        red_rows, blue_rows = self._build_consolidated_rows(src_idx, target_lookups, placeholder)

        wb = Workbook()
        wb.remove(wb.active)
        self._write_summary(wb, len(red_rows), len(blue_rows))
        self._write_review_sheet(wb, red_rows, blue_rows)
        wb.save(out)
        self._log.info("mappings_comparison.written", path=str(out))
        return out

    # ------------------------------------------------------------------
    # Placeholder workbook.

    @staticmethod
    def _load_placeholder_sheets(xlsx: XlsxReader, path) -> dict[str, pd.DataFrame]:
        sheets = (
            "match with ecoinvent v3.9.1",
            "match with EF v3.1",
        )
        return {s: xlsx.read(path, sheet_name=s) for s in sheets}

    # ------------------------------------------------------------------
    # Target metadata: (db, code) → flow meta for every biosphere target DB.

    def _build_target_lookups(self) -> dict[str, dict[tuple[str, str], dict]]:
        """Read biosphere flow metadata from the parquet catalog."""
        catalog_path = self.settings.paths.registry_biosphere_catalog
        if not catalog_path.exists():
            return {}
        catalog = BiosphereCatalog.load(
            catalog_path,
            db_names=(self._ei_bio_db, self._ef_db, "biosphere3"),
        )
        out: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
        for flow in catalog.flows:
            out[flow.db][(flow.db, flow.code)] = {
                "name": flow.name,
                "categories": tuple(flow.categories) if flow.categories else (),
                "cas": self._norm_cas(flow.cas),
                "unit": flow.unit,
            }
        return dict(out)

    # ------------------------------------------------------------------
    # Source index: aggregate AGB biosphere exchanges.

    def _build_source_index(self) -> dict[tuple[str, tuple[str, ...], str], _SourceRecord]:
        if self.sp_data is None:
            raise RuntimeError(
                "MappingsComparisonExporter requires in-memory sp.data after F5; "
                "the bw2data db-walk path was removed."
            )
        return self._index_from_sp_data(self.sp_data)

    def _index_from_sp_data(
        self, sp_data: list[dict]
    ) -> dict[tuple[str, tuple[str, ...], str], _SourceRecord]:
        accum: dict[tuple[str, tuple[str, ...], str], dict] = {}
        for proc in sp_data:
            for exc in proc.get("exchanges", []):
                if exc.get("type") != "biosphere":
                    continue
                name = str(exc.get("name") or "").strip()
                if not name:
                    continue
                cats = self._coerce_cats(exc.get("categories") or exc.get("context"))
                unit = str(exc.get("unit") or "").strip()
                cas = self._norm_cas(exc.get("cas_number") or exc.get("CAS number"))
                inp = exc.get("input")
                target = inp if isinstance(inp, tuple) and len(inp) == 2 else None
                self._accumulate(accum, name, cats, unit, cas, target)
        return self._finalise(accum)

    @staticmethod
    def _accumulate(
        accum: dict,
        name: str,
        cats: tuple[str, ...],
        unit: str,
        cas: str,
        target: tuple[str, str] | None,
    ) -> None:
        key = (name.lower(), cats, unit.lower())
        rec = accum.get(key)
        if rec is None:
            rec = {
                "name": name,
                "cats": cats,
                "unit": unit,
                "cas": cas,
                "targets": Counter(),
                "n": 0,
            }
            accum[key] = rec
        rec["n"] += 1
        if target:
            rec["targets"][target] += 1
        if cas and not rec["cas"]:
            rec["cas"] = cas

    @staticmethod
    def _finalise(
        accum: dict[tuple[str, tuple[str, ...], str], dict],
    ) -> dict[tuple[str, tuple[str, ...], str], _SourceRecord]:
        out: dict[tuple[str, tuple[str, ...], str], _SourceRecord] = {}
        for key, rec in accum.items():
            if rec["targets"]:
                # Deterministic dominant target: highest count, ties broken by
                # lex (db, code) so re-runs are byte-identical.
                target = sorted(
                    rec["targets"].items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
                )[0][0]
            else:
                target = None
            out[key] = _SourceRecord(
                name=rec["name"],
                cats=rec["cats"],
                unit=rec["unit"],
                cas=rec["cas"],
                target=target,
                n=rec["n"],
            )
        return out

    # ------------------------------------------------------------------
    # Consolidated row building.

    def _build_consolidated_rows(
        self,
        src_idx: dict[tuple[str, tuple[str, ...], str], _SourceRecord],
        target_lookups: dict[str, dict[tuple[str, str], dict]],
        placeholder: dict[str, pd.DataFrame],
    ) -> tuple[list[dict], list[dict]]:
        """Return ``(red_rows, blue_rows)``.

        BLUE = placeholder ei + ef rows verbatim, normalised to the unified
        column schema. RED = AGB-linked unique source flows whose name is
        absent from placeholder ei AND ef (Neither-rescues + truly novel).
        """
        ph_ei = placeholder["match with ecoinvent v3.9.1"]
        ph_ef = placeholder["match with EF v3.1"]
        ph_names = self._placeholder_name_keys(ph_ei) | self._placeholder_name_keys(ph_ef)

        blue_rows = [self._blue_row_ei(r) for _, r in ph_ei.iterrows()]
        blue_rows.extend(self._blue_row_ef(r) for _, r in ph_ef.iterrows())
        blue_rows.sort(
            key=lambda r: (str(r.get("name") or "").lower(), str(r.get("categories") or ""))
        )

        red_rows: list[dict] = []
        for (name_lc, _cats, _unit), rec in src_idx.items():
            if name_lc in ph_names:
                continue
            if rec.target is None:
                continue
            red_rows.append(self._red_row(rec, target_lookups))
        red_rows.sort(key=lambda r: (str(r["name"]).lower(), str(r["categories"])))

        return red_rows, blue_rows

    def _blue_row_ei(self, r: pd.Series) -> dict:
        return {
            "name": r.get("name"),
            "categories": r.get("categories"),
            "unit": r.get("unit"),
            "code": r.get("code"),
            "cas": r.get("cas"),
            "target_database": self._ei_bio_db,
            "target_name": r.get("matched_ecoinvent_name"),
            "target_categories": r.get("matched_ecoinvent_categories"),
            "target_cas": r.get("matched_ecoinvent_cas"),
            "match_type": r.get("ecoinvent_match_type"),
            "category_proxy_match": r.get("category_proxy_match"),
            "status": self.STATUS_REVIEWED,
        }

    def _blue_row_ef(self, r: pd.Series) -> dict:
        return {
            "name": r.get("name"),
            "categories": r.get("categories"),
            "unit": r.get("unit"),
            "code": r.get("code"),
            "cas": r.get("cas"),
            "target_database": self._ef_db,
            "target_name": r.get("matched_ef_name"),
            "target_categories": r.get("matched_ef_categories"),
            "target_cas": r.get("matched_ef_cas"),
            "match_type": r.get("ef_match_type"),
            "category_proxy_match": r.get("category_proxy_match"),
            "status": self.STATUS_REVIEWED,
        }

    def _red_row(
        self,
        rec: _SourceRecord,
        target_lookups: dict[str, dict[tuple[str, str], dict]],
    ) -> dict:
        flow = target_lookups.get(rec.target[0], {}).get(rec.target) or {}
        proxy = self._is_proxy_match(rec.cats, flow.get("categories", ()))
        code = ActivityHash.of({"name": rec.name, "categories": rec.cats, "unit": rec.unit})
        return {
            "name": rec.name,
            "categories": str(rec.cats) if rec.cats else "()",
            "unit": rec.unit,
            "code": code,
            "cas": rec.cas,
            "target_database": rec.target[0],
            "target_name": flow.get("name", ""),
            "target_categories": str(flow.get("categories") or ()),
            "target_cas": flow.get("cas", ""),
            "match_type": "novel",
            "category_proxy_match": proxy,
            "status": self.STATUS_IN_REVIEW,
        }

    @staticmethod
    def _placeholder_name_keys(df: pd.DataFrame) -> set[str]:
        if "name" not in df.columns:
            return set()
        return set(df["name"].dropna().astype(str).str.strip().str.lower())

    # ------------------------------------------------------------------
    # Workbook writing.

    def _write_summary(self, wb: Workbook, n_red: int, n_blue: int) -> None:
        ws = wb.create_sheet("Summary")
        ws.append(["Status", "Rows"])
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = self.BANNER_FILL
        ws.append([f"{self.STATUS_IN_REVIEW} (RED — new)", n_red])
        ws.append([f"{self.STATUS_REVIEWED} (BLUE — placeholder)", n_blue])
        ws.append(["TOTAL", n_red + n_blue])
        ws.column_dimensions["A"].width = 56
        ws.column_dimensions["B"].width = 10
        ws.freeze_panes = "A2"

    def _write_review_sheet(
        self, wb: Workbook, red_rows: list[dict], blue_rows: list[dict]
    ) -> None:
        ws = wb.create_sheet(self.REVIEW_SHEET_NAME)
        ws.append(list(self.COLUMNS))
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True)

        self._append_banner(
            ws,
            f"{self.STATUS_IN_REVIEW} — RED (new mappings, not in placeholder)",
            len(red_rows),
            len(self.COLUMNS),
        )
        self._append_rows(ws, red_rows, self.RED_FILL)

        ws.append([""] * len(self.COLUMNS))

        self._append_banner(
            ws,
            f"{self.STATUS_REVIEWED} — BLUE (placeholder ei + ef)",
            len(blue_rows),
            len(self.COLUMNS),
        )
        self._append_rows(ws, blue_rows, self.BLUE_FILL)

        ws.freeze_panes = "A2"
        self._set_widths(ws, self.COLUMN_WIDTHS)

    def _append_banner(self, ws, label: str, n: int, ncols: int) -> None:
        ws.append([f"{label} ({n} rows)"] + [""] * (ncols - 1))
        r = ws.max_row
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ncols)
        cell = ws.cell(row=r, column=1)
        cell.font = Font(bold=True)
        cell.fill = self.BANNER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[r].height = 22

    def _append_rows(
        self,
        ws,
        rows: list[dict],
        fill: PatternFill,
    ) -> None:
        for row in rows:
            ws.append([_render(row.get(c, "")) for c in self.COLUMNS])
            for cell in ws[ws.max_row]:
                cell.fill = fill

    @staticmethod
    def _set_widths(ws, widths: tuple[int, ...]) -> None:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w

    # ------------------------------------------------------------------
    # Pure helpers (static — no per-instance state).

    @staticmethod
    def _coerce_cats(value) -> tuple[str, ...]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ()
        if isinstance(value, (tuple, list)):
            return tuple(str(x) for x in value)
        return ()

    @staticmethod
    def _norm_cas(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        s = str(value).strip()
        if not s:
            return ""
        digits = s.replace("-", "").lstrip("0") or "0"
        return digits

    @staticmethod
    def _is_proxy_match(source_cats: tuple[str, ...], target_cats: tuple[str, ...]) -> bool:
        if not source_cats or not target_cats:
            return False
        return (
            len(target_cats) <= len(source_cats)
            and source_cats[0].lower() == target_cats[0].lower()
        )


def _render(value) -> str:
    """Workbook-cell-safe coercion: NaN/None → '', tuples/lists stringified."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (tuple, list)):
        return str(value)
    return str(value)
