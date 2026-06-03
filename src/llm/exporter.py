"""``LlmReviewedXlsxAppender`` — merge fresh LLM proposals into the existing source xlsx.

There is no human-in-the-loop. The LLM's proposals are written directly to
``source/agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx`` with
``decision="accept"`` so the next ``dds-build-registry`` run picks them up
at tier 10. Existing rows in the file are preserved (we don't overwrite a
human-flipped ``"reject"``); only source names not yet present are added.

Schema follows what :class:`LlmReviewedSource` reads — ``source_name`` /
``source_unit`` / ``source_top_cat`` / ``source_sub_cat`` / ``target_name``
/ ``target_top_cat`` / ``target_sub_cat`` / ``decision`` / ``reviewer_note``
plus ``multiplier``. The reader ignores extra columns, so we also tack on
``llm_confidence`` and ``llm_rationale`` for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.logging import Logging
from llm.suggester import Suggestion


@dataclass(frozen=True)
class LlmReviewedXlsxAppender:
    """Merge proposed mappings into the canonical LLM-reviewed source xlsx."""

    output_path: Path

    COLUMNS = (
        "source_name",
        "source_unit",
        "source_top_cat",
        "source_sub_cat",
        "target_name",
        "target_top_cat",
        "target_sub_cat",
        "target_db",
        "target_code",
        "target_unit",
        "multiplier",
        "decision",
        "reviewer_note",
        "llm_confidence",
        "llm_rationale",
    )

    @property
    def _log(self):
        return Logging.get(__name__)

    # ------------------------------------------------------------------

    def append(self, suggestions: list[Suggestion]) -> dict[str, int]:
        proposed = [s for s in suggestions if s.decision == "propose"]
        existing = self._load_existing()
        existing_names = self._names(existing)

        new_rows: list[dict] = []
        for s in proposed:
            if s.source.name.strip().lower() in existing_names:
                continue
            new_rows.append(self._row(s))

        if not new_rows and existing is None:
            # Nothing to write at all (no existing file, no proposals).
            self._log.info(
                "llm.appender.skipped",
                path=str(self.output_path),
                reason="no_existing_file_and_no_proposals",
            )
            return {"appended": 0, "kept_existing": 0, "rejected_or_skipped": len(suggestions)}

        new_df = pd.DataFrame(new_rows, columns=list(self.COLUMNS))
        merged = new_df if existing is None else pd.concat([existing, new_df], ignore_index=True)
        # Ensure all canonical columns exist even if the existing file was sparse.
        for col in self.COLUMNS:
            if col not in merged.columns:
                merged[col] = ""
        merged = merged[list(self.COLUMNS)]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_excel(self.output_path, index=False)
        stats = {
            "appended": len(new_rows),
            "kept_existing": 0 if existing is None else len(existing),
            "rejected_or_skipped": len(suggestions)
            - len(proposed)
            + sum(1 for s in proposed if s.source.name.strip().lower() in existing_names),
        }
        self._log.info("llm.appender.written", path=str(self.output_path), **stats)
        return stats

    # ------------------------------------------------------------------

    def _load_existing(self) -> pd.DataFrame | None:
        if not self.output_path.exists():
            return None
        try:
            return pd.read_excel(self.output_path)
        except Exception as exc:
            self._log.warning(
                "llm.appender.load_failed",
                path=str(self.output_path),
                error=str(exc)[:200],
            )
            return None

    @staticmethod
    def _names(df: pd.DataFrame | None) -> set[str]:
        if df is None or df.empty or "source_name" not in df.columns:
            return set()
        return set(df["source_name"].dropna().astype(str).str.strip().str.lower())

    @staticmethod
    def _row(s: Suggestion) -> dict:
        target_top = s.target_categories[0] if s.target_categories else ""
        target_sub = s.target_categories[1] if len(s.target_categories) >= 2 else ""
        return {
            "source_name": s.source.name,
            "source_unit": s.source.unit,
            "source_top_cat": s.source.top_cat,
            "source_sub_cat": s.source.sub_cat,
            "target_name": s.target_name,
            "target_top_cat": target_top,
            "target_sub_cat": target_sub,
            "target_db": s.target_db,
            "target_code": s.target_code,
            "target_unit": s.target_unit,
            "multiplier": "",
            "decision": "accept",  # auto-accepted; reviewer can flip post-hoc
            "reviewer_note": "auto-accepted by dds-llm-suggest-mappings",
            "llm_confidence": s.confidence,
            "llm_rationale": s.rationale,
        }
