"""``LlmMappingSuggester`` — last-resort tier for residual unlinked AGB flows.

For each unlinked flow, builds a short list of plausible biosphere targets
via :class:`CandidatePool`, then asks the LLM (CLI or API) to pick one. The
LLM is constrained to the candidate list — it can never invent a flow ID.
Decisions are auto-accepted; output goes straight into the existing
``source/agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx`` with
``decision="accept"`` so the next ``dds-build-registry`` picks them up at
tier 10. No sidecar file, no human gate.

The ``client`` is constructor-injected and only needs an ``ask(system, user)``
method, so tests can pass a stub that returns canned text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from config import Settings
from core.logging import Logging
from domain import Bucket
from llm.candidates import CandidatePool
from llm.client import LlmClient

# ---------------------------------------------------------------------------
# Data records.

Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class AgbUnlinkedFlow:
    """One row from ``unlinked/biosphere_unlinked.xlsx`` ready for the LLM."""

    name: str
    unit: str
    top_cat: str
    sub_cat: str

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(c for c in (self.top_cat, self.sub_cat) if c)

    @property
    def bucket(self) -> Bucket:
        return Bucket.from_categories(self.categories)


@dataclass(frozen=True)
class Suggestion:
    """LLM's verdict on one residual flow."""

    source: AgbUnlinkedFlow
    decision: Literal["propose", "reject", "no_candidates"]
    target_db: str = ""
    target_code: str = ""
    target_name: str = ""
    target_categories: tuple[str, ...] = ()
    target_unit: str = ""
    confidence: Confidence | str = ""
    rationale: str = ""


# ---------------------------------------------------------------------------
# Reading the residual xlsx.


@dataclass(frozen=True)
class UnlinkedReader:
    """Read ``unlinked/biosphere_unlinked.xlsx`` and prepare residuals for the LLM.

    Skips:

    * Rows whose name is in ``unmatchable.parquet`` (we already know there's
      no mappable target — running the LLM is wasted compute).
    * Rows whose name appears (with ``decision="accept"``) in the existing
      LLM-reviewed source xlsx — those will be picked up at tier 10 by the
      next registry build, so they're already in flight.

    Crucially, we DO NOT skip names that have a mapping row in
    ``mappings_biosphere.parquet``. If a name is in unlinked.xlsx, its
    mapping (if any) didn't fire — typically because the target name doesn't
    exist in the exchange's compartment. The LLM's job is to find a target
    that DOES exist there.
    """

    settings: Settings

    @property
    def _log(self):
        return Logging.get(__name__)

    def read(
        self,
        registry_unmatchable: pd.DataFrame,
        already_reviewed_xlsx: Path | None = None,
    ) -> list[AgbUnlinkedFlow]:
        path = self.settings.paths.unlinked / "biosphere_unlinked.xlsx"
        if not path.exists():
            self._log.warning("llm.unlinked.missing", path=str(path))
            return []

        df = pd.read_excel(path)
        existing_unmatchable = self._lower_name_set(registry_unmatchable, "source_name")
        already_reviewed = self._already_reviewed_names(already_reviewed_xlsx)

        out: list[AgbUnlinkedFlow] = []
        for _, r in df.iterrows():
            name = str(r.get("source_name", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key in existing_unmatchable or key in already_reviewed:
                continue
            out.append(
                AgbUnlinkedFlow(
                    name=name,
                    unit=str(r.get("source_unit", "") or "").strip(),
                    top_cat=str(r.get("source_top_cat", "") or "").strip(),
                    sub_cat=str(r.get("source_sub_cat", "") or "").strip(),
                )
            )
        self._log.info(
            "llm.unlinked.loaded",
            residuals=len(out),
            input_rows=len(df),
            skipped_unmatchable=len(existing_unmatchable),
            skipped_already_reviewed=len(already_reviewed),
        )
        return out

    @staticmethod
    def _lower_name_set(df: pd.DataFrame, col: str) -> set[str]:
        if df.empty or col not in df.columns:
            return set()
        return set(df[col].dropna().astype(str).str.strip().str.lower())

    @classmethod
    def _already_reviewed_names(cls, path: Path | None) -> set[str]:
        if path is None or not path.exists():
            return set()
        try:
            df = pd.read_excel(path)
        except Exception:
            return set()
        # Skip whatever the existing xlsx already covers — accepted, rejected,
        # or pending. We don't want to spam the LLM with the same residual on
        # every run, and we don't want to overwrite a human's existing verdict.
        return cls._lower_name_set(df, "source_name")


# ---------------------------------------------------------------------------
# The suggester.


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an LCA expert mapping AGRIBALYSE SimaPro biosphere flows to "
    "their canonical equivalent in the Brightway biosphere databases "
    "(biosphere3 / ecoinvent-3.9.1-biosphere / ef). The user gives you one "
    "AGB source flow and a numbered list of candidate flows pre-filtered "
    "to the same compartment.\n\n"
    "STRICT RULES — when in doubt, REJECT:\n"
    "1. PROPOSE only when the candidate is the SAME substance, a recognised "
    "   chemical synonym (e.g. Acetylene ↔ Ethyne), or a tightly defensible "
    "   parent class (e.g. country-suffixed → unsuffixed: 'NO2, FR' → "
    "   'Nitrogen oxides'). High or medium confidence is required.\n"
    "2. REJECT if the candidates are merely topically related (e.g. "
    "   '1-Naphthalene acetamide' → 'Naphthalene' is NOT acceptable: an "
    "   acetamide is not a polycyclic aromatic). REJECT loose proxies, "
    "   unrelated chemistries, generic catch-alls, and 'unspecified' "
    "   targets when the source is specific.\n"
    "3. NEVER propose a target where the unit cannot be made consistent "
    "   with the source unit (mass↔activity, area↔length, etc.).\n"
    "4. NEVER invent flow IDs — only pick from the provided 0-based index.\n"
    "5. NEVER cross compartments (the candidates are already filtered, but "
    "   verify the candidate's compartment matches before proposing).\n\n"
    "We prefer no link to a wrong link. A REJECT is a useful signal — it "
    "tells the maintainers a manual mapping is required, which is far less "
    "harmful than a nonsense match polluting LCIA results.\n\n"
    "Reply with EXACTLY one JSON object on a single line, no prose, no "
    "markdown fences:\n"
    '  {"decision":"propose","candidate_index":N,"confidence":"high|medium","rationale":"..."}\n'
    "or:\n"
    '  {"decision":"reject","confidence":"high|medium|low","rationale":"..."}'
)


@dataclass(frozen=True)
class LlmMappingSuggester:
    """Ask the LLM (CLI or API) to pick a target flow per residual."""

    settings: Settings
    candidate_pool: CandidatePool
    client: LlmClient
    max_candidates: int = 20
    system_prompt: str = SYSTEM_PROMPT

    @property
    def _log(self):
        return Logging.get(__name__)

    # ------------------------------------------------------------------

    def suggest(self, unlinked: list[AgbUnlinkedFlow]) -> list[Suggestion]:
        out: list[Suggestion] = []
        for i, flow in enumerate(unlinked, start=1):
            cands = self.candidate_pool.candidates_for(
                flow.name, flow.bucket, max_n=self.max_candidates
            )
            if not cands:
                verdict = Suggestion(source=flow, decision="no_candidates")
            else:
                verdict = self._ask_llm(flow, cands)
            out.append(verdict)
            self._log.info(
                "llm.suggestion",
                index=i,
                of=len(unlinked),
                source_name=flow.name,
                decision=verdict.decision,
                target=verdict.target_name or None,
            )
        return out

    # ------------------------------------------------------------------

    def _ask_llm(self, flow: AgbUnlinkedFlow, candidates: list) -> Suggestion:
        user = self._build_user_message(flow, candidates)
        try:
            raw = self.client.ask(self.system_prompt, user)
        except Exception as exc:  # client failures shouldn't crash the run
            self._log.warning("llm.client.failed", source_name=flow.name, error=str(exc)[:200])
            return Suggestion(source=flow, decision="reject", rationale=f"client error: {exc}")
        return self._parse_response(raw, flow, candidates)

    @staticmethod
    def _build_user_message(flow: AgbUnlinkedFlow, candidates: list) -> str:
        cand_lines = [
            f"  [{i}] db={c.db} code={c.code} name={c.name!r} unit={c.unit!r} "
            f"compartment={c.bucket.value}"
            for i, c in enumerate(candidates)
        ]
        return (
            "AGB source flow:\n"
            f"  name: {flow.name!r}\n"
            f"  compartment: {flow.top_cat}/{flow.sub_cat or '-'} (bucket={flow.bucket.value})\n"
            f"  unit: {flow.unit!r}\n\n"
            "Candidates (same compartment, ranked by name overlap):\n"
            + "\n".join(cand_lines)
            + f"\n\nPick ONE candidate index (0..{len(candidates) - 1}), or reject all."
        )

    def _parse_response(self, raw: str, flow: AgbUnlinkedFlow, candidates: list) -> Suggestion:
        payload = self._extract_json(raw)
        if payload is None:
            return Suggestion(
                source=flow,
                decision="reject",
                rationale="LLM response did not contain a parseable JSON object",
            )
        decision = payload.get("decision")
        confidence = str(payload.get("confidence", ""))
        rationale = str(payload.get("rationale", ""))
        if decision == "reject":
            return Suggestion(
                source=flow,
                decision="reject",
                confidence=confidence,
                rationale=rationale,
            )
        if decision != "propose":
            return Suggestion(
                source=flow,
                decision="reject",
                confidence=confidence,
                rationale=f"unknown decision={decision!r}; {rationale}",
            )
        idx = payload.get("candidate_index")
        if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
            return Suggestion(
                source=flow,
                decision="reject",
                confidence=confidence,
                rationale=f"invalid candidate_index={idx}; {rationale}",
            )
        chosen = candidates[idx]
        return Suggestion(
            source=flow,
            decision="propose",
            target_db=chosen.db,
            target_code=chosen.code,
            target_name=chosen.name,
            target_categories=(chosen.bucket.value,),
            target_unit=chosen.unit,
            confidence=confidence,
            rationale=rationale,
        )

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Find the first JSON object in ``raw`` and parse it. ``None`` on failure."""
        if not raw:
            return None
        for match in _JSON_OBJECT_RE.finditer(raw):
            chunk = match.group(0)
            try:
                value = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None
