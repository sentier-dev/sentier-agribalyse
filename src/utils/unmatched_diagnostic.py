"""Replay ``BiosphereMatcher`` candidate selection on residual unlinked flows.

When ``unlinked/biosphere_unlinked.xlsx`` shows a flow that is "stuck",
this module walks the same registry/tier/catalog logic the matcher uses
and explains, per candidate, why it did or didn't link. Designed as a
permanent OOP utility (not a one-off scratch script) so the same checks
can be rerun whenever the registry or matcher behaviour changes.

Construction is dependency-injected: pass an already-loaded
``MappingRegistry`` and ``BiosphereCatalog``, and the diagnostic walks
each requested ``(name, unit, top_cat, sub_cat)`` tuple. Tests inject
in-memory fakes; the linking pipeline can dispatch a CLI run after
``dds-link-all`` for an audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import openpyxl

from config import Settings
from domain import Bucket
from matching.bio_catalog import BiosphereCatalog
from registry import MappingRegistry


@dataclass(frozen=True)
class UnmatchedInputRow:
    """One row from the unlinked-biosphere artifact."""

    name: str
    unit: str
    top_cat: str | None
    sub_cat: str | None

    @property
    def bucket(self) -> Bucket:
        return Bucket.from_categories((self.top_cat or "", self.sub_cat or ""))


@dataclass(frozen=True)
class CandidateTrace:
    """How one registry candidate fared during simulation."""

    tier: int
    provenance: str
    target_db: str
    target_code: str
    target_name: str
    target_unit: str
    is_unmatchable: bool
    outcome: str  # "match" | "target_unresolved" | "unit_mismatch" | "skipped" | "halt_unmatchable"
    resolved_db: str = ""
    resolved_code: str = ""
    resolved_unit: str = ""
    multiplier: float | None = None
    detail_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnmatchedFlowOutcome:
    """End-to-end outcome for one input flow."""

    row: UnmatchedInputRow
    candidates: tuple[CandidateTrace, ...]
    matched: bool
    final_target_db: str | None
    final_target_code: str | None
    final_multiplier: float | None

    @property
    def status(self) -> str:
        if self.matched:
            return "matched"
        if any(c.outcome == "halt_unmatchable" for c in self.candidates):
            return "recognised_unmatchable"
        if not self.candidates:
            return "no_registry"
        unit_mism = sum(1 for c in self.candidates if c.outcome == "unit_mismatch")
        unres = sum(1 for c in self.candidates if c.outcome == "target_unresolved")
        if unit_mism and not unres:
            return "unit_mismatch_only"
        if unres and not unit_mism:
            return "target_unresolved_only"
        return "mixed"


@dataclass(frozen=True)
class UnmatchedDiagnosticReport:
    """Aggregate report across many ``UnmatchedFlowOutcome`` rows."""

    outcomes: tuple[UnmatchedFlowOutcome, ...]

    def render_text(self) -> str:
        lines: list[str] = []
        for o in self.outcomes:
            lines.append("=" * 78)
            lines.append(
                f">>> {o.row.name!r} | unit={o.row.unit!r} | "
                f"source_cat=({o.row.top_cat!r},{o.row.sub_cat!r}) | "
                f"bucket={o.row.bucket.value}"
            )
            if not o.candidates:
                lines.append("  NO REGISTRY CANDIDATES")
                continue
            lines.append(f"  {len(o.candidates)} candidates:")
            for c in o.candidates:
                lines.append(
                    f"    tier={c.tier:>2} prov={c.provenance!r:<55} -> outcome={c.outcome}"
                )
                for d in c.detail_lines:
                    lines.append(f"        {d}")
            if o.matched:
                lines.append(
                    f"  MATCH: {o.final_target_db}::{o.final_target_code} "
                    f"(multiplier={o.final_multiplier})"
                )
            else:
                lines.append("  *** EXHAUSTED — no candidate produced a target ***")
        return "\n".join(lines)

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.status] = out.get(o.status, 0) + 1
        return out


@dataclass(frozen=True)
class UnmatchedFlowDiagnostic:
    """Replay the matcher's candidate logic on a list of unlinked flows."""

    settings: Settings
    registry: MappingRegistry
    catalog: BiosphereCatalog

    def diagnose_many(self, rows: list[UnmatchedInputRow]) -> UnmatchedDiagnosticReport:
        return UnmatchedDiagnosticReport(outcomes=tuple(self.diagnose(r) for r in rows))

    def diagnose(self, row: UnmatchedInputRow) -> UnmatchedFlowOutcome:
        candidates = list(
            self.registry.biosphere_index.lookup("agb_flow", row.name.lower(), row.bucket)
        )
        candidates.sort(key=lambda c: (int(c.tier), c.provenance, c.target_code, c.target_name))

        traces: list[CandidateTrace] = []
        matched = False
        final_db = final_code = None
        final_mult: float | None = None

        # Mirror BiosphereMatcher: when no candidate produces a target, the
        # actual matcher consults ``unmatchable_index.is_unmatchable`` as a
        # post-hoc fallback. Surface that as a synthetic trace so the
        # diagnostic distinguishes "registry says dead-end" from "registry
        # has nothing".
        unmatch_recognised = self.registry.unmatchable_index.is_unmatchable(row.name, row.bucket)

        for cand in candidates:
            if cand.is_unmatchable:
                traces.append(self._trace(cand, "halt_unmatchable"))
                break
            if cand.tier.is_llm_gated() and not self.settings.apply_llm_overrides:
                traces.append(self._trace(cand, "skipped"))
                continue

            details: list[str] = []
            target = self._resolve_target(cand, row.bucket, details)
            if target is None:
                traces.append(self._trace(cand, "target_unresolved", details=details))
                continue
            db, code, t_unit = target
            mult = self._candidate_multiplier(cand, row.unit, t_unit)
            if mult is None:
                details.append(f"unit mismatch {row.unit!r} -> {t_unit!r} (no conversion)")
                traces.append(
                    self._trace(
                        cand,
                        "unit_mismatch",
                        resolved_db=db,
                        resolved_code=code,
                        resolved_unit=t_unit,
                        details=details,
                    )
                )
                continue
            details.append(f"unit conversion {row.unit!r} -> {t_unit!r} = {mult}")
            traces.append(
                self._trace(
                    cand,
                    "match",
                    resolved_db=db,
                    resolved_code=code,
                    resolved_unit=t_unit,
                    multiplier=mult,
                    details=details,
                )
            )
            matched = True
            final_db, final_code, final_mult = db, code, mult
            break

        # Append synthetic halt trace if no real candidate halted but the
        # unmatchable index marks this flow as a recognised dead-end.
        if (
            not matched
            and unmatch_recognised
            and not any(t.outcome == "halt_unmatchable" for t in traces)
        ):
            traces.append(
                CandidateTrace(
                    tier=12,
                    provenance="unmatchable_index",
                    target_db="",
                    target_code="",
                    target_name="",
                    target_unit="",
                    is_unmatchable=True,
                    outcome="halt_unmatchable",
                    detail_lines=("recognised by curated unmatchable list",),
                )
            )

        return UnmatchedFlowOutcome(
            row=row,
            candidates=tuple(traces),
            matched=matched,
            final_target_db=final_db,
            final_target_code=final_code,
            final_multiplier=final_mult,
        )

    @staticmethod
    def _trace(
        cand,
        outcome: str,
        resolved_db: str = "",
        resolved_code: str = "",
        resolved_unit: str = "",
        multiplier: float | None = None,
        details: list[str] | None = None,
    ) -> CandidateTrace:
        return CandidateTrace(
            tier=int(cand.tier),
            provenance=cand.provenance,
            target_db=cand.target_db,
            target_code=cand.target_code,
            target_name=cand.target_name,
            target_unit=cand.target_unit,
            is_unmatchable=cand.is_unmatchable,
            outcome=outcome,
            resolved_db=resolved_db,
            resolved_code=resolved_code,
            resolved_unit=resolved_unit,
            multiplier=multiplier,
            detail_lines=tuple(details or ()),
        )

    def _resolve_target(self, cand, bucket: Bucket, details: list[str]):
        """Mirror of ``BiosphereMatcher._resolve_target`` with logging."""
        if cand.target_db and cand.target_code:
            ref = self.catalog.get(cand.target_db, cand.target_code)
            unit = ref.unit if ref is not None else cand.target_unit
            details.append(
                f"direct: {cand.target_db}::{cand.target_code} -> "
                f"{'HIT' if ref else 'MISS'} unit={unit!r}"
            )
            return cand.target_db, cand.target_code, unit

        target_name = (cand.target_name or "").strip()
        if not target_name:
            details.append("no target_name and no target_code — skip")
            return None

        prefs: list[str] = []
        if cand.target_db:
            prefs.append(cand.target_db)
        for d in (
            self.settings.biosphere_db_name,
            self.settings.ef_db_name,
            "biosphere3",
        ):
            if d not in prefs:
                prefs.append(d)

        for db in prefs:
            hits = self.catalog.lookup_name_bucket(db, target_name, bucket)
            if hits:
                ref = sorted(hits, key=lambda r: r.code)[0]
                details.append(
                    f"same-bucket lookup({db!r}, {target_name!r}, "
                    f"bucket={bucket.value}): picked {ref.code} ({ref.unit!r})"
                )
                return ref.db, ref.code, ref.unit
            details.append(
                f"same-bucket lookup({db!r}, {target_name!r}, bucket={bucket.value}): no hits"
            )

        if bucket != Bucket.UNSPECIFIED:
            for db in prefs:
                hits = self.catalog.lookup_name_bucket(db, target_name, Bucket.UNSPECIFIED)
                if hits:
                    ref = sorted(hits, key=lambda r: r.code)[0]
                    details.append(
                        f"unspecified-bucket fallback ({db!r}): picked {ref.code} ({ref.unit!r})"
                    )
                    return ref.db, ref.code, ref.unit

        details.append(f"no target resolved across {prefs} (and fallbacks)")
        return None

    def _unit_mult(self, source_unit: str, target_unit: str) -> float | None:
        if not target_unit:
            return 1.0
        return self.registry.unit_converter.multiplier(source_unit, target_unit)

    def _candidate_multiplier(self, cand, source_unit: str, target_unit: str) -> float | None:
        """Mirror of ``BiosphereMatcher._resolve_candidate_multiplier``."""
        if cand.unit_conversion not in (None, 1.0):
            return float(cand.unit_conversion)
        return self._unit_mult(source_unit, target_unit)


@dataclass(frozen=True)
class UnlinkedXlsxLoader:
    """Read the unlinked-biosphere artifact into ``UnmatchedInputRow``s."""

    path: Path

    def read(self) -> list[UnmatchedInputRow]:
        wb = openpyxl.load_workbook(self.path, data_only=True)
        ws = wb.active
        out: list[UnmatchedInputRow] = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0 or not row or not row[0]:
                continue
            name, unit, top, sub, *_ = row
            out.append(
                UnmatchedInputRow(
                    name=str(name),
                    unit=str(unit) if unit is not None else "",
                    top_cat=str(top) if top is not None else None,
                    sub_cat=str(sub) if sub is not None else None,
                )
            )
        return out


@dataclass(frozen=True)
class UnmatchedDiagnosticRunner:
    """Convenience: discover everything from ``Settings`` and produce a report."""

    settings: Settings = field(default_factory=Settings)

    def run(self, unlinked_xlsx: Path | None = None) -> UnmatchedDiagnosticReport:
        registry = MappingRegistry.load(self.settings)
        catalog = BiosphereCatalog.load(
            self.settings.paths.registry_biosphere_catalog,
            db_names=(
                self.settings.biosphere_db_name,
                self.settings.ef_db_name,
                "biosphere3",
            ),
        )
        diag = UnmatchedFlowDiagnostic(settings=self.settings, registry=registry, catalog=catalog)
        path = unlinked_xlsx or (self.settings.paths.unlinked / "biosphere_unlinked.xlsx")
        rows = UnlinkedXlsxLoader(path=path).read()
        return diag.diagnose_many(rows)
