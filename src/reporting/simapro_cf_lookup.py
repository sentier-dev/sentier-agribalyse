"""``SimaProCfLookup`` — per-flow SimaPro CF lookup for the flow panel.

Wraps ``registry/cf_comparison_by_code.parquet`` (produced by ``dds-compare-cfs``
via :class:`reporting.CfComparisonByCodeBuilder`: one row per registry biosphere
``code`` × method, carrying SimaPro's properly-matched characterization factor
``cf_simapro``) as an in-memory map keyed by ``(biosphere code, method
category)``. The flow decomposition emitter consults it to put SimaPro's CF
beside our EF CF for every characterised flow it serialises.

The comparison is a deterministic per-flow 1:1 join (``FlowLevelCfJoiner``):
every registry biosphere ``code`` resolves to exactly one SimaPro CF via full
``(name, compartment, sub_compartment)`` identity, so the SimaPro CF shown is the
value SimaPro/ADEME actually applied to that flow — not a name-heuristic match
(and not one of several "candidates"). The sidecar's ``method`` column is the
registry *category* (e.g.
``"acidification"``), which uniquely identifies a method, so a
``(code, category)`` pair resolves to a single SimaPro CF. Only flows for which
SimaPro has a comparable CF are stored; a miss returns ``None`` and the
dashboard renders an em-dash.

OOP-only per ``CLAUDE.md``: frozen dataclasses, dependencies injected,
no module-level helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class SimaProCfEntry:
    """SimaPro's CF for one flow under one method, plus how it was matched.

    ``sp_name`` / ``ef_name`` are the comparison's matched molecule names on the
    SimaPro and registry sides — surfaced in the flow panel so a reviewer can see
    *which* molecule each CF came from (empty string when the sidecar omits them).
    """

    sp_cf: float
    provenance: str
    sp_name: str = ""
    ef_name: str = ""


@dataclass(frozen=True)
class SimaProCfLookup:
    """Map ``(code, method_category) -> SimaProCfEntry``."""

    by_key: dict[tuple[str, str], SimaProCfEntry]

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "code",
        "method",
        "cf_simapro",
        "match_basis",
    )

    @classmethod
    def from_parquet(cls, path: Path) -> SimaProCfLookup:
        return cls.from_dataframe(pd.read_parquet(path))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SimaProCfLookup:
        missing = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"cf_comparison_by_code frame missing columns {missing}; got {list(df.columns)}"
            )
        has_sp_name = "name_simapro" in df.columns
        has_ef_name = "name_registry" in df.columns

        def _name(value: object) -> str:
            return "" if value is None or pd.isna(value) else str(value)

        by_key: dict[tuple[str, str], SimaProCfEntry] = {}
        for row in df.itertuples(index=False):
            sp = row.cf_simapro
            if sp is None or pd.isna(sp):
                continue
            key = (str(row.code), str(row.method))
            # First-wins over the parquet's stable row order keeps the map
            # deterministic; the sidecar is already deduped on (method, code).
            if key in by_key:
                continue
            basis = row.match_basis
            by_key[key] = SimaProCfEntry(
                sp_cf=float(sp),
                provenance=("" if basis is None or pd.isna(basis) else str(basis)),
                sp_name=_name(row.name_simapro) if has_sp_name else "",
                ef_name=_name(row.name_registry) if has_ef_name else "",
            )
        return cls(by_key=by_key)

    def get(self, code: str, method_category: str) -> SimaProCfEntry | None:
        return self.by_key.get((code, method_category))
