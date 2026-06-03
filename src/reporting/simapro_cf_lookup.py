"""``SimaProCfLookup`` — per-flow SimaPro CF lookup for the flow panel.

Wraps ``cache/cf_per_flow_joined.parquet`` (produced by ``dds-compare-cfs``:
one row per biosphere flow × method, carrying SimaPro's characterization
factor ``sp_cf`` joined against our EF CF ``ef_cf``) as an in-memory map
keyed by ``(biosphere code, method category, method indicator)``. The flow
decomposition emitter consults it to put SimaPro's CF beside our EF CF for
every characterised flow it serialises.

The key is unique in the joined frame — each distinct biosphere ``code``
already encodes its full context (compartment + sub-compartment), so a
``(code, category, indicator)`` triple resolves to a single SimaPro CF.
Only flows for which SimaPro has a comparable CF are stored; a miss returns
``None`` and the dashboard renders an em-dash.

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
    """SimaPro's CF for one flow under one method, plus how it was matched."""

    sp_cf: float
    provenance: str


@dataclass(frozen=True)
class SimaProCfLookup:
    """Map ``(code, category, indicator) -> SimaProCfEntry``."""

    by_key: dict[tuple[str, str, str], SimaProCfEntry]

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "code",
        "method_category",
        "method_indicator",
        "sp_cf",
        "sp_match_provenance",
    )

    @classmethod
    def from_parquet(cls, path: Path) -> SimaProCfLookup:
        return cls.from_dataframe(pd.read_parquet(path))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SimaProCfLookup:
        missing = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"cf_per_flow_joined frame missing columns {missing}; got {list(df.columns)}"
            )
        by_key: dict[tuple[str, str, str], SimaProCfEntry] = {}
        for row in df.itertuples(index=False):
            sp = row.sp_cf
            if sp is None or pd.isna(sp):
                continue
            key = (str(row.code), str(row.method_category), str(row.method_indicator))
            # First-wins over the parquet's stable row order keeps the map
            # deterministic; the key is unique in practice (verified on build).
            if key in by_key:
                continue
            prov = row.sp_match_provenance
            by_key[key] = SimaProCfEntry(
                sp_cf=float(sp),
                provenance=("" if prov is None or pd.isna(prov) else str(prov)),
            )
        return cls(by_key=by_key)

    def get(self, code: str, category: str, indicator: str) -> SimaProCfEntry | None:
        return self.by_key.get((code, category, indicator))
