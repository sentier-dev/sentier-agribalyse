"""``ActivityLocationOverrides`` — per-activity ISO-2 location overrides.

AGB stand-in activities frequently inherit aggregate ecoinvent locations
(``GLO`` / ``RoW`` / ``RER``) that hide the regional context ADEME's
synthese assumes. The default resolver collapses those aggregates to
``AGGREGATE_FALLBACK = None`` (see ``link_all.py``) which leaves the
scorer on the global CF — fine for tropical-fruit GLOs where the global
mean is roughly right, but catastrophically wrong for FR-greenhouse
crops where ADEME implicitly applies a FR water context (sweet pepper
greenhouse: score 1.42 vs ADEME 0.29, 4.8× over).

This class loads a curated JSON of ``(database, code) → ISO-2`` rows
that the link pipeline applies *after* the catalog / name-parser passes,
so the override is final for the listed activities and leaves every
other column at its parsed/global resolution. Curation is intentional —
we tested a blanket ``GLO → FR`` fallback and it broke the tropical
products it doesn't apply to (mean |Δ| 35.9% → 66.9%, outliers 307 →
980; see ``link_all.py:566-577``).

File format::

    {
      "version": 1,
      "overrides": [
        {
          "database": "agribalyse-3.2",
          "code": "AGRIBALU000000003101321",
          "location": "FR",
          "rationale": "free-form note kept for human review"
        }
      ]
    }

Missing file → empty overrides (no-op). Duplicate ``(database, code)``
pairs raise at load time so the file stays the single source of truth.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar


@dataclass(frozen=True)
class ActivityLocationOverrides:
    """Frozen view over the curated ``(database, code) → location`` table."""

    _table: Mapping[tuple[str, str], str]

    SCHEMA_VERSION: ClassVar[int] = 1

    @classmethod
    def empty(cls) -> ActivityLocationOverrides:
        return cls(_table=MappingProxyType({}))

    @classmethod
    def load(cls, path: Path) -> ActivityLocationOverrides:
        """Load overrides from a JSON file. Missing file → empty (no-op)."""
        if not path.exists():
            return cls.empty()
        payload: Any = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                f"activity location overrides at {path}: expected object root, got {type(payload).__name__}"
            )
        version = payload.get("version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"activity location overrides at {path}: unsupported version {version!r}; "
                f"expected {cls.SCHEMA_VERSION}"
            )
        rows = payload.get("overrides", [])
        if not isinstance(rows, list):
            raise ValueError(f"activity location overrides at {path}: 'overrides' must be a list")
        table: dict[tuple[str, str], str] = {}
        for row in rows:
            db, code, loc = cls._read_row(row, path)
            key = (db, code)
            if key in table:
                raise ValueError(f"activity location overrides at {path}: duplicate ({db}, {code})")
            table[key] = loc
        return cls(_table=MappingProxyType(table))

    @staticmethod
    def _read_row(row: Any, path: Path) -> tuple[str, str, str]:
        if not isinstance(row, dict):
            raise ValueError(
                f"activity location overrides at {path}: row must be an object, got {type(row).__name__}"
            )
        db = str(row.get("database") or "")
        code = str(row.get("code") or "")
        loc = str(row.get("location") or "").strip().upper()
        if not db or not code:
            raise ValueError(
                f"activity location overrides at {path}: missing database/code in {row!r}"
            )
        if not loc:
            raise ValueError(f"activity location overrides at {path}: empty location in {row!r}")
        return db, code, loc

    def location_for(self, database: str, code: str) -> str | None:
        return self._table.get((database, code))

    def items(self) -> Iterator[tuple[str, str, str]]:
        """Yield ``(database, code, location)`` tuples for all overrides."""
        for (db, code), loc in self._table.items():
            yield db, code, loc

    def __len__(self) -> int:
        return len(self._table)

    def __bool__(self) -> bool:
        return bool(self._table)
