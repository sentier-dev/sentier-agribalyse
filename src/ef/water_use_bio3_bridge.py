"""``WaterUseBio3Bridge`` — curated EF v3.1 (adapted) CF mapping for water-use.

The bw2io snapshot at ``source/ef-v31-methods.json`` carries only 5 inherited
``Water [air, *]`` CFs on the ecoinvent biosphere side — none of the natural-
resource water inputs (lake / river / well / turbine / cooling) nor the
water-compartment releases get CFs. The result is the shrimp-class
under-scoring documented in the session log: pond water inputs of ~5 m³/kg
are unmapped, scoring 0 contribution against ADEME's ~220 m³_eq reference.

``SimaProCfFilter.augment_rows`` was an earlier attempt at this same gap.
It misbehaved (337× over-count) because it matched SimaPro names character-
for-character: SimaPro carries both ``Water`` (per-kg, CF=-0.042955) and
``Water/m3`` (per-m³, CF=-42.955). The bio3 catalog name is just
``Water``, so augment picked the per-kg variant and applied it to m³-unit
matrix entries — silently 1000× too small on releases while inputs got the
correct m³ value. The asymmetric magnitudes broke balanced flow
cancellation (turbine / cooling in & out) and detonated cumulative scores.

This class skips that pitfall by curating the mapping explicitly. Each
entry is a ``(bio3_code, cf)`` row with the CF normalised to m³ (the unit
the matrix uses). Both inputs (+CF) and matching releases (-CF) are
included so balanced flows cancel correctly. Pairs with disabling
``AwareConsumptionCorrectionBuilder`` — once the base CFs are complete on
the Q row, the per-activity asymmetric correction becomes a double-count.

The JSON file format::

    {
      "version": 1,
      "method_key": ["ecoinvent-3.9.1", "EF v3.1", "water use", "..."],
      "mappings": [
        {
          "code": "8c75e7ab-8ab8-41e4-b394-c166ff5b050d",
          "name": "Water, river",
          "compartment": "natural resource/in water",
          "cf": 42.95,
          "direction": "input",
          "rationale": "free-form note kept for human review"
        }
      ]
    }

Database is always ``ecoinvent-3.9.1-biosphere`` (the only biosphere
database the scoring matrix actually carries — see
``RegionalCfRegistryBuilder`` for the same precedent).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar


@dataclass(frozen=True)
class WaterUseBio3Bridge:
    """Frozen view of the curated EF v3.1 → bio3 water-use CF table."""

    _table: Mapping[str, float]
    _method_key: tuple[str, ...]

    SCHEMA_VERSION: ClassVar[int] = 1
    BIO3_DATABASE: ClassVar[str] = "ecoinvent-3.9.1-biosphere"

    @classmethod
    def empty(cls) -> WaterUseBio3Bridge:
        return cls(_table=MappingProxyType({}), _method_key=())

    @classmethod
    def load(cls, path: Path) -> WaterUseBio3Bridge:
        """Load the bridge from JSON; missing file → empty (no-op)."""
        if not path.exists():
            return cls.empty()
        payload: Any = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                f"water-use bio3 bridge at {path}: expected object root, got {type(payload).__name__}"
            )
        version = payload.get("version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"water-use bio3 bridge at {path}: unsupported version {version!r}; "
                f"expected {cls.SCHEMA_VERSION}"
            )
        method_key = tuple(payload.get("method_key", ()))
        if len(method_key) != 4:
            raise ValueError(
                f"water-use bio3 bridge at {path}: method_key must be a 4-tuple, got {method_key!r}"
            )
        rows = payload.get("mappings", [])
        if not isinstance(rows, list):
            raise ValueError(f"water-use bio3 bridge at {path}: 'mappings' must be a list")
        table: dict[str, float] = {}
        for row in rows:
            code, cf = cls._read_row(row, path)
            if code in table:
                raise ValueError(f"water-use bio3 bridge at {path}: duplicate code {code}")
            table[code] = cf
        return cls(_table=MappingProxyType(table), _method_key=method_key)

    @staticmethod
    def _read_row(row: Any, path: Path) -> tuple[str, float]:
        if not isinstance(row, dict):
            raise ValueError(
                f"water-use bio3 bridge at {path}: row must be an object, got {type(row).__name__}"
            )
        code = str(row.get("code") or "").strip()
        if not code:
            raise ValueError(f"water-use bio3 bridge at {path}: missing code in row {row!r}")
        cf_raw = row.get("cf")
        if cf_raw is None:
            raise ValueError(f"water-use bio3 bridge at {path}: missing cf in row {row!r}")
        try:
            cf = float(cf_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"water-use bio3 bridge at {path}: non-numeric cf in row {row!r}"
            ) from exc
        return code, cf

    @property
    def method_key(self) -> tuple[str, ...]:
        return self._method_key

    def cf_rows(self) -> list[dict]:
        """Return ``[{"database", "code", "amount"}, ...]`` for registry merge.

        Mirrors the shape ``WaterResourceCfAugmenter.cf_rows`` produces so
        ``MethodCfRegistryBuilder._merge_augmenter_rows`` consumes them
        uniformly.
        """
        return [
            {"database": self.BIO3_DATABASE, "code": code, "amount": cf}
            for code, cf in sorted(self._table.items())
        ]

    def items(self) -> Iterator[tuple[str, float]]:
        yield from self._table.items()

    def __len__(self) -> int:
        return len(self._table)

    def __bool__(self) -> bool:
        return bool(self._table)
