"""``EcoinventCatalog`` — parquet-backed in-memory index of ecoinvent activities.

The technosphere matcher uses this to resolve ``(name, unit, location,
reference product)`` to an ``(database, code)`` activity reference. Built
once into ``registry/ecoinvent_catalog.parquet`` by ``EcoinventCatalogBuilder``;
read at runtime via ``EcoinventCatalog.load(path)`` — no SQLite, no bw2data.

Schema (one row per ecoinvent activity):

| col              | dtype  | notes                                          |
|------------------|--------|------------------------------------------------|
| database         | string | ``settings.ecoinvent_db_name`` (e.g. cutoff)   |
| code             | string | ecoinvent activity uuid                        |
| name             | string | activity name (lowercased on lookup)           |
| unit             | string | reference unit                                 |
| location         | string | RoW / GLO / FR / ...                           |
| reference_product| string | reference product name (lowercased on lookup)  |

Determinism: rows sorted by ``(database, code)`` before write so two builds
against the same project produce byte-equal parquets.

Disambiguation: ``match_full`` and ``match_relaxed`` return ``None`` when
the lookup key resolves to more than one activity — same behavior as
``bw2io.SimaProBlockCSVImporter.match_database``, which leaves ambiguous
exchanges unlinked rather than picking arbitrarily.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from core.logging import Logging
from core.parquet_io import ParquetAtomicWriter


@dataclass(frozen=True)
class EcoinventActivityRef:
    """One ecoinvent activity, the slim subset the technosphere matcher needs."""

    db: str
    code: str
    name: str
    unit: str
    location: str
    reference_product: str


@dataclass(frozen=True)
class EcoinventCatalog:
    """Parquet-backed snapshot of an ecoinvent database keyed for fast lookup."""

    db_names: tuple[str, ...]
    activities: tuple[EcoinventActivityRef, ...]

    # ------------------------------------------------------------------
    # Construction.

    @classmethod
    def load(
        cls,
        path: Path,
        db_names: Iterable[str] | None = None,
    ) -> EcoinventCatalog:
        """Read ``ecoinvent_catalog.parquet`` into an in-memory catalog.

        ``db_names`` optionally filters the catalog down to a subset of
        databases. ``None`` keeps every database present in the parquet.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"ecoinvent catalog parquet not found: {path}. "
                f"Run `dds-build-ecoinvent-catalog` to materialise it."
            )

        df = pd.read_parquet(path)
        if db_names is not None:
            wanted = set(db_names)
            df = df[df["database"].isin(wanted)]

        activities: list[EcoinventActivityRef] = []
        for row in df.itertuples(index=False):
            activities.append(
                EcoinventActivityRef(
                    db=str(row.database),
                    code=str(row.code),
                    name=str(row.name) if row.name else "",
                    unit=str(row.unit) if row.unit else "",
                    location=str(row.location) if row.location else "",
                    reference_product=(str(row.reference_product) if row.reference_product else ""),
                )
            )
        present_dbs = tuple(sorted({a.db for a in activities}))
        return cls(db_names=present_dbs, activities=tuple(activities))

    # ------------------------------------------------------------------
    # Indices — built lazily, cached on first access.

    @cached_property
    def by_db_code(self) -> dict[tuple[str, str], EcoinventActivityRef]:
        return {(a.db, a.code): a for a in self.activities}

    @cached_property
    def _by_full_key(
        self,
    ) -> dict[tuple[str, str, str, str, str], list[EcoinventActivityRef]]:
        """``(db, name_lower, unit, location, refprod_lower) → activities``."""
        out: dict[tuple[str, str, str, str, str], list[EcoinventActivityRef]] = defaultdict(list)
        for a in self.activities:
            key = (
                a.db,
                a.name.strip().lower(),
                a.unit,
                a.location,
                a.reference_product.strip().lower(),
            )
            out[key].append(a)
        return dict(out)

    @cached_property
    def _by_nul_key(self) -> dict[tuple[str, str, str, str], list[EcoinventActivityRef]]:
        """``(db, name_lower, unit, location) → activities``. Refprod-agnostic fallback."""
        out: dict[tuple[str, str, str, str], list[EcoinventActivityRef]] = defaultdict(list)
        for a in self.activities:
            key = (a.db, a.name.strip().lower(), a.unit, a.location)
            out[key].append(a)
        return dict(out)

    # ------------------------------------------------------------------
    # Public lookup surface consumed by ``TechnosphereMatcher``.

    def get(self, db: str, code: str) -> EcoinventActivityRef | None:
        return self.by_db_code.get((db, code))

    def match_full(
        self,
        db: str,
        name: str,
        unit: str,
        location: str,
        reference_product: str,
    ) -> EcoinventActivityRef | None:
        """Strict 4-field match. Returns the unique activity, or ``None`` if 0 or >1.

        Mirrors ``sp.match_database(ei_db, fields=["name","unit","location","reference product"])``:
        ambiguous keys leave the exchange unlinked.
        """
        key = (
            db,
            (name or "").strip().lower(),
            unit or "",
            location or "",
            (reference_product or "").strip().lower(),
        )
        hits = self._by_full_key.get(key, [])
        return hits[0] if len(hits) == 1 else None

    def match_relaxed(
        self,
        db: str,
        name: str,
        unit: str,
        location: str,
    ) -> EcoinventActivityRef | None:
        """3-field match (drops ``reference product``). Returns unique activity or ``None``."""
        key = (
            db,
            (name or "").strip().lower(),
            unit or "",
            location or "",
        )
        hits = self._by_nul_key.get(key, [])
        return hits[0] if len(hits) == 1 else None


@dataclass(frozen=True)
class EcoinventCatalogBuilder:
    """Build ``registry/ecoinvent_catalog.parquet`` from a JSON snapshot.

    REFACTOR_FINAL F6: bw2data is no longer queried at build time. The
    ecoinvent activities are snapshotted once into
    ``source/ecoinvent-3.9.1-cutoff-activities.json`` (or whatever
    ``ecoinvent_activities_json`` resolves to) and the builder converts
    that JSON to the parquet shape the runtime catalog reads.
    """

    settings: Settings

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Public entry point.

    def build(self) -> Path:
        s = self.settings
        s.paths.ensure_runtime_dirs()

        snap = self._load_snapshot()
        db_name = snap.get("db_name") or s.ecoinvent_db_name
        rows = [self._row_for(db_name, activity) for activity in snap.get("activities", [])]
        df = self._frame(rows)
        target = s.paths.registry_ecoinvent_catalog
        ParquetAtomicWriter.write(df, target)
        self._log.info(
            "ecoinvent.catalog.built",
            path=str(target),
            n_rows=len(df),
            db=db_name,
        )
        return target

    def _load_snapshot(self) -> dict[str, Any]:
        path = self.settings.paths.source / "ecoinvent-3.9.1-cutoff-activities.json"
        if not path.exists():
            raise FileNotFoundError(
                f"ecoinvent activity snapshot missing: {path}. "
                f"Snapshot via REFACTOR_FINAL phase F6 (one-time bootstrap)."
            )
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Helpers — small, single-purpose, testable in isolation.

    @staticmethod
    def _row_for(db_name: str, activity: dict) -> dict:
        return {
            "database": db_name,
            "code": str(activity.get("code", "") or ""),
            "name": str(activity.get("name", "") or ""),
            "unit": str(activity.get("unit", "") or ""),
            "location": str(activity.get("location", "") or ""),
            "reference_product": str(
                activity.get("reference_product") or activity.get("reference product") or ""
            ),
        }

    @staticmethod
    def _frame(rows: list[dict]) -> pd.DataFrame:
        cols = ["database", "code", "name", "unit", "location", "reference_product"]
        df = pd.DataFrame(rows, columns=cols)
        df = df.astype({c: "string" for c in cols})
        df = df.sort_values(by=["database", "code"], kind="mergesort").reset_index(drop=True)
        return df
