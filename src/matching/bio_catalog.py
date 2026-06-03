"""``BiosphereCatalog`` — parquet-backed in-memory index of biosphere DBs.

The matcher uses this to resolve target codes when a registry row says
"map to flow X by name" without a UUID. Biosphere DBs (``biosphere3``,
``ecoinvent-3.9.1-biosphere``, ``ef``) are indexed by
``(db, name_lower, bucket) → [(code, unit)]``.

The source of truth is ``registry/biosphere_catalog.parquet``, written by
``BiosphereRegistryBuilder`` at build time. ``load(path)`` reads that
parquet and returns an immutable catalog. Reading from parquet (not
SQLite) keeps the runtime path off ``bw2data`` — see
``docs/REFACTOR_LINKING.md`` Phase L1 for the design rationale.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from domain import Bucket


@dataclass(frozen=True)
class BioFlowRef:
    db: str
    code: str
    name: str
    unit: str
    bucket: Bucket
    cas: str | None = None
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class BiosphereCatalog:
    """Parquet-backed snapshots of biosphere/EF databases keyed for fast lookup."""

    db_names: tuple[str, ...]
    flows: tuple[BioFlowRef, ...]

    # ------------------------------------------------------------------
    # Construction.

    @classmethod
    def load(
        cls,
        path: Path,
        db_names: Iterable[str] | None = None,
    ) -> BiosphereCatalog:
        """Read ``biosphere_catalog.parquet`` into an in-memory catalog.

        ``db_names`` optionally filters the catalog down to a subset of
        databases. ``None`` keeps every database present in the parquet.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"biosphere catalog parquet not found: {path}. "
                f"Run `dds-build-biosphere-catalog` to materialise it."
            )

        df = pd.read_parquet(path)
        if db_names is not None:
            wanted = set(db_names)
            df = df[df["database"].isin(wanted)]

        flows: list[BioFlowRef] = []
        for row in df.itertuples(index=False):
            cats = tuple(row.categories) if row.categories is not None else ()
            cas = row.cas if isinstance(row.cas, str) and row.cas else None
            flows.append(
                BioFlowRef(
                    db=str(row.database),
                    code=str(row.code),
                    name=str(row.name),
                    unit=str(row.unit),
                    bucket=Bucket.from_categories(cats),
                    cas=cas,
                    categories=tuple(str(c) for c in cats),
                )
            )
        present_dbs = tuple(sorted({f.db for f in flows}))
        return cls(db_names=present_dbs, flows=tuple(flows))

    # ------------------------------------------------------------------
    # Indices — built lazily, cached on first access.

    @cached_property
    def by_name_bucket(self) -> dict[tuple[str, str, str], list[BioFlowRef]]:
        out: dict[tuple[str, str, str], list[BioFlowRef]] = defaultdict(list)
        for f in self.flows:
            out[(f.db, f.name.strip().lower(), f.bucket.value)].append(f)
        return dict(out)

    @cached_property
    def by_db_code(self) -> dict[tuple[str, str], BioFlowRef]:
        return {(f.db, f.code): f for f in self.flows}

    @cached_property
    def by_cas(self) -> dict[tuple[str, str], list[BioFlowRef]]:
        """``(db, cas) → flows``. CAS is sparse so this is small."""
        out: dict[tuple[str, str], list[BioFlowRef]] = defaultdict(list)
        for f in self.flows:
            if f.cas:
                out[(f.db, f.cas.strip())].append(f)
        return dict(out)

    # ------------------------------------------------------------------
    # Public lookup surface consumed by ``BiosphereMatcher``.

    def lookup_name_bucket(self, db: str, name: str, bucket: Bucket | str) -> list[BioFlowRef]:
        b = bucket.value if isinstance(bucket, Bucket) else str(bucket)
        return self.by_name_bucket.get((db, (name or "").strip().lower(), b), [])

    def get(self, db: str, code: str) -> BioFlowRef | None:
        return self.by_db_code.get((db, code))

    def lookup_cas(self, db: str, cas: str) -> list[BioFlowRef]:
        return self.by_cas.get((db, cas.strip()), [])
