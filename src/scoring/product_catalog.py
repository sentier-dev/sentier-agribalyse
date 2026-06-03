"""``ProductCatalogBuilder`` — emit one parquet row per linked activity.

Replaces ``BacktestPipeline._map_products``'s reliance on
``bw2data.Database(agb_db_name)`` (which won't exist after Phase L3
cuts the SQLite tail). The product catalog is a flat parquet of every
activity in ``sp.data`` keyed on ``(database, code)``, carrying the
fields backtests need to map a CIQUAL code → product id:

| col         | dtype  | notes                                              |
|-------------|--------|----------------------------------------------------|
| database    | string | source database for the activity                   |
| code        | string | activity code                                      |
| name        | string | activity name (post-RestoreSimaproNamesTransform)  |
| type        | string | ``process`` / ``product`` / ``multifunctional`` /…  |
| unit        | string | reference unit                                     |
| product_id  | int64  | ``ExchangeFrameBuilder.flow_id_for((db, code))``    |

Determinism: rows sorted by ``(database, code)`` before write. Atomic
write via ``ParquetAtomicWriter`` so a SIGKILL leaves no partial file
at the publish path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.parquet_io import ParquetAtomicWriter
from scoring.exchange_frame_builder import ExchangeFrameBuilder


@dataclass(frozen=True)
class ProductCatalogBuilder:
    """Stateless builder. ``build(sp_data, target)`` writes the parquet."""

    def build(self, sp_data: Iterable[dict], target: Path) -> Path:
        """Project ``sp_data`` into the product catalog parquet.

        One row per activity. ``product_id`` is the integer id assigned
        by :meth:`ExchangeFrameBuilder.flow_id_for` so downstream
        consumers can join directly against the technosphere row map.
        """
        rows = [self._row_for(ds) for ds in sp_data]
        df = pd.DataFrame(
            rows,
            columns=["database", "code", "name", "type", "unit", "product_id"],
        )
        df = df.astype(
            {
                "database": "string",
                "code": "string",
                "name": "string",
                "type": "string",
                "unit": "string",
                "product_id": "int64",
            }
        )
        df = df.drop_duplicates(subset=["database", "code"])
        df = df.sort_values(by=["database", "code"]).reset_index(drop=True)
        ParquetAtomicWriter.write(df, target)
        return target

    @staticmethod
    def _row_for(ds: dict) -> dict:
        database = str(ds.get("database") or "")
        code = str(ds.get("code") or "")
        # In the AGB / SimaPro shape, an activity's production edge
        # may point at a *different* (db, code) than the activity
        # itself — the product is its own node. The technosphere row
        # map is keyed by the production edge's ``input``, so we have
        # to resolve to that key here. If there's no linked production
        # (e.g., a ``type='product'`` stub or an unlinked dummy), fall
        # back to the activity's own (db, code) — the entry will be
        # absent from the row map at scoring time and ``NativeLciaScorer``
        # will report ``"product id not in technosphere"`` rather than
        # silently mis-scoring.
        product_key = (database, code)
        for exc in ds.get("exchanges", ()) or ():
            etype = (exc.get("type") or "").lower()
            if not etype.startswith("production"):
                continue
            inp = exc.get("input")
            if inp is None:
                continue
            product_key = (str(inp[0]), str(inp[1]))
            break
        return {
            "database": database,
            "code": code,
            "name": str(ds.get("name") or ""),
            "type": str(ds.get("type") or ""),
            "unit": str(ds.get("unit") or ""),
            "product_id": ExchangeFrameBuilder.flow_id_for(product_key),
        }


@dataclass(frozen=True)
class ProductCatalog:
    """Read-only view over ``registry/product_catalog.parquet``.

    Provides a ``(database, code) → product_id`` lookup used by L4
    scoring consumers that must not go through ``bw2data.Database``.
    """

    _df: pd.DataFrame

    @classmethod
    def load(cls, path: Path) -> ProductCatalog:
        """Load the product catalog from *path*."""
        df = pd.read_parquet(path)
        return cls(_df=df)

    _SCOREABLE_TYPES: frozenset[str] = frozenset(
        {"process", "multifunctional", "product", "processwithreferenceproduct"}
    )

    def product_id_for(self, database: str, code: str) -> int | None:
        """Return the integer product_id for *(database, code)*, or None."""
        mask = (self._df["database"] == database) & (self._df["code"] == code)
        matches = self._df.loc[mask, "product_id"]
        if matches.empty:
            return None
        return int(matches.iloc[0])

    def scoreable_entries(self) -> pd.DataFrame:
        """Return a DataFrame of all activities that can be scored.

        Filters to the activity types that ``BacktestPipeline`` maps:
        ``process``, ``multifunctional``, ``product``, and
        ``processwithreferenceproduct``. Columns: database, code, name, type,
        unit, product_id.
        """
        return self._df[self._df["type"].isin(self._SCOREABLE_TYPES)].reset_index(drop=True)
