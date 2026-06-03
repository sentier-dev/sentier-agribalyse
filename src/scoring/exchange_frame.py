"""``ExchangeFrame`` — long-form DataFrame of the linked exchange graph.

Schema (every row = one edge in the technosphere or biosphere)::

    output_id   int64    activity that owns this edge (the "from" node)
    input_id    int64    flow on the other end of this edge (the "to" node)
    amount      float64  signed exchange amount (per unit of output activity)
    edge_type   string   'production' | 'technosphere' | 'biosphere'
                         | 'substitution' | 'generic production' | ...
    is_biosphere bool    True iff input_id refers to a biosphere flow

Why long-form: every downstream builder (technosphere, biosphere, CF)
is a single pass + ``scipy.sparse.coo_matrix(...)`` from a 4-column
slice. No graph walk, no allocation, no SQLite. Pickling a parquet of
this frame is also the cheapest possible content-addressable cache key:
hash the parquet bytes once, key the matrices by that hash.

Compare with bw2data's editable graph: that one carries an Activity ⇆
Exchange ⇆ Activity object graph, an FTS5 search index, and a peewee
session that turns every read into a multifunctional-allocation
checkpoint. We don't need any of that to compute LCA scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Edge-type taxonomy. Anything in ``_PRODUCTION_TYPES`` makes the row
# part of the technosphere matrix's "production" structure (which row
# represents which product). ``_BIOSPHERE_TYPES`` maps to rows of the
# biosphere matrix B. Everything else is a regular technosphere
# coefficient (contributes to the off-diagonal).
_PRODUCTION_TYPES = frozenset({"production", "generic production"})
_TECHNOSPHERE_TYPES = frozenset({"technosphere", "substitution"})
_BIOSPHERE_TYPES = frozenset({"biosphere"})

_REQUIRED_COLUMNS = ("output_id", "input_id", "amount", "edge_type", "is_biosphere")

# ``allocation_factor`` is optional — present only after multifunctional
# inputs are tagged for the allocator. The default of 1.0 means "no
# split" and lets single-product activities flow through untouched.
_ALLOCATION_COLUMN = "allocation_factor"


@dataclass(frozen=True)
class ExchangeFrame:
    """Immutable wrapper around the long-form exchange table.

    Validates schema once at construction. Every method returns a new
    ``ExchangeFrame`` (or a derived DataFrame) — never mutates the
    underlying frame. This is what makes the data plane safe to share
    across worker processes without locks: each worker can slice,
    project, group; nothing they do can alter what another worker sees.
    """

    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"ExchangeFrame is missing required columns: {missing}. "
                f"Got columns={list(self.df.columns)}"
            )
        # Catch the silent-int-as-string failure mode early — the matrix
        # builder relies on output_id / input_id being numeric to map
        # to row/column integers via ``pd.factorize``.
        for col in ("output_id", "input_id"):
            if not pd.api.types.is_integer_dtype(self.df[col]):
                raise ValueError(
                    f"ExchangeFrame column {col!r} must be integer, got {self.df[col].dtype}"
                )
        if not pd.api.types.is_float_dtype(self.df["amount"]):
            raise ValueError(
                f"ExchangeFrame column 'amount' must be float, got {self.df['amount'].dtype}"
            )

    # ------------------------------------------------------------------
    # Slices.

    @property
    def production(self) -> pd.DataFrame:
        """Rows that define which activity produces which product."""
        return self.df[self.df["edge_type"].isin(_PRODUCTION_TYPES)].copy()

    @property
    def technosphere(self) -> pd.DataFrame:
        """Production + technosphere consumption + substitution.

        Everything that contributes to the technosphere matrix A.
        """
        types = _PRODUCTION_TYPES | _TECHNOSPHERE_TYPES
        return self.df[self.df["edge_type"].isin(types)].copy()

    @property
    def biosphere(self) -> pd.DataFrame:
        """Rows feeding the biosphere matrix B."""
        return self.df[self.df["edge_type"].isin(_BIOSPHERE_TYPES)].copy()

    # ------------------------------------------------------------------
    # Identity / shape.

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def activities(self) -> pd.Index:
        """All distinct ``output_id`` values — one per activity column
        of the technosphere."""
        return pd.Index(sorted(self.df["output_id"].unique()))

    @property
    def products(self) -> pd.Index:
        """Distinct product ids — the inputs of production edges."""
        prod = self.production
        return pd.Index(sorted(prod["input_id"].unique()))

    @property
    def biosphere_flows(self) -> pd.Index:
        bio = self.biosphere
        return pd.Index(sorted(bio["input_id"].unique()))

    # ------------------------------------------------------------------
    # Construction helpers.

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> ExchangeFrame:
        """Coerce a partially-typed frame into the schema. The caller
        is responsible for column presence; we only widen dtypes."""
        out = df.copy()
        for col in ("output_id", "input_id"):
            out[col] = out[col].astype("int64")
        out["amount"] = out["amount"].astype("float64")
        out["edge_type"] = out["edge_type"].astype("string")
        out["is_biosphere"] = out["is_biosphere"].astype(bool)
        return cls(df=out[list(_REQUIRED_COLUMNS)].reset_index(drop=True))
