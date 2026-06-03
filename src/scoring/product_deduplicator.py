"""``ProductDeduplicator`` — one producer per product in the technosphere.

When AGB has multiple activities producing the same product (a common
artefact of orphan-product relinking, market activities, etc.), the
matrix becomes non-square: one product row but several activity columns
all claiming a positive diagonal in it. The pre-refactor pipeline solved
this with ``MatrixPurger`` (a fixed-point loop that deleted excess
producers from SQLite); this class is the SQL-free equivalent.

Choice rule: keep the producer with the smallest ``output_id``. The
hash-derived ``output_id`` is stable across runs, so the dedup is
deterministic and re-runnable. Discarded activities take their non-
production edges with them — no dangling rows.

Runs *after* ``Allocator`` so the synthetic activity ids produced by
multifunctional splits are visible. Multifunctional split itself does
not introduce new product duplicates: each synthetic activity produces
a distinct product."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scoring.exchange_frame import ExchangeFrame

_PRODUCTION_TYPES = frozenset({"production", "generic production"})


@dataclass(frozen=True)
class ProductDeduplicator:
    """Stateless. ``deduplicate(frame)`` returns a frame with at most
    one producer per product; the smallest ``output_id`` wins."""

    def deduplicate(self, frame: ExchangeFrame) -> ExchangeFrame:
        df = frame.df
        prod_mask = df["edge_type"].isin(_PRODUCTION_TYPES)
        production = df[prod_mask]
        if production.empty:
            return frame

        excess = self._find_excess_producers(production)
        if not excess:
            return frame

        out = df[~df["output_id"].isin(excess)].reset_index(drop=True)
        # ``ExchangeFrame.from_long`` keeps the 5 required columns.
        # ``allocation_factor`` is no longer load-bearing past this
        # point (the allocator has already applied it), so dropping it
        # is fine and lets the frame round-trip through the standard
        # constructor.
        return ExchangeFrame.from_long(out)

    @staticmethod
    def _find_excess_producers(production: pd.DataFrame) -> set[int]:
        """Per product (input_id of production rows), keep the min
        output_id; everyone else is excess."""
        canonical = production.groupby("input_id", sort=False)["output_id"].min()
        excess: set[int] = set()
        # ``zip`` iterates the production rows in order; we collect any
        # output_id that isn't the canonical producer of its product.
        for input_id, output_id in zip(
            production["input_id"], production["output_id"], strict=True
        ):
            if output_id != canonical[input_id]:
                excess.add(int(output_id))
        return excess
