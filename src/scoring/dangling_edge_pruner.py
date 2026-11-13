"""``DanglingEdgePruner`` — drop technosphere edges with no producer.

A consumption edge that references a product no activity produces ends
up as a NaN row index in the technosphere matrix builder; ``bw2calc``
or ``scipy.sparse.linalg.spsolve`` then fail with
``LinAlgError: Factor is exactly singular`` (or worse, NaNs that
silently propagate). The pre-refactor pipeline solved this with
``MatrixPurger._orphan_products`` — a fixed-point loop that deleted
product rows with no positive diagonal. This class is the SQL-free
equivalent: drop the consumption edges whose ``input_id`` isn't in
the production set, log the count, return a clean frame.

The pruner runs *after* ``Allocator`` and ``ProductDeduplicator`` so
it sees the final activity / product universe. It deliberately does
NOT touch biosphere edges — biosphere flows aren't in the
technosphere row map and have their own (rectangular) matrix B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.logging import Logging
from scoring.exchange_frame import ExchangeFrame

_PRODUCTION_TYPES = frozenset({"production", "generic production"})
_TECHNOSPHERE_TYPES = frozenset({"technosphere", "substitution"})


@dataclass(frozen=True)
class DanglingEdgePruner:
    """Stateless. ``prune(frame)`` returns a frame with no dangling
    technosphere edges plus a stats dict summarising what was dropped."""

    def prune(self, frame: ExchangeFrame) -> tuple[ExchangeFrame, dict[str, int]]:
        df = frame.df
        # Drop the self-loops that cancel their own production first — an
        # activity whose ``technosphere`` row consumes as much of its product
        # as its ``production`` row makes would zero its diagonal in matrix A
        # (production +X, tech -X). The pre-refactor bw2data flow never
        # emitted these because ``Database.process`` treated waste-treatment
        # self-substitutions as ``substitution`` edges; after the refactor
        # they survive as plain technosphere rows. Partial self-consumption
        # (ecoinvent's own use and losses) is kept: see ``_drop_self_loops``.
        df, n_self_loops = self._drop_self_loops(df)
        prod_mask = df["edge_type"].isin(_PRODUCTION_TYPES)
        tech_mask = df["edge_type"].isin(_TECHNOSPHERE_TYPES)

        # A product is "really produced" iff at least one of its production
        # rows has a positive amount. ``drop_unlinked`` strips the unlinked
        # half of orphan-product entries but the *consumer* edge can survive
        # with a zero-amount production row left as the sole producer; that
        # leaves a structurally singular column in A — pypardiso pivots
        # through it but produces astronomical supply vectors that drive
        # downstream LCIA scores into the 1e+11 range. Treating those
        # zero-amount producers as "no producer" reproduces the pre-refactor
        # ``MatrixPurger._orphan_products`` walk on a single pass.
        positive_prod = prod_mask & (df["amount"] > 0)
        product_ids = set(df.loc[positive_prod, "input_id"].unique())
        activity_ids = set(df.loc[positive_prod, "output_id"].unique())

        # A technosphere edge is dangling if its input or output isn't in the
        # respective set. Production rows are exempt — they define the sets.
        dangling_input = tech_mask & ~df["input_id"].isin(product_ids)
        dangling_output = tech_mask & ~df["output_id"].isin(activity_ids)
        # Also drop the zero-amount production rows themselves and any other
        # rows owned by activities whose only production was zero — those
        # activities are no-ops in the matrix and would leave empty cols.
        zero_or_empty_producers = set(df.loc[prod_mask, "output_id"].unique()) - activity_ids
        zero_prod_rows = df["output_id"].isin(zero_or_empty_producers)
        dangling = dangling_input | dangling_output | zero_prod_rows

        if not dangling.any():
            if n_self_loops == 0:
                return frame, {
                    "dropped": 0,
                    "missing_inputs": 0,
                    "missing_outputs": 0,
                    "zero_amount_producers": 0,
                    "self_loops": 0,
                }
            # Self-loops were dropped but nothing else; rebuild frame with the
            # cleaned df.
            return ExchangeFrame.from_long(df), {
                "dropped": n_self_loops,
                "missing_inputs": 0,
                "missing_outputs": 0,
                "zero_amount_producers": 0,
                "self_loops": n_self_loops,
            }

        log = Logging.get(__name__)
        n_input = int(dangling_input.sum())
        n_output = int(dangling_output.sum())
        n_zero = int(zero_prod_rows.sum())
        sample = self._sample_first(df.loc[dangling])
        log.warning(
            "scoring.dangling_edges.dropped",
            n_dropped=int(dangling.sum()),
            missing_inputs=n_input,
            missing_outputs=n_output,
            zero_amount_producers=n_zero,
            n_zero_activities=len(zero_or_empty_producers),
            sample=sample,
        )

        out = df.loc[~dangling].reset_index(drop=True)
        return ExchangeFrame.from_long(out), {
            "dropped": int(dangling.sum()) + n_self_loops,
            "missing_inputs": n_input,
            "missing_outputs": n_output,
            "zero_amount_producers": n_zero,
            "self_loops": n_self_loops,
        }

    @staticmethod
    def _drop_self_loops(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop the self-loops that would break the production diagonal.

        A self-loop is a technosphere-side row whose ``(output_id,
        input_id)`` pair also carries a production row: the activity
        consumes a product it produces. Two cases:

        * **Cancelling** — the consumed amount reaches the produced amount
          (or the row is a ``substitution``, the shape waste-treatment
          self-substitutions take after the refactor). Summed into A it
          zeroes or flips the diagonal, so the row is dropped.
        * **Partial** — the activity consumes a share of its own output
          (own use and losses: ecoinvent's electricity markets, pipelines,
          seed kept for the next crop). This is real consumption. The row
          is kept and A nets it on the diagonal (``1 - share``).

        Returns ``(filtered_df, n_dropped)``.
        """
        prod_mask = df["edge_type"].isin(_PRODUCTION_TYPES)
        tech_mask = df["edge_type"].isin(_TECHNOSPHERE_TYPES)
        if not prod_mask.any() or not tech_mask.any():
            return df, 0

        produced = (
            df.loc[prod_mask]
            .groupby(["output_id", "input_id"], sort=False)["amount"]
            .sum()
            .rename("produced")
        )
        tech = df.loc[tech_mask, ["output_id", "input_id", "amount", "edge_type"]].join(
            produced, on=["output_id", "input_id"]
        )
        is_loop = tech["produced"].notna()
        cancels = (tech["edge_type"] == "substitution") | (
            tech["amount"].abs() >= tech["produced"].abs()
        )
        drop_index = tech.index[(is_loop & cancels).to_numpy()]
        if len(drop_index) == 0:
            return df, 0
        return df.drop(index=drop_index).reset_index(drop=True), len(drop_index)

    @staticmethod
    def _sample_first(rows: pd.DataFrame) -> dict[str, Any]:
        if rows.empty:
            return {}
        first = rows.iloc[0]
        return {
            "output_id": int(first["output_id"]),
            "input_id": int(first["input_id"]),
            "amount": float(first["amount"]),
            "edge_type": str(first["edge_type"]),
        }
