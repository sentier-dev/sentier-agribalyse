"""``Allocator`` — split multifunctional activities into single-product
synthetic activities, deterministically and without bw2data.

What it replaces:

    bw2data.Database.process()  →  multifunctional.allocation strategies
                                →  DELETE FROM exchangedataset; INSERT ...

That code path is the single biggest reason scoring workers serialise
on the SQLite write lock. Brightway re-runs allocation on every read of
the technosphere matrix unless the on-disk processed zip is hot — and
nothing in the linker pipeline guarantees that the zip stays hot. This
class moves allocation upstream into a pure DataFrame transform so the
zip can be content-addressable and read-only.

Strategy: SimaPro / AGB activities carry an ``allocation_factor`` per
production edge (written by ``WasteTreatmentDummyFixer`` and friends).
For each multifunctional activity we:

1. Enumerate its production rows and read their factors.
2. Emit one synthetic activity per production row.
3. Scale the activity's consumption + biosphere edges by the
   allocation factor and assign them to the synthetic activity.

Determinism: synthetic activity ids are derived from the parent activity
id and the product id (``hash`` of the pair, taken modulo a large
prime). That means re-running the allocator on the same frame gives
the same ids — important for content-addressable caching of the
downstream ``ScoringPackage``.

Conservation: factors are normalised per parent activity so they sum
to 1.0 before scaling. The input scale is unspecified — SimaPro
``manual_allocation`` is conventionally in per-cent (sum 100), other
upstreams emit fractions (sum 1.0). Both are accepted; a sum of zero
or any NaN is treated as a configuration bug and raises rather than
silently dropping the activity from the technosphere. This mirrors
the legacy ``bw2data`` ``manual_allocation`` strategy, which divided
by the row sum without asserting a particular scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scoring.exchange_frame import ExchangeFrame

# Prime modulus for synthetic activity-id derivation. 2**61 - 1 stays
# inside Python ``int64`` and is large enough that collisions across a
# realistic database (50-100k activities) are vanishingly unlikely.
_SYNTHETIC_ID_MODULUS = 2_305_843_009_213_693_951


@dataclass(frozen=True)
class Allocator:
    """Pure DataFrame allocator. Stateless — every call to ``allocate``
    is independent of prior calls."""

    def allocate(self, frame: ExchangeFrame) -> ExchangeFrame:
        """Split multifunctional activities; return only the frame.

        Thin wrapper over :meth:`allocate_with_provenance` for callers
        that don't need the synthetic-id provenance map."""
        return self.allocate_with_provenance(frame)[0]

    def allocate_with_provenance(
        self, frame: ExchangeFrame
    ) -> tuple[ExchangeFrame, dict[int, tuple[int, int]]]:
        """Allocate and also return the synthetic-activity provenance.

        The provenance maps every synthetic ``output_id`` this call
        minted to the ``(parent_activity_id, product_id)`` pair it was
        derived from. Catalog builders use it to label synthetic columns
        (whose ids are *not* ``flow_id_for((database, code))`` hashes and
        therefore resolve against no source ``(database, code)``)."""
        if "allocation_factor" not in frame.df.columns:
            # Nothing to allocate — every activity is single-product
            # by assumption (or the allocator is not in use yet).
            return frame, {}

        df = frame.df.copy()
        production_mask = df["edge_type"].isin(("production", "generic production"))
        # Drop rows whose owning activity has no production edge — those
        # activities can't sit in a square technosphere column. They appear
        # when ``drop_unlinked`` strips the only production exchange but
        # leaves consumption / biosphere edges behind. Producing a square
        # matrix is part of the Allocator's contract.
        producing = set(df.loc[production_mask, "output_id"].unique())
        if (~df["output_id"].isin(producing)).any():
            df = df[df["output_id"].isin(producing)].reset_index(drop=True)
            production_mask = df["edge_type"].isin(("production", "generic production"))

        production = df[production_mask]
        # Multifunctional ⇔ same output_id appears in ≥2 production rows.
        prod_counts = production.groupby("output_id").size()
        multi_ids = set(prod_counts[prod_counts >= 2].index)
        if not multi_ids:
            out = ExchangeFrame.from_long(df) if len(df) != frame.n_rows else frame
            return out, {}

        self._validate_factors(production, multi_ids)
        # Normalise per-activity so factors sum to 1.0 regardless of input
        # scale (SimaPro emits per-cent, other upstreams emit fractions).
        df = self._normalise_factors(df, production_mask, multi_ids)

        keep_rows: list[pd.DataFrame] = []
        new_rows: list[pd.DataFrame] = []
        provenance: dict[int, tuple[int, int]] = {}

        for activity_id, group in df.groupby("output_id", sort=False):
            if activity_id not in multi_ids:
                keep_rows.append(group)
                continue
            rows, prov = self._split_activity(activity_id, group)
            new_rows.extend(rows)
            provenance.update(prov)

        out = pd.concat([*keep_rows, *new_rows], ignore_index=True)
        return ExchangeFrame.from_long(out), provenance

    # ------------------------------------------------------------------

    def _split_activity(
        self, activity_id: int, rows: pd.DataFrame
    ) -> tuple[list[pd.DataFrame], dict[int, tuple[int, int]]]:
        """Emit one synthetic activity per production row.

        Consumption + biosphere edges are scaled by the allocation
        factor of the corresponding synthetic activity. The activity's
        original ``output_id`` is replaced with a derived synthetic id
        so the technosphere matrix becomes square. Returns the emitted
        row frames and the ``{synthetic_id: (activity_id, product_id)}``
        provenance for the splits it produced.
        """
        production = rows[rows["edge_type"].isin(("production", "generic production"))]
        non_production = rows[~rows["edge_type"].isin(("production", "generic production"))]
        outputs: list[pd.DataFrame] = []
        provenance: dict[int, tuple[int, int]] = {}
        for _, prod_edge in production.iterrows():
            product_id = int(prod_edge["input_id"])
            factor = float(prod_edge["allocation_factor"])
            synthetic_id = self._synthetic_id(activity_id, product_id)
            provenance[synthetic_id] = (int(activity_id), product_id)

            # The single production edge for the synthetic activity:
            # output_id = synthetic, input_id = product_id, amount kept.
            prod_row = prod_edge.copy()
            prod_row["output_id"] = synthetic_id
            outputs.append(pd.DataFrame([prod_row]))

            if non_production.empty or factor == 0.0:
                continue
            scaled = non_production.copy()
            scaled["output_id"] = synthetic_id
            scaled["amount"] = scaled["amount"] * factor
            outputs.append(scaled)
        return outputs, provenance

    @staticmethod
    def _synthetic_id(activity_id: int, product_id: int) -> int:
        # A deterministic mix that's stable across runs and platforms —
        # ``hash()`` is per-process-randomised in CPython, so we use a
        # plain arithmetic mix instead. ``activity_id * P + product_id``
        # taken mod a 61-bit prime gives a uniform distribution and a
        # tight collision probability.
        return (int(activity_id) * 1_000_003 + int(product_id)) % _SYNTHETIC_ID_MODULUS

    @staticmethod
    def _validate_factors(production: pd.DataFrame, multi_ids: set[int]) -> None:
        for activity_id in multi_ids:
            sub = production[production["output_id"] == activity_id]
            factors = sub["allocation_factor"].to_numpy(dtype="float64")
            if np.isnan(factors).any():
                raise ValueError(
                    f"Allocator: activity {activity_id} has a NaN allocation_factor "
                    f"on at least one production edge. Tag every functional edge "
                    f"with a numeric allocation factor (0.0 is fine, NaN is not)."
                )
            if (factors < 0).any():
                raise ValueError(
                    f"Allocator: activity {activity_id} has a negative "
                    f"allocation_factor; factors must be non-negative."
                )
            total = float(factors.sum())
            if total <= 0.0:
                raise ValueError(
                    f"Allocator: activity {activity_id} allocation factors sum to "
                    f"{total:.6f} (must be > 0). Cannot normalise."
                )

    @staticmethod
    def _normalise_factors(
        df: pd.DataFrame,
        production_mask: pd.Series,
        multi_ids: set[int],
    ) -> pd.DataFrame:
        """Rewrite ``allocation_factor`` so factors sum to 1.0 per parent.

        Touches only multifunctional production rows; single-product
        rows and non-production rows are passed through. SimaPro
        ``manual_allocation`` is per-cent (sum 100); upstream packages
        sometimes emit fractions (sum 1.0). Normalising here lets the
        Allocator accept either without round-trip risk."""
        if not multi_ids:
            return df
        prod = df[production_mask].copy()
        sums = prod.groupby("output_id")["allocation_factor"].transform("sum")
        is_multi = prod["output_id"].isin(multi_ids)
        prod.loc[is_multi, "allocation_factor"] = (
            prod.loc[is_multi, "allocation_factor"] / sums.loc[is_multi]
        )
        df = df.copy()
        df.loc[production_mask, "allocation_factor"] = prod["allocation_factor"]
        return df
