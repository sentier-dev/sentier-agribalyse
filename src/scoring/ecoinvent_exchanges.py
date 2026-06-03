"""``EcoinventExchangesIngester`` — runtime reader for the ecoinvent snapshot.

Companion to ``cli.snapshot_ecoinvent_exchanges``. Reads
``source/ecoinvent-3.9.1-cutoff-exchanges.parquet`` and projects every
row to the long-form schema the ``ExchangeFrame`` matrix builders
expect — keyed by the same ``flow_id_for`` integer hash as
``ExchangeFrameBuilder.from_sp_data``, so concatenating AGB rows with
ecoinvent rows yields a single coherent technosphere.

Why ingest the full ecoinvent universe and not just the reachable
subset:

* The pre-refactor ``Database.process()`` walker also pulled all
  reachable activities (and the closure of their consumption edges).
  In practice this collapses to "all of ``ecoinvent-3.9.1-cutoff``"
  because every product is consumed somewhere.
* Pre-factorisation cost on a 40 k square A is dominated by fill-in,
  not size; with one big LU cached on the ``NativeLciaScorer``
  instance, every subsequent score is a triangular solve. Selectively
  pruning rows would save a few MB at best and reduce reuse — net
  loss for any backtest with more than a handful of products.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.exchange_frame_builder import ExchangeFrameBuilder

# Edge types the linker uses. ``substitution`` rows carry a negative
# amount and live alongside ``technosphere`` in matrix A; ``biosphere``
# is what pushes a row into matrix B.
_BIOSPHERE_TYPES = frozenset({"biosphere"})


@dataclass(frozen=True)
class EcoinventExchangesIngester:
    """Read the parquet snapshot and emit long-form exchange rows.

    The output is a ``pd.DataFrame`` with the columns the
    ``ExchangeFrame`` constructor needs plus ``allocation_factor`` so
    the downstream ``Allocator`` is a no-op for ecoinvent (every
    activity is single-product).
    """

    path: Path

    def load_long(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Ecoinvent exchanges snapshot not found: {self.path}. "
                f"Run `dds-snapshot-ecoinvent-exchanges --sqlite ...` once "
                f"to materialise it from a bw2data project."
            )
        df = pd.read_parquet(self.path)
        if df.empty:
            return self._empty_frame()
        df = self._normalise_waste_signs(df)
        # Vectorised key→id mapping. ``flow_id_for`` is per-tuple,
        # so we cache distinct keys before zipping back to the rows —
        # 21 k unique activities + 4 k biosphere flows ≈ 25 k hashes
        # rather than 670 k row-by-row calls.
        out_keys = list(zip(df["output_database"], df["output_code"], strict=False))
        in_keys = list(zip(df["input_database"], df["input_code"], strict=False))
        unique = {k: ExchangeFrameBuilder.flow_id_for(k) for k in set(out_keys) | set(in_keys)}
        out_id = np.fromiter((unique[k] for k in out_keys), dtype="int64", count=len(out_keys))
        in_id = np.fromiter((unique[k] for k in in_keys), dtype="int64", count=len(in_keys))
        long = pd.DataFrame(
            {
                "output_id": out_id,
                "input_id": in_id,
                "amount": df["amount"].astype("float64").to_numpy(),
                "edge_type": df["type"].astype("string").to_numpy(),
                "is_biosphere": df["type"].isin(_BIOSPHERE_TYPES).to_numpy(),
                "allocation_factor": np.ones(len(df), dtype="float64"),
            }
        )
        return long

    @staticmethod
    def _normalise_waste_signs(df: pd.DataFrame) -> pd.DataFrame:
        """Flip ecoinvent's "system 0" waste-handling sign convention into the
        standard "production positive, consumption positive" form.

        Ecoinvent's cutoff snapshot stores waste-treatment activities with a
        ``production amount = -1`` and any consumer of the treated product
        (i.e. an activity that produces the waste) with ``technosphere amount
        = -X``. Both halves are signed negatively so the two cancel inside the
        matrix balance. The pre-refactor bw2data path consumed these via
        ``bw_processing`` which handled the flip during matrix build; our
        SQL-free ingester reads the raw amounts, so the matrix builder's
        ``amount > 0`` production filter drops the treatment activities and
        the AGB side (which uses ``technosphere amount = +12_216`` for the
        same waste-product input) is left with a one-sided balance that
        drives downstream LCIA scores in the wrong direction.

        The fix: identify activities whose production amount is negative,
        flip those production rows to positive, and flip any technosphere
        row anywhere in the snapshot whose ``input`` is one of those
        activities' reference products. Other negative-amount technosphere
        rows (typically allocation by-products or substitution edges that
        landed on the wrong edge type) are left untouched.
        """
        amount = df["amount"].astype("float64")
        prod_mask = df["type"].eq("production")
        neg_prod = prod_mask & (amount < 0)
        if not neg_prod.any():
            return df
        neg_prod_idx = pd.MultiIndex.from_arrays(
            [df.loc[neg_prod, "output_database"], df.loc[neg_prod, "output_code"]]
        ).unique()
        tech_idx = pd.MultiIndex.from_arrays([df["input_database"], df["input_code"]])
        tech_to_flip = df["type"].eq("technosphere") & tech_idx.isin(neg_prod_idx)

        new_amount = amount.copy()
        new_amount.loc[neg_prod] = -new_amount.loc[neg_prod]
        new_amount.loc[tech_to_flip] = -new_amount.loc[tech_to_flip]
        out = df.copy()
        out["amount"] = new_amount
        return out

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "output_id": pd.Series([], dtype="int64"),
                "input_id": pd.Series([], dtype="int64"),
                "amount": pd.Series([], dtype="float64"),
                "edge_type": pd.Series([], dtype="string"),
                "is_biosphere": pd.Series([], dtype=bool),
                "allocation_factor": pd.Series([], dtype="float64"),
            }
        )
