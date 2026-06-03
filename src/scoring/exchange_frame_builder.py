"""``ExchangeFrameBuilder`` — bridge from SimaPro-shape data to ExchangeFrame.

The linker's output (``SimaProImporter.data``) is a list of dicts with
nested ``exchanges`` arrays, each carrying string ``(database, code)``
references. We project that nested shape into a flat long-form frame
keyed by integer ids — ready for the ``Allocator`` and the matrix
builders.

Id assignment: deterministic. Each ``(database, code)`` string pair is
hashed into a 63-bit integer via SHA-256 truncation. Collisions across
a real database (≤100k activities) are vanishingly improbable, and
this avoids the need to coordinate sequential counters across runs.
The same ``(database, code)`` always maps to the same id, so caches
keyed by content hash remain stable.

This is the Phase 5 building block: with it, ``LinkAllPipeline`` can
emit a ``ScoringPackage`` directly from ``sp.data`` and never touch
``bw2data.Database.process()``."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from scoring.exchange_frame import ExchangeFrame

# Id space: 63 bits keeps every value safely within ``int64`` and well
# above the 32-bit threshold where pandas would silently downcast.
_ID_BITS = 63
_ID_MASK = (1 << _ID_BITS) - 1

_PRODUCTION_TYPES = frozenset({"production", "generic production"})
_BIOSPHERE_TYPES = frozenset({"biosphere"})


@dataclass(frozen=True)
class ExchangeFrameBuilder:
    """Stateless builder. ``from_sp_data(data)`` returns an ``ExchangeFrame``."""

    def from_sp_data(self, data: Iterable[dict]) -> ExchangeFrame:
        """Project a list of activity dicts into a long-form frame.

        Each activity ``ds`` contributes one row per ``ds["exchanges"]``
        entry. The activity's own ``(database, code)`` is the row's
        ``output_id``; the exchange's ``input`` is the ``input_id``.
        Allocation factors are read from ``properties["manual_allocation"]``
        when present, defaulting to 1.0 for single-product activities."""
        return self._frame_from_long(self.long_from_sp_data(data))

    def long_from_sp_data(self, data: Iterable[dict]) -> pd.DataFrame:
        """Public projection helper. Returns the long-form DataFrame —
        same schema as the parquet snapshots — so callers can ``pd.concat``
        AGB rows with ecoinvent rows and rebuild a single ExchangeFrame
        in one go (see ``frame_from_long``).
        """
        rows: list[dict] = []
        for ds in data:
            output_key = (ds.get("database") or "", ds.get("code") or "")
            output_id = self.flow_id_for(output_key)
            for exc in ds.get("exchanges", ()) or ():
                input_key = exc.get("input")
                if input_key is None:
                    # Unlinked exchange — skip. The linker is supposed
                    # to have stripped these via ``drop_unlinked`` before
                    # we get here; if any survive, scoring would be
                    # garbage.
                    continue
                input_id = self.flow_id_for(tuple(input_key))
                edge_type = exc.get("type") or "technosphere"
                amount = float(exc.get("amount") or 0.0)
                allocation_factor = float(
                    (exc.get("properties") or {}).get("manual_allocation", 1.0)
                )
                is_biosphere = edge_type in _BIOSPHERE_TYPES
                rows.append(
                    {
                        "output_id": output_id,
                        "input_id": input_id,
                        "amount": amount,
                        "edge_type": edge_type,
                        "is_biosphere": is_biosphere,
                        "allocation_factor": allocation_factor,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    "output_id",
                    "input_id",
                    "amount",
                    "edge_type",
                    "is_biosphere",
                    "allocation_factor",
                ]
            )
        return df

    def frame_from_long(self, df: pd.DataFrame) -> ExchangeFrame:
        """Public alias for ``_frame_from_long`` — preserves
        ``allocation_factor`` through ``ExchangeFrame.from_long``'s
        column-stripping. Use when concatenating multiple long-form
        sources (AGB sp.data + ecoinvent parquet snapshot)."""
        return self._frame_from_long(df)

    @staticmethod
    def _frame_from_long(df: pd.DataFrame) -> ExchangeFrame:
        # ``ExchangeFrame.from_long`` strips extras to the required
        # subset; we then re-attach allocation_factor so the allocator
        # can pick it up.
        frame = ExchangeFrame.from_long(df)
        out = frame.df.copy()
        if "allocation_factor" in df.columns:
            out["allocation_factor"] = df["allocation_factor"].astype("float64").to_numpy()
        return ExchangeFrame(df=out)

    @classmethod
    def flow_id_for(cls, key: tuple[str, str]) -> int:
        """SHA-256 → 63-bit integer for a ``(database, code)`` key.

        This is the canonical key→id hash for the entire SQL-free data
        plane: every consumer that needs to join an external flow table
        (method CFs, product catalog, etc.) against the technosphere /
        biosphere row maps must compute its integer ids through this
        classmethod. Sharing the function guarantees the integer space
        stays consistent — the alternative is duplicated hash logic
        drifting silently.

        Stable across runs and platforms: SHA-256 + truncation to 63 bits
        keeps every value safely within ``int64``.
        """
        h = hashlib.sha256(f"{key[0]}\x00{key[1]}".encode()).digest()
        return int.from_bytes(h[:8], "big") & _ID_MASK
