"""``ParquetAtomicWriter`` — atomic ``DataFrame → parquet`` writes.

Three builders in this codebase emit parquet artifacts as their final
side effect (``BiosphereRegistryBuilder``, ``EfFlowsRegistryBuilder``,
``MethodCfRegistryBuilder``). All three need the same crash-safe
``<path>.partial`` + ``os.replace`` pattern so a SIGKILL mid-write never
leaves a half-finished file at the publish path. Centralising the helper
removes the drift hazard — a tweak to compression or engine lands once.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class ParquetAtomicWriter:
    """Stateless writer. Use ``ParquetAtomicWriter.write(df, target)``."""

    @staticmethod
    def write(df: pd.DataFrame, target: Path) -> None:
        """Write ``df`` to ``target`` atomically.

        ``compression=None`` keeps writes deterministic across pyarrow
        versions — no codec metadata leakage between rebuilds.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        df.to_parquet(partial, index=False, engine="pyarrow", compression=None)
        os.replace(partial, target)
