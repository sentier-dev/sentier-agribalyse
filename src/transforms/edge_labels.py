"""``EdgeLabelCorrector`` — applies ``edge_label_corrections.parquet``."""

from __future__ import annotations

from dataclasses import dataclass

from core.logging import Logging
from registry import MappingRegistry


@dataclass(frozen=True)
class EdgeLabelCorrector:
    registry: MappingRegistry

    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(self, sp) -> dict[str, int]:
        df = self.registry.edge_label_corrections
        if df.empty:
            return {"renamed": 0}

        rename: dict[str, str] = {}
        for r in df.itertuples(index=False):
            if r.source_name and r.target_name and r.source_name not in rename:
                rename[r.source_name] = r.target_name

        n = 0
        for proc in sp.data:
            for exc in proc.get("exchanges", []):
                src = exc.get("name")
                if src and src in rename:
                    exc["name"] = rename[src]
                    n += 1
        self._log.info("transforms.edge_label_corrections", renamed=n, rules=len(rename))
        return {"renamed": n}
