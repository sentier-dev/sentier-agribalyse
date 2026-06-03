"""``RunReport`` — collects every stage's stats, writes ``dashboard/run_report.json``."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunReport:
    """Mutable accumulator. Serialise via ``write(path)``."""

    settings_summary: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    drops_by_strategy: dict[str, int] = field(default_factory=dict)
    suppressed_strategies: dict[str, int] = field(default_factory=dict)
    matrix_shape: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def add_stage(self, name: str, payload: Any) -> None:
        self.stages[name] = payload

    def add_coverage(self, snapshot) -> None:
        self.coverage.append(snapshot.as_dict())

    def set_drops(self, totals: dict[str, int]) -> None:
        self.drops_by_strategy = dict(totals)

    def set_suppressed(self, totals: dict[str, int]) -> None:
        self.suppressed_strategies = dict(totals)

    def set_matrix_shape(self, rows: int, cols: int) -> None:
        self.matrix_shape = {
            "rows": rows,
            "cols": cols,
            "deficit": rows - cols,
            "square": rows == cols and rows > 0,
        }

    def write(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False, default=str))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> RunReport:
        """Rehydrate from a previously written ``run_report.json``.

        Used when a downstream stage (``--skip-linking`` end-to-end,
        backtest re-runs) needs to *augment* the existing report rather
        than start from scratch and clobber the scoring_package
        metadata that ``dds-backtest`` and friends read back.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"run_report.json not found at {path}. Run `dds-link-all` first."
            )
        data = json.loads(path.read_text())
        return cls(
            settings_summary=data.get("settings") or {},
            stages=data.get("stages") or {},
            coverage=data.get("coverage") or [],
            drops_by_strategy=data.get("drops_by_strategy") or {},
            suppressed_strategies=data.get("suppressed_strategies") or {},
            matrix_shape=data.get("matrix_shape") or {},
            timestamp=data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "settings": self.settings_summary,
            "stages": self.stages,
            "coverage": self.coverage,
            "drops_by_strategy": self.drops_by_strategy,
            "suppressed_strategies": self.suppressed_strategies,
            "matrix_shape": self.matrix_shape,
        }
