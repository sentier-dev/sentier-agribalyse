"""Run a bw2io strategy and capture exceptions explicitly.

Replaces ``contextlib.suppress(Exception)`` in the legacy linker
(fix 1.g): every failed strategy is recorded with type+message, the run
report surfaces totals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.logging import Logging
from matching.audit import SuppressedStrategyLog


@dataclass(frozen=True)
class StrategyRunner:
    """Apply a strategy callable; on exception, log + record without crashing the pipeline."""

    suppressed_log: SuppressedStrategyLog

    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(
        self, strategy: Callable[..., Any], *args: Any, label: str = "", **kwargs: Any
    ) -> Any:
        """Call ``strategy(*args, **kwargs)``. Returns its result, or ``None`` on failure."""
        name = label or getattr(strategy, "__name__", str(strategy))
        try:
            return strategy(*args, **kwargs)
        except Exception as exc:
            self._log.warning(
                "strategy.suppressed",
                strategy=name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self.suppressed_log.record(name, exc)
            return None
