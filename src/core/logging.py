"""Logging facade. Class-based, no module-level configuration helpers."""

from __future__ import annotations

import io
import logging
import sys
import time
from typing import Any

import structlog


class Logging:
    """Central logging configuration. Idempotent.

    Usage::

        Logging.configure(level=logging.INFO)
        log = Logging.get(__name__)
    """

    _configured: bool = False

    @classmethod
    def configure(cls, level: int = logging.INFO, *, line_buffered: bool = True) -> None:
        if line_buffered:
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)
            except (AttributeError, ValueError):
                pass

        logging.basicConfig(
            level=level,
            format="%(message)s",
            stream=sys.stdout,
            force=True,
        )
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="%H:%M:%S"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            cache_logger_on_first_use=True,
        )
        cls._configured = True

    @classmethod
    def get(cls, name: str | None = None) -> Any:
        """Return a bound logger; safe even before ``configure()``."""
        return structlog.get_logger(name)


class StepTimer:
    """Context manager: emits ``step.start`` / ``step.done`` with elapsed seconds.

    Use for long-running steps so silence never looks like a hang::

        with StepTimer(log, "csv.parse", path=csv) as bound:
            bound.info("...")
    """

    def __init__(self, log: Any, name: str, **bind: Any) -> None:
        self._log = log
        self._name = name
        self._bind = bind
        self._t0: float = 0.0
        self._bound: Any = None

    def __enter__(self) -> Any:
        self._bound = self._log.bind(step=self._name, **self._bind)
        self._t0 = time.time()
        self._bound.info("step.start")
        return self._bound

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = round(time.time() - self._t0, 2)
        if exc_type is None:
            self._bound.info("step.done", elapsed_s=elapsed)
        else:
            self._bound.error("step.failed", elapsed_s=elapsed, error=str(exc))


class IdleHeartbeat:
    """Emit a heartbeat every N seconds while waiting on an external job.

    Use as a thread/loop ``while not done: heartbeat.tick()``. Avoids the
    "looks identical to hung" problem during long idle waits.
    """

    def __init__(self, log: Any, interval_s: float = 10.0, **bind: Any) -> None:
        self._log = log.bind(**bind) if bind else log
        self._interval_s = interval_s
        self._last = time.time()

    def tick(self, **fields: Any) -> None:
        now = time.time()
        if now - self._last >= self._interval_s:
            self._log.info("heartbeat", **fields)
            self._last = now
