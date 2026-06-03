"""Unit tests for ``core.logging`` — Logging, StepTimer, IdleHeartbeat.

Use a structlog-shaped recorder rather than reloading the module — reloading
``core.logging`` resets ``sys.stdout``/``stderr`` and breaks downstream tests
in unrelated modules.
"""

from __future__ import annotations

import time

import pytest


class _Recorder:
    """A minimal structlog-shaped logger that records every emission."""

    def __init__(self, **bound):
        self.bound = dict(bound)
        self.events: list[tuple[str, str, dict]] = []  # (level, name, fields)

    def bind(self, **kw):
        merged = {**self.bound, **kw}
        new = _Recorder(**merged)
        new.events = self.events  # share log
        return new

    def _emit(self, level: str, name: str, **kw):
        self.events.append((level, name, {**self.bound, **kw}))

    def info(self, name, **kw):
        self._emit("info", name, **kw)

    def warning(self, name, **kw):
        self._emit("warning", name, **kw)

    def error(self, name, **kw):
        self._emit("error", name, **kw)

    def exception(self, name, **kw):
        self._emit("exception", name, **kw)


class TestStepTimer:
    def test_emits_start_and_done_events(self):
        from core.logging import StepTimer

        rec = _Recorder()
        with StepTimer(rec, "my_step", extra="payload"):
            pass

        names = [e[1] for e in rec.events]
        assert "step.start" in names
        assert "step.done" in names
        # Bound kwargs propagate to every event.
        for level, _name, fields in rec.events:
            assert fields["extra"] == "payload"
            assert fields["step"] == "my_step"
            assert level in {"info", "error"}

    def test_done_event_includes_elapsed_seconds(self):
        from core.logging import StepTimer

        rec = _Recorder()
        with StepTimer(rec, "my_step"):
            time.sleep(0.001)

        done = next(e for e in rec.events if e[1] == "step.done")
        _, _, fields = done
        assert "elapsed_s" in fields
        assert fields["elapsed_s"] >= 0

    def test_failure_records_step_failed_with_error(self):
        from core.logging import StepTimer

        rec = _Recorder()
        with pytest.raises(RuntimeError), StepTimer(rec, "boom"):
            raise RuntimeError("die")

        levels = [e[0] for e in rec.events]
        names = [e[1] for e in rec.events]
        assert "error" in levels
        assert "step.failed" in names
        failure = next(e for e in rec.events if e[1] == "step.failed")
        assert "die" in str(failure[2].get("error"))


class TestIdleHeartbeat:
    def test_first_tick_is_throttled_when_interval_is_large(self):
        from core.logging import IdleHeartbeat

        rec = _Recorder()
        # last is set at construction time, interval is huge → no emission yet.
        hb = IdleHeartbeat(rec, interval_s=10_000.0, what="loading")
        hb.tick(stage=1)
        hb.tick(stage=2)
        assert rec.events == []

    def test_zero_interval_emits_every_tick(self):
        from core.logging import IdleHeartbeat

        rec = _Recorder()
        hb = IdleHeartbeat(rec, interval_s=0.0, what="loading")
        hb.tick(stage=1)
        time.sleep(0.001)
        hb.tick(stage=2)

        names = [e[1] for e in rec.events]
        assert names.count("heartbeat") >= 1


class TestLoggingFacade:
    def test_get_returns_a_logger_with_info_method(self):
        from core.logging import Logging

        log = Logging.get("test")
        assert hasattr(log, "info")
        assert hasattr(log, "bind")

    def test_get_with_no_args_returns_root_logger(self):
        from core.logging import Logging

        log = Logging.get()
        assert log is not None
