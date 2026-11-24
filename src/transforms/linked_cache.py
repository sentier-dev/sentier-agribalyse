"""``LinkedSpCache`` — pickle the fully linked ``ParsedSimaProCsv``.

Written by ``dds-link-all`` immediately before the parameter-overrides
stage and the scoring-package emit (post-transforms, post-matching), so
the snapshot is always the PRISTINE baseline regardless of what the
overrides store holds. The fast rescore path (``dds-set-parameter
--fast``) loads it, applies the overrides store with the same applier the
full path uses, and replays only the emit stage — skipping the parse,
transforms, and matching that dominate link runtime.

The exchanges still carry their ``formula`` keys here (transforms rescale
``amount`` in place but preserve unknown keys), which is what makes the
ratio patch possible on post-transform amounts.

Size note: this pickles the entire linked graph (~1.2GB); it lives in
the gitignored ``cache/`` and is cleared by ``dds-reset``. Writes are
atomic (temp file + ``os.replace``) so a crash mid-write never destroys
the previous good cache.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, ClassVar

from config import Settings
from core.logging import Logging, StepTimer
from transforms.agribalyse_csv_source import AgribalyseCsvSource


class LinkedCacheMissingError(FileNotFoundError):
    """Raised when the fast path is requested but no linked cache exists."""


class LinkedCacheCorruptError(RuntimeError):
    """Raised when the linked cache exists but cannot be unpickled."""


class LinkedCacheStaleError(RuntimeError):
    """Raised when the linked cache was built from a different source CSV."""


@dataclass(frozen=True)
class LinkedSpCache:
    """Read/write ``cache/linked_cache.pkl``."""

    settings: Settings = field(default_factory=Settings)

    # Bump when the pickled payload's expected shape changes so old caches
    # fail loudly instead of mis-replaying.
    # 2: the exchanges of the processes rebuilt from ecoinvent carry their
    # source (``RebuiltRecipe.ATTRIBUTE``). Replayed, a version-1 cache would
    # write an empty recipe over the package's good one.
    SCHEMA_VERSION: ClassVar[int] = 2

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    @property
    def path(self):
        return self.settings.paths.linked_cache_pkl

    def _csv_stamp(self) -> tuple[str, str, int, str] | None:
        """Stamp of the file this run reads; see ``AgribalyseCsvSource.stamp``."""
        return AgribalyseCsvSource(settings=self.settings).stamp()

    def write(self, sp: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(self.path.suffix + ".partial")
        # The stamp the importer recorded is the file this link parsed; hashing
        # the CSV again here would cost another pass over it and could name a
        # file that changed during the run.
        stamp = getattr(sp, "csv_stamp", None) or self._csv_stamp()
        payload = {"schema": self.SCHEMA_VERSION, "csv_stamp": stamp, "sp": sp}
        with StepTimer(self._log, "linked_cache.write", path=self.path.name):
            try:
                with partial.open("wb") as fh:
                    pickle.dump(payload, fh)
                os.replace(partial, self.path)
            finally:
                partial.unlink(missing_ok=True)
        AgribalyseCsvSource(settings=self.settings).record_request()

    def load(self) -> Any:
        if not self.path.exists():
            raise LinkedCacheMissingError(
                f"{self.path} not found — the fast rescore path needs one full "
                f"dds-link-all run (with current code) to seed the linked cache. "
                f"Run dds-link-all once, or drop --fast."
            )
        with StepTimer(self._log, "linked_cache.load", path=self.path.name):
            try:
                with self.path.open("rb") as fh:
                    payload = pickle.load(fh)
            except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
                raise LinkedCacheCorruptError(
                    f"{self.path} is corrupt or from an incompatible code "
                    f"version ({exc}) — rerun dds-link-all to regenerate it."
                ) from exc
        if not isinstance(payload, dict) or payload.get("schema") != self.SCHEMA_VERSION:
            raise LinkedCacheStaleError(
                f"{self.path} uses an old cache format — rerun dds-link-all to regenerate it."
            )
        # A command that asks for no CSV (``dds-set-parameter --fast``) checks
        # the one the last ``dds-link-all`` asked for (``AgribalyseCsvSource``),
        # so a link with ``--csv ecoinvent-free`` replays while the full export
        # stays in ``source/``.
        expected = self._csv_stamp()
        if payload.get("csv_stamp") != expected:
            raise LinkedCacheStaleError(
                f"{self.path} was built from a different source CSV "
                f"({payload.get('csv_stamp')}; this run reads {expected}: variant, file, "
                f"size, content hash) — rerun dds-link-all to regenerate it."
            )
        return payload["sp"]
