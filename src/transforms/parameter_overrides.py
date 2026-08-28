"""``ParameterOverridesStore`` + ``ParameterOverridesApplier``.

``source/parameter_overrides.csv`` is the single source of override truth
(gitignored — it is local what-if state, not canonical AGB data, so the
"same CSV → bit-identical outputs" guarantee holds for the canonical build).
``dds-set-parameter`` writes it; ``dds-reset`` deletes it (opt out with
``--keep-overrides``); the applier folds it into BOTH pipelines on the
fully LINKED graph in ratio mode — in ``LinkAllPipeline`` directly after
the pristine ``LinkedSpCache`` snapshot, and in ``FastRescorePipeline``
after loading that snapshot — so the two paths share the exact same
override semantics. The scoring-package content hash covers the exchange
frame bytes, so overridden and baseline runs land in distinct cached
packages.

File format (``;``-separated, ``.`` decimals, header row):
``parameter_name;scope;value;set_at;comment`` — ``scope`` is a process code
or ``*`` (every process defining the parameter). A process-specific row
beats a ``*`` row for the same parameter.
"""

from __future__ import annotations

import csv
import datetime
import math
import os
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from core.logging import Logging
from transforms.parameter_reevaluator import GLOBAL_SCOPE, ParameterReevaluator

_HEADER: tuple[str, ...] = ("parameter_name", "scope", "value", "set_at", "comment")


@dataclass(frozen=True)
class ParameterOverride:
    """One override row. ``scope`` is a process code or ``*``."""

    parameter_name: str
    scope: str
    value: float
    set_at: str = ""
    comment: str = ""

    def key(self) -> tuple[str, str]:
        return (self.parameter_name.lower(), self.scope)


@dataclass(frozen=True)
class ParameterOverridesStore:
    """Read/write ``source/parameter_overrides.csv``."""

    settings: Settings = field(default_factory=Settings)

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    @property
    def path(self):
        return self.settings.paths.parameter_overrides_csv

    def load(self) -> list[ParameterOverride]:
        if not self.path.exists():
            return []
        overrides: list[ParameterOverride] = []
        with self.path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for lineno, row in enumerate(reader, start=2):
                try:
                    value = float(row["value"])
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite value {row['value']!r}")
                    overrides.append(
                        ParameterOverride(
                            parameter_name=row["parameter_name"].strip(),
                            scope=(row["scope"] or "").strip() or GLOBAL_SCOPE,
                            value=value,
                            set_at=row.get("set_at") or "",
                            comment=row.get("comment") or "",
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    # The file is user-editable what-if state; a hand-edit
                    # must fail with the file and line, not a bare
                    # float()/KeyError deep inside dds-link-all.
                    raise ValueError(
                        f"Malformed override row at {self.path}:{lineno} "
                        f"({exc}) — expected "
                        f"'parameter_name;scope;value;set_at;comment'. Fix the "
                        f"row or delete the file (dds-clear-parameters)."
                    ) from exc
        return overrides

    def save(self, overrides: list[ParameterOverride]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: this file is the single source of override truth; a torn
        # write must never truncate it.
        partial = self.path.with_suffix(self.path.suffix + ".partial")
        try:
            with partial.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh, delimiter=";")
                writer.writerow(_HEADER)
                for ov in overrides:
                    writer.writerow(
                        [ov.parameter_name, ov.scope, repr(ov.value), ov.set_at, ov.comment]
                    )
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)
        self._log.info("overrides.saved", path=str(self.path), n=len(overrides))

    def upsert(self, override: ParameterOverride) -> ParameterOverride:
        """Insert or replace the row with the same (name, scope). Returns the
        stamped override actually written."""
        stamped = ParameterOverride(
            parameter_name=override.parameter_name,
            scope=override.scope,
            value=override.value,
            set_at=override.set_at
            or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            comment=override.comment,
        )
        rows = [ov for ov in self.load() if ov.key() != stamped.key()]
        rows.append(stamped)
        self.save(rows)
        return stamped

    def clear(self, name: str | None = None, scope: str | None = None) -> int:
        """Remove matching rows; both filters ``None`` removes everything.

        Returns the number of rows removed. Deletes the file when nothing
        remains, so a cleared state is indistinguishable from pristine.
        """
        rows = self.load()

        def matches(ov: ParameterOverride) -> bool:
            if name is not None and ov.parameter_name.lower() != name.lower():
                return False
            return not (scope is not None and ov.scope != scope)

        kept = [ov for ov in rows if not matches(ov)]
        removed = len(rows) - len(kept)
        if kept:
            self.save(kept)
        elif self.path.exists():
            self.path.unlink()
            self._log.info("overrides.file_removed", path=str(self.path))
        return removed


@dataclass(frozen=True)
class ParameterOverridesApplier:
    """Fold the overrides file into a freshly loaded ``ParsedSimaProCsv``."""

    settings: Settings = field(default_factory=Settings)

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def apply(self, sp: Any) -> dict[str, Any]:
        overrides = ParameterOverridesStore(settings=self.settings).load()
        if not overrides:
            return {"overrides": 0, "changed_exchanges": 0, "processes_affected": 0}
        # Ratio mode even at parse level: baseline evaluations equal the
        # baked amounts (the machinery is identical), so the ratio is exact —
        # and edges bw_simapro_csv post-fixed after evaluation (e.g. zeroed
        # waste-model production rows) keep their fix-up (0 * ratio = 0)
        # instead of being resurrected by an absolute re-evaluation.
        result = ParameterReevaluator(parameters=sp.parameters).ratio_patch(sp.data, overrides)
        sp.data = result.data
        for warning in result.warnings:
            self._log.warning("overrides.no_effect", detail=warning)
        stats = {
            "overrides": len(overrides),
            "changed_exchanges": len(result.changes),
            "processes_affected": len({c.process_code for c in result.changes}),
        }
        self._log.info("overrides.applied", **stats)
        # Loud stdout banner: a lingering override silently taints every
        # subsequent "baseline" build — the operator must SEE it, not have
        # to find a log line.
        print(
            f"*** PARAMETER OVERRIDES ACTIVE: {stats['overrides']} override(s) "
            f"changed {stats['changed_exchanges']} exchange(s) across "
            f"{stats['processes_affected']} process(es). This is NOT a "
            f"baseline build — dds-clear-parameters restores it. ***"
        )
        return stats
