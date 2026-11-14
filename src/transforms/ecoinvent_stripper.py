"""``EcoinventStripper`` — remove embedded ecoinvent copies from the SimaPro CSV.

The ADEME export of Agribalyse 3.2 embeds ecoinvent datasets as process
blocks whose product name ends in ``… - Copied from Ecoinvent U``. Those
blocks are licensed ecoinvent data and cannot be published. This class
removes them **as text**: the file is split on SimaPro's ``Process`` /
``End`` block boundaries and every other byte (header, kept blocks,
trailing sections, line endings, encoding) is left untouched. There is no
parse and re-serialise step, so kept blocks cannot drift.

``… - Adapted from Ecoinvent U`` blocks are ADEME's own modified datasets
and are kept, as in the Agribalyse 4.0 "without ecoinvent" export.

Maintainer-side tool: it needs the full CSV. Users of the public repo get
its output, ``source/AGB32_ecoinvent_free.CSV.xz``.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from core.logging import Logging


class EcoinventLeftoverError(RuntimeError):
    """A process that must be removed is still present after the strip."""


@dataclass(frozen=True)
class StrippedProcess:
    """One removed block. No name and no amount: identifier, type, hash."""

    identifier: str
    process_type: str
    sha256: str


@dataclass(frozen=True)
class StripResult:
    stripped: bytes
    removed: tuple[StrippedProcess, ...]
    processes_total: int
    unit_process_names: tuple[str, ...] = ()
    """Product names of the removed blocks of type ``Unit process``."""

    @property
    def processes_kept(self) -> int:
        return self.processes_total - len(self.removed)


@dataclass(frozen=True)
class EcoinventStripper:
    """Strip every process block whose product name carries ``marker``.

    ``required_identifiers`` is a cross-check, not a selector: the strip
    fails when one of them survives (the registry's deletion list of
    aggregated ecoinvent processes is passed here by the CLI).
    """

    marker: str = "Copied from Ecoinvent"
    encoding: str = "latin-1"
    required_identifiers: frozenset[str] = frozenset()

    XZ_PRESET: ClassVar[int] = 6
    UNIT_PROCESS: ClassVar[str] = "Unit process"

    # A block runs from a ``Process`` line to the next ``End`` line. The
    # blank lines that follow belong to it, so removing a block leaves the
    # spacing between its neighbours exactly as SimaPro wrote it.
    _BLOCK: ClassVar[re.Pattern[bytes]] = re.compile(
        rb"^Process\r?\n.*?^End\r?\n(?:\r?\n)*", re.MULTILINE | re.DOTALL
    )
    _END: ClassVar[re.Pattern[bytes]] = re.compile(rb"^End\r?\n", re.MULTILINE)
    _IDENTIFIER: ClassVar[re.Pattern[bytes]] = re.compile(
        rb"^Process identifier\r?\n([^\r\n]*)", re.MULTILINE
    )
    _TYPE: ClassVar[re.Pattern[bytes]] = re.compile(rb"^Type\r?\n([^\r\n]*)", re.MULTILINE)
    _PRODUCTS: ClassVar[re.Pattern[bytes]] = re.compile(
        rb"^(?:Products|Waste treatment)\r?\n((?:[^\r\n]+\r?\n)+)", re.MULTILINE
    )

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ── public API ───────────────────────────────────────────────────────

    def strip(self, raw: bytes) -> StripResult:
        kept: list[bytes] = []
        removed: list[StrippedProcess] = []
        unit_names: list[str] = []
        total = 0
        cursor = 0
        for match in self._BLOCK.finditer(raw):
            total += 1
            block = match.group(0)
            if not self._is_marked(block):
                continue
            kept.append(raw[cursor : match.start()])
            cursor = match.end()
            process_type = self._field(self._TYPE, block)
            removed.append(
                StrippedProcess(
                    identifier=self._field(self._IDENTIFIER, block),
                    process_type=process_type,
                    sha256=hashlib.sha256(self._body(block)).hexdigest(),
                )
            )
            if process_type == self.UNIT_PROCESS:
                unit_names += self._product_names(block)
        kept.append(raw[cursor:])
        stripped = b"".join(kept)
        self._check(stripped)
        return StripResult(
            stripped=stripped,
            removed=tuple(removed),
            processes_total=total,
            unit_process_names=tuple(sorted(set(unit_names))),
        )

    def leftovers(self, raw: bytes) -> tuple[str, ...]:
        """Identifiers of the marked blocks still present in ``raw``."""
        return tuple(
            self._field(self._IDENTIFIER, m.group(0))
            for m in self._BLOCK.finditer(raw)
            if self._is_marked(m.group(0))
        )

    def write(
        self,
        *,
        full_csv: Path,
        stripped_xz: Path,
        manifest_json: Path,
        unit_processes_json: Path | None = None,
    ) -> StripResult:
        raw = Path(full_csv).read_bytes()
        result = self.strip(raw)
        if unit_processes_json is not None:
            unit_processes_json.parent.mkdir(parents=True, exist_ok=True)
            unit_processes_json.write_text(
                json.dumps(self._unit_processes(result), indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        stripped_xz.parent.mkdir(parents=True, exist_ok=True)
        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        stripped_xz.write_bytes(
            lzma.compress(result.stripped, format=lzma.FORMAT_XZ, preset=self.XZ_PRESET)
        )
        manifest_json.write_text(
            json.dumps(self._manifest(raw, result), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._log.info(
            "ecoinvent_strip.written",
            removed=len(result.removed),
            kept=result.processes_kept,
            stripped_mb=round(len(result.stripped) / 1e6, 1),
            xz_mb=round(stripped_xz.stat().st_size / 1e6, 2),
        )
        return result

    # ── internals ────────────────────────────────────────────────────────

    def _is_marked(self, block: bytes) -> bool:
        needle = self.marker.lower()
        return any(needle in name.lower() for name in self._product_names(block))

    def _product_names(self, block: bytes) -> list[str]:
        products = self._PRODUCTS.search(block)
        if products is None:
            return []
        lines = products.group(1).decode(self.encoding).splitlines()
        return [line.split(";", 1)[0] for line in lines]

    def _unit_processes(self, result: StripResult) -> dict[str, Any]:
        return {
            "name": "agribalyse-3.2-copied-unit-processes",
            "description": (
                "SimaPro names of the 'Copied from Ecoinvent' blocks of type 'Unit process' "
                "removed from the ADEME export. Names only, no exchange and no amount. "
                "Inside the Agribalyse library these unit processes draw ADEME's 'Adapted "
                "from Ecoinvent' datasets; the pipeline rebuilds them from the user's own "
                "ecoinvent snapshot so the adaptations stay connected."
            ),
            "names": list(result.unit_process_names),
        }

    def _check(self, stripped: bytes) -> None:
        left = self.leftovers(stripped)
        if left:
            raise EcoinventLeftoverError(
                f"{len(left)} block(s) still carry the marker {self.marker!r}: {left[:5]}"
            )
        if not self.required_identifiers:
            return
        present = {
            self._field(self._IDENTIFIER, m.group(0)) for m in self._BLOCK.finditer(stripped)
        }
        survivors = sorted(self.required_identifiers & present)
        if survivors:
            raise EcoinventLeftoverError(
                f"{len(survivors)} process(es) on the deletion list do not carry the "
                f"marker {self.marker!r} and were not removed: {survivors[:5]}"
            )

    def _manifest(self, raw: bytes, result: StripResult) -> dict[str, Any]:
        return {
            "name": "agribalyse-3.2-stripped-ecoinvent-processes",
            "description": (
                "Process blocks removed from the ADEME SimaPro export of Agribalyse 3.2 "
                "to produce the ecoinvent-free CSV. One row per removed block: SimaPro "
                "process identifier, process type and the sha256 of the removed block. "
                "No names and no amounts. Anyone holding the full export can re-run "
                "dds-strip-ecoinvent and compare."
            ),
            "marker": self.marker,
            "full_csv": self._fingerprint(raw),
            "stripped_csv": self._fingerprint(result.stripped),
            "processes": {
                "total": result.processes_total,
                "kept": result.processes_kept,
                "removed": len(result.removed),
            },
            "removed": [
                {"identifier": r.identifier, "type": r.process_type, "sha256": r.sha256}
                for r in result.removed
            ],
        }

    @classmethod
    def _body(cls, block: bytes) -> bytes:
        """The block without the blank lines that trail its ``End`` line."""
        ends = [m.end() for m in cls._END.finditer(block)]
        return block[: ends[-1]] if ends else block

    @staticmethod
    def _field(pattern: re.Pattern[bytes], block: bytes) -> str:
        match = pattern.search(block)
        return match.group(1).decode("latin-1").strip() if match else ""

    @staticmethod
    def _fingerprint(data: bytes) -> dict[str, Any]:
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
