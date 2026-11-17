"""``AgribalyseCsvSource`` — which SimaPro CSV the pipeline reads.

Two variants of the Agribalyse 3.2 export exist:

* ``full`` — ``source/AGB32_final.CSV``, ADEME's export with the embedded
  ecoinvent copies. Local-only, never committed.
* ``ecoinvent-free`` — ``source/AGB32_ecoinvent_free.CSV.xz``, the same
  export without the ``Copied from Ecoinvent`` blocks (see
  ``EcoinventStripper``). Committed, so a clone runs with only an
  ecoinvent licence.

By default (``Settings.agribalyse_csv == "auto"``) the full export wins when
it is present, so results of existing local set-ups do not move. Otherwise
the committed file is decompressed once into ``cache/``. ``dds-link-all
--csv full|ecoinvent-free`` asks for one variant; the full export can then
stay in ``source/`` while the ecoinvent-free one is read.

``stamp()`` names the file a run reads, its size and a hash of its content.
The importer's parse cache and the linked cache record it, so neither
replays a parse of another CSV, or of an older state of the same one. The
decompressed copy records the hash of the archive it came from.

Content, not modification time: a checkout or a copy moves the time of an
unchanged file, which would force a re-parse (about 100 s for the full
export), and a rewrite at the same size within one tick of a coarse clock
keeps it. Hashing the 507 MB full export takes about a second.
"""

from __future__ import annotations

import hashlib
import lzma
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from config import Settings
from core.logging import Logging


class AgribalyseCsvMissingError(FileNotFoundError):
    """Neither variant of the Agribalyse CSV is on disk."""


@dataclass(frozen=True)
class AgribalyseCsvSource:
    settings: Settings

    AUTO: ClassVar[str] = "auto"
    FULL: ClassVar[str] = "full"
    ECOINVENT_FREE: ClassVar[str] = "ecoinvent-free"

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    @property
    def requested(self) -> str:
        request = self.settings.agribalyse_csv
        if request not in (self.AUTO, self.FULL, self.ECOINVENT_FREE):
            raise ValueError(
                f"Unknown Agribalyse CSV {request!r}; choose one of "
                f"{self.AUTO}, {self.FULL}, {self.ECOINVENT_FREE}."
            )
        return request

    @property
    def variant(self) -> str | None:
        """The variant this run reads, or ``None`` when its file is missing."""
        request = self.requested
        files = self._files()
        candidates = (self.FULL, self.ECOINVENT_FREE) if request == self.AUTO else (request,)
        return next((v for v in candidates if files[v].exists()), None)

    def source_file(self) -> Path | None:
        """The file the run is fed from: the full CSV, or the committed archive
        (not its decompressed copy, which is a cache of it)."""
        variant = self.variant
        return self._files()[variant] if variant else None

    def stamp(self) -> tuple[str, str, int, str] | None:
        """``(variant, file name, size, sha256)`` of the file the run reads."""
        source = self.source_file()
        if source is None:
            return None
        return (self.variant, source.name, source.stat().st_size, _sha256(source))

    def path(self) -> Path:
        """The CSV to parse. Decompresses the committed file when needed."""
        paths = self.settings.paths
        variant = self.variant
        if variant == self.FULL:
            return paths.agribalyse_csv
        if variant == self.ECOINVENT_FREE:
            return self._decompressed()
        if self.requested == self.FULL:
            raise AgribalyseCsvMissingError(
                f"The full export was asked for (--csv full), but {paths.agribalyse_csv.name} "
                f"is not in {paths.source}."
            )
        if self.requested == self.ECOINVENT_FREE:
            raise AgribalyseCsvMissingError(
                f"The ecoinvent-free CSV was asked for (--csv ecoinvent-free), but "
                f"{paths.agribalyse_csv_ecoinvent_free_xz.name} (shipped with the repository) "
                f"is not in {paths.source}."
            )
        raise AgribalyseCsvMissingError(
            f"No Agribalyse CSV found: expected {paths.agribalyse_csv_ecoinvent_free_xz.name} "
            f"(shipped with the repository) or {paths.agribalyse_csv.name} (ADEME's full "
            f"export) in {paths.source}."
        )

    def describe(self) -> dict[str, str | None]:
        """Run-report payload: what was asked for, and which variant and file fed the run."""
        source = self.source_file()
        return {
            "requested": self.requested,
            "variant": self.variant,
            "file": source.name if source else None,
        }

    def _files(self) -> dict[str, Path]:
        paths = self.settings.paths
        return {
            self.FULL: paths.agribalyse_csv,
            self.ECOINVENT_FREE: paths.agribalyse_csv_ecoinvent_free_xz,
        }

    def _decompressed(self) -> Path:
        paths = self.settings.paths
        xz = paths.agribalyse_csv_ecoinvent_free_xz
        out = paths.agribalyse_csv_ecoinvent_free_cache
        marker = out.with_name(out.name + ".sha256")
        digest = _sha256(xz)
        if out.exists() and marker.exists() and marker.read_text().strip() == digest:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_suffix(out.suffix + ".partial")
        try:
            with lzma.open(xz, "rb") as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(partial, out)
            marker.write_text(digest + "\n")
        finally:
            partial.unlink(missing_ok=True)
        self._log.info(
            "agribalyse_csv.decompressed", path=out.name, mb=round(out.stat().st_size / 1e6, 1)
        )
        return out


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()
