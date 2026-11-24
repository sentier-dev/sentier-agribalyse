"""``AgribalyseCsvSource`` — which SimaPro CSV the pipeline reads.

Two variants of the Agribalyse 3.2 export exist:

* ``full`` — ``source/AGB32_final.CSV``, ADEME's export with the embedded
  ecoinvent copies. Local-only, never committed.
* ``ecoinvent-free`` — ``source/AGB32_ecoinvent_free.CSV.xz``, the same
  export without the ``Copied from Ecoinvent`` and ``Adapted from
  Ecoinvent`` blocks (see
  ``EcoinventStripper``). Committed, so a clone runs with only an
  ecoinvent licence.

Under ``auto`` (``dds-link-all``'s default) the full export wins when it is
present, so results of existing local set-ups do not move. Otherwise the
committed file is decompressed once into ``cache/``. ``dds-link-all --csv
full|ecoinvent-free`` asks for one variant; the full export can then stay in
``source/`` while the ecoinvent-free one is read.

Every other command (``dds-set-parameter``, ``dds-list-parameters``,
``dds-build-parameters``, the fast path, ...) asks for no CSV
(``Settings.agribalyse_csv is None``) and follows what the last
``dds-link-all`` asked for. The link records its request in
``cache/agribalyse_csv_request.txt`` when it writes its linked cache
(``record_request``); without a record, ``auto`` applies.

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
from typing import Any

from config import Settings
from core.logging import Logging


class AgribalyseCsvMissingError(FileNotFoundError):
    """Neither variant of the Agribalyse CSV is on disk."""


@dataclass(frozen=True)
class AgribalyseCsvSource:
    settings: Settings

    AUTO, FULL, ECOINVENT_FREE = Settings.AGRIBALYSE_CSV_CHOICES

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    @property
    def requested(self) -> str:
        """``auto``, ``full`` or ``ecoinvent-free``: the request of this run,
        or, when it makes none, the last link's (``auto`` without a record)."""
        request = self.settings.agribalyse_csv
        if request is None:
            return self.last_link_request() or self.AUTO
        return Settings.checked_agribalyse_csv(request)

    @property
    def follows_last_link(self) -> bool:
        return self.settings.agribalyse_csv is None and self.last_link_request() is not None

    def last_link_request(self) -> str | None:
        """What the last ``dds-link-all`` asked for, or ``None`` without a record."""
        record = self.settings.paths.agribalyse_csv_request
        try:
            request = record.read_text().strip()
        except FileNotFoundError:
            return None
        if request not in Settings.AGRIBALYSE_CSV_CHOICES:
            raise ValueError(
                f"{record} names no Agribalyse CSV ({request!r}); rerun dds-link-all "
                f"(it rewrites the file) or delete it."
            )
        return request

    def record_request(self) -> None:
        """Record this run's request for the commands that ask for none.
        Called when a link writes its linked cache."""
        record = self.settings.paths.agribalyse_csv_request
        record.parent.mkdir(parents=True, exist_ok=True)
        partial = record.with_name(record.name + ".partial")
        partial.write_text(self.requested + "\n")
        os.replace(partial, record)

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
        asked = f"--csv {self.requested}"
        if self.follows_last_link:
            asked = f"by the last dds-link-all, which ran with {asked}"
        if self.requested == self.FULL:
            raise AgribalyseCsvMissingError(
                f"The full export was asked for ({asked}), but {paths.agribalyse_csv.name} "
                f"is not in {paths.source}."
            )
        if self.requested == self.ECOINVENT_FREE:
            raise AgribalyseCsvMissingError(
                f"The ecoinvent-free CSV was asked for ({asked}), but "
                f"{paths.agribalyse_csv_ecoinvent_free_xz.name} (shipped with the repository) "
                f"is not in {paths.source}."
            )
        raise AgribalyseCsvMissingError(
            f"No Agribalyse CSV found: expected {paths.agribalyse_csv_ecoinvent_free_xz.name} "
            f"(shipped with the repository) or {paths.agribalyse_csv.name} (ADEME's full "
            f"export) in {paths.source}."
        )

    def describe(self) -> dict[str, str | bool | None]:
        """Run-report payload: what was asked for (and whether that is the last
        link's request), and which variant and file fed the run."""
        source = self.source_file()
        return {
            "requested": self.requested,
            "follows_last_link": self.follows_last_link,
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
