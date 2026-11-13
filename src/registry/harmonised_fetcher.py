"""``HarmonisedFlowsFetcher`` — download the pinned Brightway Flows release.

``source/harmonised-flows-simple.json.gz`` is the consensus elementary
flow list published by ``brightway-labs/brightway-flows`` on Zenodo
(ODbL). It is 46 MB, above the repository's file-size guard, so it is
fetched once per clone and verified against a pinned sha256 instead of
being committed.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from config import Settings
from core.logging import Logging


class HarmonisedFlowsMismatchError(RuntimeError):
    """A flow list is on disk (or was downloaded) with an unexpected sha256."""


@dataclass(frozen=True)
class HarmonisedFlowsRelease:
    """The pinned release. Bump all four fields together."""

    version: str = "1.1.0"
    doi: str = "10.5281/zenodo.23554066"
    url: str = (
        "https://zenodo.org/api/records/23554066/files/harmonised-flows-simple.json.gz/content"
    )
    sha256: str = "47850016a251ca00d4f806f305293eaa735e3cb59219ea2a27ef7ceb12028271"


@dataclass(frozen=True)
class HarmonisedFlowsFetcher:
    settings: Settings
    release: HarmonisedFlowsRelease = field(default_factory=HarmonisedFlowsRelease)
    opener: Callable[..., Any] = urllib.request.urlopen

    PRESENT: ClassVar[str] = "present"
    DOWNLOADED: ClassVar[str] = "downloaded"
    USER_AGENT: ClassVar[str] = "sentier-agribalyse"
    CHUNK: ClassVar[int] = 1 << 20

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    @property
    def path(self) -> Path:
        return self.settings.paths.harmonised_flows_gz

    def is_current(self) -> bool:
        return self.path.exists() and self._sha256(self.path) == self.release.sha256

    def fetch(self, *, force: bool = False) -> str:
        """Ensure the pinned release is on disk. Returns ``present`` or ``downloaded``."""
        if self.path.exists():
            found = self._sha256(self.path)
            if found == self.release.sha256:
                return self.PRESENT
            if not force:
                raise HarmonisedFlowsMismatchError(
                    f"{self.path} is not Brightway Flows {self.release.version} "
                    f"(sha256 {found[:12]}…, expected {self.release.sha256[:12]}…). "
                    f"Rerun with --force to replace it."
                )
        self._download()
        return self.DOWNLOADED

    def _download(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(self.path.suffix + ".partial")
        request = urllib.request.Request(self.release.url, headers={"User-Agent": self.USER_AGENT})
        try:
            with self.opener(request) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, self.CHUNK)
            found = self._sha256(partial)
            if found != self.release.sha256:
                raise HarmonisedFlowsMismatchError(
                    f"Download from {self.release.url} has sha256 {found}, expected "
                    f"{self.release.sha256}. Nothing was written."
                )
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)
        self._log.info(
            "harmonised_flows.downloaded",
            version=self.release.version,
            mb=round(self.path.stat().st_size / 1e6, 1),
        )

    @classmethod
    def _sha256(cls, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(cls.CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()
