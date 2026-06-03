"""Read randonneur datapackages directly from the installed ``randonneur_data``.

Bundled packages are not vendored into ``source/``; they ship with the
``randonneur_data`` Python package and are read in place via its
``Registry.get_file(label)`` API — which handles case-mapping,
compression (.gz/.lzma) and filename resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import randonneur_data


@dataclass(frozen=True)
class RandonneurDataLoader:
    """Load bundled randonneur datapackages by registry label."""

    @cached_property
    def _registry(self) -> Any:
        return randonneur_data.Registry()

    @cached_property
    def data_dir(self) -> Path:
        return Path(self._registry.data_dir)

    def labels(self) -> list[str]:
        """All registered package labels (capitalisation as authored)."""
        return list(self._registry.keys())

    def metadata(self, label: str) -> dict[str, Any]:
        """Just the registry entry (name, filename, compression, mapping, …)."""
        return dict(self._registry[label])

    def load(self, label: str) -> dict[str, Any]:
        """Return the fully-decoded datapackage dict for ``label``."""
        return self._registry.get_file(label)

    def has(self, label: str) -> bool:
        return label in self._registry
