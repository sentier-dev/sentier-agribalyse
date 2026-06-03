"""Plain-JSON and gzipped-JSON readers."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonReader:
    """Read a UTF-8 JSON file. Stateless; reusable for any path."""

    def read(self, path: Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class GzJsonReader:
    """Read a gzipped UTF-8 JSON file."""

    def read(self, path: Path) -> Any:
        with gzip.open(Path(path), "rt", encoding="utf-8") as f:
            return json.load(f)


@dataclass(frozen=True)
class JsonOrGzJsonReader:
    """Auto-dispatch by suffix: ``.json`` → plain, ``.gz`` → gzipped."""

    plain: JsonReader = field(default_factory=JsonReader)
    gzipped: GzJsonReader = field(default_factory=GzJsonReader)

    def read(self, path: Path) -> Any:
        path = Path(path)
        if path.suffix == ".gz":
            return self.gzipped.read(path)
        return self.plain.read(path)
