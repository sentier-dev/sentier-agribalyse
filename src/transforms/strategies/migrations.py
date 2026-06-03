"""``MigrationStore`` + ``MigrationApplier`` — apply bw2io-style migrations in-house.

bw2io's ``migrate_exchanges`` / ``migrate_datasets`` look up migrations
from ``bw2data``'s SQLite-backed registry. We snapshot the relevant
migration JSON files into ``source/bw2io-data/`` once (see
``REFACTOR_FINAL.md`` Phase F3) and apply them ourselves with a small
hash-based mapping. Pure dict transforms — no bw2data, no bw2io.

Migration JSON shape (lifted from bw2io's data getters):

```json
{
  "fields": ["categories", "type"],
  "data": [
    [[["resource", "in ground"], "biosphere"], {"categories": ["natural resource", "in ground"]}],
    ...
  ]
}
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MigrationStore:
    """Loads + caches migration JSON files from a directory.

    The bw2io snapshot lives at ``source/bw2io-data/``; ``Settings`` does
    not expose a path for it because the directory is just static
    reference data — same status as the AGB CSV.
    """

    directory: Path

    def load(self, name: str) -> dict[str, Any]:
        """Return the migration dict ``{fields, data}``. Cached on disk only."""
        path = self.directory / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"migration {name!r} not found at {path}. "
                f"Snapshot the bw2io migration via REFACTOR_FINAL phase F3."
            )
        return json.loads(path.read_text())


@dataclass(frozen=True)
class MigrationApplier:
    """Apply a stored bw2io-shaped migration to a list of dicts.

    Mirrors the semantics of bw2io's ``migrate_exchanges`` / ``migrate_datasets``:
    build a ``tuple(field-values) → patch`` lookup, walk the input,
    overwrite matching keys with the patch, and special-case
    ``multiplier`` → call ``RescaleExchange.apply``.
    """

    migration: dict[str, Any]

    @classmethod
    def from_store(cls, store: MigrationStore, name: str) -> MigrationApplier:
        return cls(migration=store.load(name))

    # ------------------------------------------------------------------
    # Public application surface — exchanges + datasets.

    def apply_to_exchanges(self, data: list[dict]) -> list[dict]:
        """Apply the migration to every exchange in every dataset."""
        from transforms.strategies.units import RescaleExchange  # local: no top-level cycle risk

        mapping = self._mapping()
        fields = list(self.migration["fields"])
        for ds in data:
            for exc in ds.get("exchanges", []):
                key = self._key_from(exc, fields)
                patch = mapping.get(key)
                if patch is None:
                    continue
                for field, value in patch.items():
                    if field == "multiplier":
                        RescaleExchange.apply(exc, float(value))
                    else:
                        exc[field] = value
        return data

    def apply_to_datasets(self, data: list[dict]) -> list[dict]:
        """Apply the migration to dataset-level fields. ``multiplier`` is ignored."""
        mapping = self._mapping()
        fields = list(self.migration["fields"])
        for ds in data:
            key = self._key_from(ds, fields)
            patch = mapping.get(key)
            if patch is None:
                continue
            for field, value in patch.items():
                if field == "multiplier":
                    # multiplier only meaningful on exchanges
                    continue
                ds[field] = value
        return data

    # ------------------------------------------------------------------
    # Internals.

    def _mapping(self) -> dict[tuple, dict]:
        out: dict[tuple, dict] = {}
        for old, new in self.migration.get("data", []):
            out[self._normalise_key(old)] = dict(new)
        return out

    @staticmethod
    def _normalise_key(values: Any) -> tuple:
        out: list = []
        for v in values:
            # Categories serialise as JSON list; collapse to tuple for hashing.
            if isinstance(v, list):
                out.append(tuple(v))
            else:
                out.append(v)
        return tuple(out)

    @classmethod
    def _key_from(cls, obj: dict, fields: list[str]) -> tuple:
        out: list = []
        for f in fields:
            v = obj.get(f)
            if isinstance(v, list):
                out.append(tuple(v))
            else:
                out.append(v)
        return tuple(out)
