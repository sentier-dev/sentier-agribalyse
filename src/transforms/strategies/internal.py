"""Internal-only strategies — no external bw2 dependency.

The remaining bw2io callables the linker still drove through ``ParsedSimaProCsv``:

* ``set_metadata_using_single_functional_exchange`` — set ``name`` /
  ``unit`` / ``reference product`` / ``production amount`` from the
  process's single functional edge.
* ``split_simapro_name_geo`` — split SimaPro-form names like ``foo/CH U``
  into ``name`` + ``location``.
* ``drop_unlinked`` — remove every exchange that lacks an ``input``.
* ``link_iterable_by_fields`` — internal name+unit linking between
  process exchanges and product nodes (the ``processes_to_products``
  pass that ``InternalAgbLinker`` runs).

All four are pure dict transforms.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

_DETOXIFY_PATTERN = re.compile(r"^(?P<name>.+?)/(?P<geo>[A-Za-z]{2,10})(/I)? [SU]$")
_MISSING = "(unknown)"


@dataclass(frozen=True)
class SetMetadataUsingSingleFunctionalExchange:
    """Lift dataset metadata from the single functional exchange when missing."""

    name: str = "set_metadata_using_single_functional_exchange"
    missing_value: str = _MISSING

    _LABELS: tuple[tuple[str, str], ...] = (
        ("name", "name"),
        ("reference product", "name"),
        ("unit", "unit"),
        ("production amount", "amount"),
    )

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            functional = [e for e in ds.get("exchanges", []) if e.get("functional")]
            if len(functional) != 1:
                continue
            f = functional[0]
            for ds_label, exc_label in self._LABELS:
                cur = ds.get(ds_label)
                if not cur or cur == self.missing_value:
                    ds[ds_label] = f.get(exc_label, self.missing_value)
        return data


@dataclass(frozen=True)
class SplitSimaproNameGeo:
    """Split SimaPro names like ``foo/CH U`` into ``name`` + ``location``."""

    name: str = "split_simapro_name_geo"

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            self._maybe_split(ds, dataset_level=True)
            for exc in ds.get("exchanges", []):
                self._maybe_split(exc, dataset_level=False)
        return data

    @staticmethod
    def _maybe_split(obj: dict, *, dataset_level: bool) -> None:
        n = obj.get("name")
        if not isinstance(n, str):
            return
        m = _DETOXIFY_PATTERN.match(n)
        if not m:
            return
        gd = m.groupdict()
        obj["simapro name"] = n
        obj["location"] = gd["geo"].strip()
        obj["name"] = gd["name"].strip()
        if dataset_level:
            obj["reference product"] = obj["name"]


@dataclass(frozen=True)
class DropUnlinkedExchanges:
    """Remove exchanges that have no ``input`` set."""

    name: str = "drop_unlinked"

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            ds["exchanges"] = [e for e in ds.get("exchanges", []) if e.get("input")]
        return data


# ============================================================================
# Internal link_iterable_by_fields equivalent.


_DEFAULT_FIELDS = ("name", "categories", "unit", "reference product", "location")


class ActivityHash:
    """Lifted from ``bw2io.utils.activity_hash`` — pure-string MD5 over fields."""

    @staticmethod
    def of(obj: dict, fields: Iterable[str] | None = None, case_insensitive: bool = True) -> str:
        chosen = list(fields) if fields else list(_DEFAULT_FIELDS)
        parts: list[str] = []
        for f in chosen:
            v = obj.get(f)
            joined = ("".join(v) if v else "") if isinstance(v, (list, tuple)) else (v or "")
            parts.append(joined.lower() if case_insensitive else joined)
        return hashlib.md5("".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LinkIterableByFields:
    """Link unlinked exchanges to nodes in ``other`` (or self) by hashed field tuple.

    Direct lift of ``bw2io.strategies.generic.link_iterable_by_fields``.
    Uses an MD5 hash over the chosen field values; ambiguous keys leave
    the exchange unlinked (raise behaviour replaced with skip — bw2io's
    ``StrategyError`` was the legacy contract; for our internal-only
    use we prefer "leave unlinked" which matches the catalog matchers).
    """

    fields: tuple[str, ...] | None = None
    edge_kinds: tuple[str, ...] | None = None
    this_node_kinds: tuple[str, ...] | None = None
    other_node_kinds: tuple[str, ...] | None = None
    internal: bool = False
    relink: bool = False

    def apply(self, unlinked: list[dict], other: list[dict] | None = None) -> list[dict]:
        if self.internal:
            other = unlinked
        if other is None:
            other = unlinked

        edge_filter = self._edge_filter()
        this_filter = self._this_filter()
        other_filter = self._other_filter()
        chosen_fields = list(self.fields) if self.fields else list(_DEFAULT_FIELDS)

        candidates: dict[str, tuple[str, str]] = {}
        duplicates: set[str] = set()
        for ds in filter(other_filter, other):
            db = ds.get("database")
            code = ds.get("code")
            if db is None or code is None:
                continue
            key = ActivityHash.of(ds, chosen_fields)
            if key in candidates:
                duplicates.add(key)
            else:
                candidates[key] = (db, code)

        for container in filter(this_filter, unlinked):
            for obj in filter(edge_filter, container.get("exchanges", [])):
                key = ActivityHash.of(obj, chosen_fields)
                if key in duplicates:
                    continue
                if key in candidates:
                    obj["input"] = candidates[key]
        return unlinked

    # ------------------------------------------------------------------
    # Filter constructors.

    def _edge_filter(self):
        kinds = set(self.edge_kinds) if self.edge_kinds else None
        if kinds and self.relink:
            return lambda x: x.get("type") in kinds
        if kinds:
            return lambda x: x.get("type") in kinds and not x.get("input")
        if self.relink:
            return lambda x: True
        return lambda x: not x.get("input")

    def _this_filter(self):
        kinds = set(self.this_node_kinds) if self.this_node_kinds else None
        if kinds:
            return lambda x: x.get("type") in kinds
        return lambda x: True

    def _other_filter(self):
        kinds = set(self.other_node_kinds) if self.other_node_kinds else None
        if kinds:
            return lambda x: x.get("type") in kinds
        return lambda x: True
