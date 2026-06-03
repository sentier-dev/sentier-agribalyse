"""Reusable builders for tiny in-memory test artifacts.

Every builder returns a fresh, independent object — no shared state. They
prefer keyword arguments with sensible defaults so a test can write
``make_settings(project_root)`` for the happy path or override fields it
specifically wants to flex.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Paths, Settings
from domain import Bucket, OverrideKind, SourceKind, Tier
from domain.audit import AuditEntry
from matching.audit import AuditLog, DropTallyTracker
from matching.bio_catalog import BioFlowRef, BiosphereCatalog
from registry import MappingRegistry

# ============================================================================
# Settings / Paths


def make_paths(root: Path) -> Paths:
    return Paths(package_root=root)


def make_settings(root: Path, **overrides) -> Settings:
    """A Settings rooted at ``root``. Overrides any field by kwarg."""
    base = dict(
        agribalyse_version="3.2",
        ecoinvent_version="3.9.1",
        ef_version="3.1",
        biosphere_db_name="ecoinvent-3.9.1-biosphere",
        ef_db_name="ef",
        paths=make_paths(root),
    )
    base.update(overrides)
    return Settings(**base)


# ============================================================================
# Registry DataFrames


def make_mappings_biosphere_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Build a mappings_biosphere parquet-shaped DataFrame.

    ``rows`` items are dicts whose keys override the defaults below.
    """
    cols = (
        "source_kind",
        "source_name",
        "source_unit",
        "source_context",
        "source_top_bucket",
        "source_cas",
        "source_formula",
        "target_db",
        "target_code",
        "target_name",
        "target_unit",
        "unit_conversion",
        "priority_tier",
        "provenance",
        "provenance_row",
        "is_unmatchable",
        "notes",
    )
    if not rows:
        return pd.DataFrame(columns=list(cols))

    defaults = {
        "source_kind": SourceKind.AGB_FLOW.value,
        "source_unit": "kg",
        "source_context": [],
        "source_top_bucket": Bucket.AIR.value,
        "source_cas": None,
        "source_formula": None,
        "target_db": "biosphere3",
        "target_code": "code-1",
        "target_name": "",
        "target_unit": "kg",
        "unit_conversion": 1.0,
        "priority_tier": int(Tier.CURATED_TARGETED),
        "provenance": "test",
        "provenance_row": "0",
        "is_unmatchable": False,
        "notes": "",
    }
    records = []
    for r in rows:
        rec = {**defaults, **r}
        if isinstance(rec["priority_tier"], Tier):
            rec["priority_tier"] = int(rec["priority_tier"])
        if isinstance(rec["source_top_bucket"], Bucket):
            rec["source_top_bucket"] = rec["source_top_bucket"].value
        if isinstance(rec["source_kind"], SourceKind):
            rec["source_kind"] = rec["source_kind"].value
        records.append(rec)
    return pd.DataFrame(records, columns=list(cols))


def make_mappings_technosphere_df(rows: list[dict] | None = None) -> pd.DataFrame:
    rows = rows or []
    for r in rows:
        r.setdefault("source_kind", SourceKind.AGB_NODE.value)
    return make_mappings_biosphere_df(rows)


def make_unit_conversions_df(rows: list[dict] | None = None) -> pd.DataFrame:
    cols = ("source_unit", "target_unit", "multiplier", "provenance")
    if not rows:
        return pd.DataFrame(columns=list(cols))
    return pd.DataFrame(
        [{c: r.get(c, "") for c in cols} | {"multiplier": r.get("multiplier", 1.0)} for r in rows],
        columns=list(cols),
    )


def make_unit_aliases_df(rows: list[dict] | None = None) -> pd.DataFrame:
    cols = ("alias", "alias_lower", "canonical", "provenance")
    if not rows:
        return pd.DataFrame(columns=list(cols))
    records = [
        {
            "alias": r["alias"],
            "alias_lower": r["alias"].lower(),
            "canonical": r["canonical"],
            "provenance": r.get("provenance", ""),
        }
        for r in rows
    ]
    return pd.DataFrame(records, columns=list(cols))


def make_unmatchable_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Same shape as mappings_biosphere — the matcher only reads
    ``source_name``, ``source_top_bucket``, ``is_unmatchable``."""
    return make_mappings_biosphere_df(rows or [])


def make_registry(
    settings: Settings,
    *,
    mappings_biosphere: pd.DataFrame | None = None,
    mappings_technosphere: pd.DataFrame | None = None,
    unmatchable: pd.DataFrame | None = None,
    unit_conversions: pd.DataFrame | None = None,
    unit_aliases: pd.DataFrame | None = None,
    deletions: pd.DataFrame | None = None,
    edge_label_corrections: pd.DataFrame | None = None,
    target_index_ef: pd.DataFrame | None = None,
    context_normalisation: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> MappingRegistry:
    """Build a MappingRegistry directly from in-memory DataFrames."""
    empty = pd.DataFrame()
    return MappingRegistry(
        settings=settings,
        mappings_biosphere=mappings_biosphere if mappings_biosphere is not None else empty,
        mappings_technosphere=mappings_technosphere if mappings_technosphere is not None else empty,
        unmatchable=unmatchable if unmatchable is not None else empty,
        unit_conversions=unit_conversions if unit_conversions is not None else empty,
        unit_aliases=unit_aliases if unit_aliases is not None else empty,
        context_normalisation=context_normalisation if context_normalisation is not None else empty,
        deletions=deletions if deletions is not None else empty,
        edge_label_corrections=edge_label_corrections
        if edge_label_corrections is not None
        else empty,
        target_index_ef=target_index_ef if target_index_ef is not None else empty,
        meta=meta or {},
    )


# ============================================================================
# Audit / drops


def make_audit_log(tmp_path: Path) -> AuditLog:
    return AuditLog(output_path=tmp_path / "audit.parquet")


def make_drop_tracker() -> DropTallyTracker:
    return DropTallyTracker()


def make_audit_entry(**overrides) -> AuditEntry:
    base = dict(
        process_name="proc",
        exchange_name="exc",
        exchange_unit="kg",
        exchange_bucket=Bucket.AIR.value,
        kind=OverrideKind.NEW_LINK,
        new_tier=Tier.CURATED_TARGETED,
        new_target_db="biosphere3",
        new_target_code="code-1",
        new_provenance="test",
    )
    base.update(overrides)
    return AuditEntry(**base)


# ============================================================================
# Fake Brightway-like activities


class FakeBwExchange(dict):
    """Dict-shaped exchange. Implements ``.get(...)`` already (it's a dict)."""

    pass


class FakeBwActivity(dict):
    """Dict-shaped Brightway activity."""

    def __init__(self, key=None, **fields):
        super().__init__(**fields)
        if key is not None:
            self["database"], self["code"] = key
        # Bind .id and .key for the scorer.
        self.id = fields.pop("id", abs(hash(self.get("code", "code"))) % (10**9))

    @property
    def key(self):
        return (self.get("database", "db"), self.get("code", "code"))

    def exchanges(self):
        return list(self.get("_exchanges", []))


class FakeBwDatabase:
    """Iterable + ``__contains__`` like ``bw2data.Database``."""

    def __init__(self, name: str, activities: list[FakeBwActivity] | None = None):
        self.name = name
        self._acts = list(activities or [])

    def __iter__(self):
        return iter(self._acts)

    def __len__(self):
        return len(self._acts)

    def write(self, _data):
        return None

    def register(self):
        return None


def make_bio_catalog(flows: list[BioFlowRef] | None = None) -> BiosphereCatalog:
    """Build a BiosphereCatalog directly from in-memory BioFlowRef instances.

    Example::

        catalog = make_bio_catalog([
            BioFlowRef(db="biosphere3", code="abc",
                       name="CO2", unit="kg", bucket=Bucket.AIR),
        ])
    """
    flows = flows or []
    db_names = tuple(sorted({f.db for f in flows}))
    return BiosphereCatalog(db_names=db_names, flows=tuple(flows))


# ============================================================================
# Fake SimaPro importer


class FakeSimaProImporter:
    """Drop-in replacement for ``bw2io.SimaProBlockCSVImporter``.

    Implements the surface used by transform classes: ``data``,
    ``apply_strategy``, ``apply_strategies``, ``match_database``,
    ``randonneur``, ``normalize_labels_to_brightway_standard``,
    ``match_database_against_top_level_context``, etc.
    """

    def __init__(self, data: list[dict] | None = None):
        self.data = data if data is not None else []
        self.applied_strategies: list[Any] = []
        self.match_calls: list[tuple] = []
        self.randonneur_calls: list[tuple] = []
        self.normalized = False
        self.drop_unlinked_calls = 0
        self.write_database_calls = 0

    def apply_strategy(self, fn, *args, **kwargs):
        self.applied_strategies.append(getattr(fn, "__name__", type(fn).__name__))
        # Mirror ``ParsedSimaProCsv.apply_strategy``: actually run the
        # callable on ``self.data`` so tests that exercise lifted
        # in-house strategies through this fake see real mutations.
        result = fn(self.data, *args, **kwargs)
        if result is not None:
            self.data = result
        return None

    def apply_strategies(self):
        self.applied_strategies.append("__chain__")
        return None

    def match_database(self, *args, **kwargs):
        self.match_calls.append((args, kwargs))
        return None

    def match_database_against_top_level_context(self, *args, **kwargs):
        self.match_calls.append(("top", args, kwargs))
        return None

    def match_database_against_only_available_in_given_context_tree(self, *args, **kwargs):
        self.match_calls.append(("ctx", args, kwargs))
        return None

    def randonneur(self, *args, **kwargs):
        self.randonneur_calls.append((args, kwargs))
        return None

    def normalize_labels_to_brightway_standard(self):
        self.normalized = True
        return None

    def drop_unlinked(self, *, i_am_reckless: bool = False) -> None:
        self.drop_unlinked_calls += 1
        # Mirror bw2io semantics: drop every exchange that lacks an
        # ``input`` key. The unit test asserts the count of dropped
        # exchanges flows into the run-report stage payload.
        for ds in self.data:
            ds["exchanges"] = [e for e in ds.get("exchanges", []) if e.get("input")]

    def write_database(self, *args, **kwargs):
        # The Phase L3 contract: nothing in the runtime path should call
        # ``write_database`` anymore. The fake records the call so a test
        # can prove negative.
        self.write_database_calls += 1
        raise AssertionError(
            "FakeSimaProImporter.write_database called — the SQLite tail "
            "must be gone after Phase L3."
        )


def make_dataset(
    name: str,
    *,
    type: str = "process",
    code: str | None = None,
    exchanges: list[dict] | None = None,
    **extra,
) -> dict:
    """Build a SimaPro-importer-shaped process dict."""
    out = {
        "name": name,
        "type": type,
        "exchanges": list(exchanges or []),
    }
    if code is not None:
        out["code"] = code
    out.update(extra)
    return out


def make_exchange(
    *,
    type: str,
    name: str = "exc",
    unit: str = "kg",
    amount: float = 1.0,
    categories: tuple[str, ...] | None = None,
    cas: str | None = None,
    input: tuple[str, str] | None = None,
    **extra,
) -> dict:
    out = {
        "type": type,
        "name": name,
        "unit": unit,
        "amount": amount,
    }
    if categories is not None:
        out["categories"] = tuple(categories)
    if cas is not None:
        out["CAS number"] = cas
    if input is not None:
        out["input"] = input
    out.update(extra)
    return out


# ============================================================================
# Fake LCIA Method


class FakeMethod:
    """Mimics ``bw2data.Method`` — ``.load()``/``.write(...)`` on a list of CFs."""

    def __init__(self, cfs: list[tuple] | None = None):
        self._cfs = list(cfs or [])

    def load(self):
        return list(self._cfs)

    def write(self, cfs):
        self._cfs = list(cfs)
