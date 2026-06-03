"""Per-flow CF join: SimaPro CFs ↔ the scoring registry's CFs.

The Stage 1 ``dds-compare-cfs --source raw`` comparison computes stats over
two independently-enumerated substance × compartment exports, which means
the SimaPro and JRC sides have different row sets and the resulting deltas
mostly reflect data-shape asymmetry, not real CF disagreement.

This module joins both sides at the **ecoinvent biosphere flow** level so
each `(method, flow)` pair gets ``(sp_cf, ef_cf)`` side-by-side. The
distribution stats the dashboard renders are then computed over the joined
set, where both sides are at the same granularity and the deltas measure
genuine CF disagreement.

Pipeline::

    registry/method_cfs/<slug>/cfs.parquet  ──┐
        (code → ef_cf)                        │
    registry/biosphere_catalog.parquet  ─────┤  FlowLevelCfJoiner
        (code → name, categories, cas, syns)  │     .join_method()
    registry/context_normalisation.parquet  ─┤
        (SP context ↔ eco context)            │
    cache/simapro-EF31-adapted-cfs.parquet ──┘
        (method × name × comp × sub → sp_cf)
                                              ▼
                  pd.DataFrame[code, name, categories,
                              sp_cf, ef_cf, sp_match_provenance]

Spec: ``docs/superpowers/specs/2026-05-22-cf-stats-flow-level-join-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ef.simapro_cf_table import SimaProEFCfTable

# ---------------------------------------------------------------------------
# Biosphere catalog loader.


@dataclass(frozen=True)
class BiosphereCatalogLoader:
    """Read ``registry/biosphere_catalog.parquet``.

    Each row describes one ecoinvent biosphere flow:
    ``(database, code, name, categories, unit, cas, synonyms)``.
    """

    path: Path

    _COLUMNS: ClassVar[tuple[str, ...]] = (
        "database",
        "code",
        "name",
        "categories",
        "unit",
        "cas",
        "synonyms",
    )

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Biosphere catalog not found: {self.path}")
        return pd.read_parquet(self.path, columns=list(self._COLUMNS))


# ---------------------------------------------------------------------------
# Context normaliser (eco → SimaPro).


@dataclass(frozen=True)
class ContextNormaliser:
    """Map ecoinvent / JRC EF ``categories`` to SimaPro ``(compartment,
    sub_compartment)``.

    Two input formats are handled:

    1. **Ecoinvent** (the common case): 2-element lists like
       ``[air, urban air close to ground]``. The forward rules in
       ``registry/context_normalisation.parquet`` go SimaPro → ecoinvent
       (e.g. ``[Air, high. pop.] → [air, urban air close to ground]``);
       this class inverts them, picking the source whose compartment is
       in the SimaPro adapted-parquet namespace (``Air, Water, Soil,
       Raw``).
    2. **JRC EF** (used for flows in the registry that come from JRC's
       raw parquet, not from ecoinvent's biosphere): 3-element lists like
       ``[Emissions, Emissions to air, Emissions to lower stratosphere
       and upper troposphere]``. Mapped via a small built-in dictionary
       since the forward rule table doesn't cover JRC's longer form.

    The top-level compartment for ecoinvent inputs is mapped via a small
    built-in dictionary (``_SP_COMP_BY_BUCKET``), not via the inverse
    rules — the rules carry the long forms ``Raw materials / Resources
    / Substances`` rather than the abbreviated ``Raw`` the adapted
    parquet uses.
    """

    rules: pd.DataFrame  # cols: source_context (list-like), target_context (list-like)

    _SP_COMP_BY_BUCKET: ClassVar[dict[str, str]] = {
        "air": "Air",
        "water": "Water",
        "soil": "Soil",
        "natural resource": "Raw",
    }
    _SP_COMPARTMENTS: ClassVar[frozenset[str]] = frozenset({"Air", "Water", "Soil", "Raw"})
    _DEFAULT_SUB: ClassVar[str] = "(unspecified)"

    # JRC EF flow categories → SimaPro (compartment, sub_compartment).
    # Covers every distinct ``categories`` tuple observed in
    # ``registry/ef_flows.parquet``. Stack height variants
    # (low/high/very-high) collapse to the closest SP sub-compartment.
    _EF_CATEGORIES_TO_SP_CONTEXT: ClassVar[dict[tuple[str, ...], tuple[str, str]]] = {
        # Air emissions
        (
            "Emissions",
            "Emissions to air",
            "Emissions to urban air close to ground",
        ): ("Air", "high. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to non-urban air close to ground",
        ): ("Air", "low. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to non-urban air or from high stacks",
        ): ("Air", "low. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to urban air low stack",
        ): ("Air", "high. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to urban air high stack",
        ): ("Air", "high. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to urban air very high stack",
        ): ("Air", "high. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to non-urban air low stack",
        ): ("Air", "low. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to non-urban air high stack",
        ): ("Air", "low. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to non-urban air very high stack",
        ): ("Air", "low. pop."),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to lower stratosphere and upper troposphere",
        ): ("Air", "stratosphere + troposphere"),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to air, indoor",
        ): ("Air", "indoor"),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to air, unspecified",
        ): ("Air", "(unspecified)"),
        (
            "Emissions",
            "Emissions to air",
            "Emissions to air, unspecified (long-term)",
        ): ("Air", "low. pop., long-term"),
        # Water emissions
        (
            "Emissions",
            "Emissions to water",
            "Emissions to fresh water",
        ): ("Water", "river"),
        (
            "Emissions",
            "Emissions to water",
            "Emissions to sea water",
        ): ("Water", "ocean"),
        (
            "Emissions",
            "Emissions to water",
            "Emissions to water, unspecified",
        ): ("Water", "(unspecified)"),
        (
            "Emissions",
            "Emissions to water",
            "Emissions to water, unspecified (long-term)",
        ): ("Water", "groundwater, long-term"),
        # Soil emissions
        (
            "Emissions",
            "Emissions to soil",
            "Emissions to agricultural soil",
        ): ("Soil", "agricultural"),
        (
            "Emissions",
            "Emissions to soil",
            "Emissions to non-agricultural soil",
        ): ("Soil", "industrial"),
        (
            "Emissions",
            "Emissions to soil",
            "Emissions to soil, unspecified",
        ): ("Soil", "(unspecified)"),
        # Resources
        (
            "Resources",
            "Resources from ground",
            "Non-renewable element resources from ground",
        ): ("Raw", "(unspecified)"),
        (
            "Resources",
            "Resources from ground",
            "Non-renewable energy resources from ground",
        ): ("Raw", "(unspecified)"),
        (
            "Resources",
            "Resources from water",
            "Renewable material resources from water",
        ): ("Raw", "in water"),
        (
            "Resources",
            "Resources from air",
            "Renewable material resources from air",
        ): ("Raw", "(unspecified)"),
        # Land use (no SP sub-compartment equivalent; map to Raw / unspecified)
        ("Land use", "Land transformation"): ("Raw", "(unspecified)"),
        ("Land use", "Land occupation"): ("Raw", "(unspecified)"),
    }

    _inverse: dict[tuple[str, ...], str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        inverse: dict[tuple[str, ...], str] = {}
        for _, r in self.rules.iterrows():
            src = tuple(str(x) for x in r["source_context"])
            tgt = tuple(str(x).lower() for x in r["target_context"])
            if not src or not tgt:
                continue
            sp_comp = src[0]
            if sp_comp not in self._SP_COMPARTMENTS:
                # Skip rules whose source compartment isn't in the adapted
                # parquet's namespace (e.g. ``Raw materials`` / ``Resources``).
                continue
            sp_sub = src[1] if len(src) > 1 else self._DEFAULT_SUB
            if not sp_sub:
                sp_sub = self._DEFAULT_SUB
            inverse.setdefault(tgt, sp_sub)
        object.__setattr__(self, "_inverse", inverse)

    @classmethod
    def from_path(cls, path: Path) -> ContextNormaliser:
        if not path.exists():
            raise FileNotFoundError(f"Context normalisation rules not found: {path}")
        return cls(rules=pd.read_parquet(path))

    def normalise(self, categories: list[str] | tuple[str, ...]) -> tuple[str, str]:
        if categories is None or len(categories) == 0:
            return ("", self._DEFAULT_SUB)
        # JRC EF flow format: 2- or 3-element list starting with a
        # known top-level marker. Look up the whole tuple directly.
        cats_tuple = tuple(str(c) for c in categories)
        if cats_tuple in self._EF_CATEGORIES_TO_SP_CONTEXT:
            return self._EF_CATEGORIES_TO_SP_CONTEXT[cats_tuple]
        top = str(categories[0]).lower()
        sp_comp = self._SP_COMP_BY_BUCKET.get(top, str(categories[0]).title())
        key = tuple(str(c).lower() for c in categories)
        sp_sub = self._inverse.get(key)
        if sp_sub is None:
            sp_sub = str(categories[1]) if len(categories) > 1 else self._DEFAULT_SUB
        return (sp_comp, sp_sub)


# ---------------------------------------------------------------------------
# SimaPro CF index.


@dataclass(frozen=True)
class SimaProCfUnitHarmoniser:
    """Per-method scale factor that brings an SP CF into the registry's
    implicit unit frame.

    Some EF v3.1 methods quote CFs in a *substance-natural* unit rather
    than the flow-tracked unit. For Water use, JRC quotes CFs as
    ``m3 deprivation / m3 water``; the registry stores those raw JRC
    values. SimaPro, on the other hand, converts each CF to the flow's
    ``flow_unit`` (so ``"Water, fresh"`` with ``flow_unit=m3`` keeps the
    raw ``42.95`` while ``"Water"`` emission with ``flow_unit=kg`` is
    divided to ``-0.042955``). Both express the same physics; comparing
    them apples-to-apples needs the conversion.

    Mapping: ``(registry_method_category, sp_flow_unit) → scale``.
    Multiplying SP's CF by ``scale`` brings it into the registry's frame.
    Missing entries default to ``1.0`` (no conversion needed — common
    case for kg-tracked methods like Climate change).
    """

    _SCALES: ClassVar[dict[tuple[str, str], float]] = {
        # JRC water-use CFs are m3 deprivation per m3 water; ecoinvent
        # water flows are in kg. SP's per-kg variant is /1000 of the
        # per-m3 form, so multiply by 1000 to compare against the registry.
        ("water use", "kg"): 1000.0,
        ("water use", "m3"): 1.0,
    }

    @classmethod
    def scale_for(cls, registry_method_category: str, sp_flow_unit: str) -> float:
        return cls._SCALES.get((registry_method_category, sp_flow_unit), 1.0)


@dataclass(frozen=True)
class SimaProCfIndex:
    """O(1) lookup of SimaPro CFs keyed on registry method + flow identity.

    The SP adapted parquet uses SimaPro method names (e.g.
    ``"Ozone depletion"`` / ``"Particulate matter"`` plus
    ``- inorganics`` / ``- organics`` variants). The index converts each
    row's method to the registry ``(category, indicator)`` tuple via
    :data:`SimaProEFCfTable.METHOD_TO_OUR_KEY` so callers can query with
    the registry-side method key directly. Inorganics/organics sub-methods
    fold into their root.

    Each indexed CF is harmonised at index-build time via
    :class:`SimaProCfUnitHarmoniser` so the stored value is directly
    comparable to the registry's CF for the same flow (e.g. Water-use
    SP CFs with ``flow_unit=kg`` are multiplied by 1000 to match JRC's
    per-m3 frame).

    Four lookup tiers, tried in order, each tier falling back from exact
    ``(compartment, sub_compartment)`` to ``(compartment, "(unspecified)")``
    so an ecoinvent flow whose sub-compartment isn't enumerated in SP
    still finds its substance-level CF.

    1. **Exact** ``(method_key, name, compartment, sub_compartment)`` →
       fall back to ``(name, compartment, "(unspecified)")``.
    2. **Synonym**: the biosphere catalog's ``synonyms`` list, same key
       structure as tier 1.
    3. **CAS**, same sub fallback. When multiple SP rows share
       ``(method, cas, compartment, sub_compartment)`` (e.g. SP has
       ``Uranium`` and ``Uranium, 2291 GJ per kg`` and
       ``Uranium, 451 GJ per kg`` all with the same CAS), the row with
       the **shortest name** wins — that's typically the canonical
       substance, not a unit-specific variant.
    4. **Short-name** (e.g. ``"Halon-2401"`` matches SP's
       ``"Ethane, 1,1,1,2-tetrafluoro-2-bromo-, Halon 2401"``). Built by
       taking the last comma-separated token of each SP name and
       normalising punctuation. Only emitted when the token contains a
       digit — chemical identifiers like ``CFC-115`` / ``Halon 2401`` /
       ``HCFC-141b`` all have digits; generic suffixes like
       ``"unspecified"`` / ``"biotic"`` / ``"brown"`` don't, and matching
       those produced false hits (e.g. ``"Coal, hard, unspecified"``
       would otherwise match SP's ``"Energy, unspecified"``).
    """

    primary: dict[tuple[str, str, str, str, str], float]
    by_cas_sub: dict[tuple[str, str, str, str, str], float]
    by_cas_unspec: dict[tuple[str, str, str, str], float]
    by_short_name: dict[tuple[str, str, str, str, str], float]

    # Provenance strings emitted by ``lookup``.
    PROV_EXACT: ClassVar[str] = "exact_name"
    PROV_SYNONYM: ClassVar[str] = "synonym"
    PROV_CAS: ClassVar[str] = "cas"
    PROV_SHORT_NAME: ClassVar[str] = "short_name"
    PROV_UNMATCHED: ClassVar[str] = "unmatched"

    _SUBMETHOD_SUFFIXES: ClassVar[tuple[str, ...]] = (" - inorganics", " - organics")
    _UNSPECIFIED: ClassVar[str] = "(unspecified)"

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SimaProCfIndex:
        primary: dict[tuple[str, str, str, str, str], float] = {}
        # CAS-keyed indices use a priority tuple to pick the canonical row
        # when SP has multiple rows sharing the same key. Higher priority
        # wins. The priority is (is_unspec, -name_len) for the unspec
        # fallback (prefer the (unspecified) sub-compartment, then the
        # shortest name) and -name_len for the exact-sub index (shortest
        # name only, since the sub is already pinned).
        by_cas_sub: dict[tuple[str, str, str, str, str], float] = {}
        by_cas_sub_priority: dict[tuple[str, str, str, str, str], int] = {}
        by_cas_unspec: dict[tuple[str, str, str, str], float] = {}
        by_cas_unspec_priority: dict[tuple[str, str, str, str], tuple[bool, int]] = {}
        by_short_name: dict[tuple[str, str, str, str, str], float] = {}
        sp_to_registry = SimaProEFCfTable.METHOD_TO_OUR_KEY
        for _, r in df.iterrows():
            sp_method = str(r["simapro_method"])
            root = sp_method
            for suffix in cls._SUBMETHOD_SUFFIXES:
                if root.endswith(suffix):
                    root = root[: -len(suffix)]
                    break
            reg_key = sp_to_registry.get(root)
            if reg_key is None:
                continue
            cat, ind = reg_key
            raw_name = str(r["name"])
            name = raw_name.lower().strip()
            name_len = len(raw_name)
            comp = str(r["compartment"]).strip()
            sub = str(r["sub_compartment"]).strip()
            sp_flow_unit = str(r["flow_unit"]).strip()
            cf = float(r["cf"]) * SimaProCfUnitHarmoniser.scale_for(cat, sp_flow_unit)
            primary[(cat, ind, name, comp, sub)] = cf
            cas = str(r["cas"]).strip() if pd.notna(r["cas"]) else ""
            if cas:
                cas_key = (cat, ind, cas, comp, sub)
                priority = -name_len
                if priority > by_cas_sub_priority.get(cas_key, -(10**9)):
                    by_cas_sub[cas_key] = cf
                    by_cas_sub_priority[cas_key] = priority
                unspec_key = (cat, ind, cas, comp)
                unspec_priority = (sub == cls._UNSPECIFIED, -name_len)
                if unspec_priority > by_cas_unspec_priority.get(unspec_key, (False, -(10**9))):
                    by_cas_unspec[unspec_key] = cf
                    by_cas_unspec_priority[unspec_key] = unspec_priority
            short = cls._short_name(raw_name)
            if short:
                short_key = (cat, ind, short, comp, sub)
                by_short_name[short_key] = cf
        return cls(
            primary=primary,
            by_cas_sub=by_cas_sub,
            by_cas_unspec=by_cas_unspec,
            by_short_name=by_short_name,
        )

    def lookup(
        self,
        *,
        registry_method_key: tuple[str, str, str, str],
        name: str,
        context: tuple[str, str],
        synonyms: list[str] | tuple[str, ...] | None,
        cas: str | None,
    ) -> tuple[float | None, str]:
        cat, ind = registry_method_key[2], registry_method_key[3]
        comp, sub = context
        primary_name = name.lower().strip()
        # Tier 1: exact name + exact sub, then exact name + (unspecified).
        hit = self.primary.get((cat, ind, primary_name, comp, sub))
        if hit is not None:
            return (hit, self.PROV_EXACT)
        if sub != self._UNSPECIFIED:
            hit = self.primary.get((cat, ind, primary_name, comp, self._UNSPECIFIED))
            if hit is not None:
                return (hit, self.PROV_EXACT)
        # Tier 2: synonyms with same sub-compartment fallback.
        for syn in synonyms or ():
            syn_norm = str(syn).lower().strip()
            if not syn_norm or syn_norm == primary_name:
                continue
            hit = self.primary.get((cat, ind, syn_norm, comp, sub))
            if hit is not None:
                return (hit, self.PROV_SYNONYM)
            if sub != self._UNSPECIFIED:
                hit = self.primary.get((cat, ind, syn_norm, comp, self._UNSPECIFIED))
                if hit is not None:
                    return (hit, self.PROV_SYNONYM)
        # Tier 3: CAS, exact sub then (unspecified). Tie-break by shortest
        # name happens at index-build time.
        if cas:
            cas_norm = str(cas).strip()
            hit = self.by_cas_sub.get((cat, ind, cas_norm, comp, sub))
            if hit is not None:
                return (hit, self.PROV_CAS)
            hit = self.by_cas_unspec.get((cat, ind, cas_norm, comp))
            if hit is not None:
                return (hit, self.PROV_CAS)
        # Tier 4: short-name only if it looks like a chemical identifier
        # (must contain a digit; generic suffixes like "unspecified" are
        # filtered out at index-build time).
        short = self._short_name(name)
        if short and self._is_chemical_identifier(short):
            hit = self.by_short_name.get((cat, ind, short, comp, sub))
            if hit is not None:
                return (hit, self.PROV_SHORT_NAME)
            hit = self.by_short_name.get((cat, ind, short, comp, self._UNSPECIFIED))
            if hit is not None:
                return (hit, self.PROV_SHORT_NAME)
        return (None, self.PROV_UNMATCHED)

    @staticmethod
    def _is_chemical_identifier(short: str) -> bool:
        """Heuristic: a useful short-name carries at least one digit.

        Chemical identifiers like ``CFC-115``, ``Halon 2401``,
        ``HCFC-141b`` all contain digits. Generic ecoinvent suffixes —
        ``unspecified``, ``brown``, ``in ground``, ``biotic`` — don't,
        and matching on them produces false hits (e.g. ``"Coal, hard,
        unspecified"`` would match SP's ``"Energy, unspecified"``).
        """
        return any(ch.isdigit() for ch in short)

    @staticmethod
    def _short_name(name: str) -> str:
        """Trailing identifier of an SP / JRC substance name, normalised.

        ``"Ethane, 1,1,1,2-tetrafluoro-2-bromo-, Halon 2401"`` and
        ``"Halon-2401"`` both produce ``"halon 2401"``. Used to bridge
        JRC's terse names (``Halon-2401``) against SP's IUPAC-style long
        names. Returns an empty string if no useful token can be
        extracted.
        """
        if not name:
            return ""
        s = name.strip()
        # Take the trailing comma-separated token (often the trivial
        # identifier like "Halon 2401" or "CFC-115").
        if "," in s:
            tail = s.split(",")[-1].strip()
            if tail:
                s = tail
        # Normalise punctuation: hyphens between alphabetic + digit groups
        # are interchangeable with spaces in chemical identifiers.
        return s.lower().replace("-", " ").strip()


# ---------------------------------------------------------------------------
# Per-method joiner.


@dataclass(frozen=True)
class JoinedFlowFrame:
    """Result of joining one method's registry CFs against SimaPro.

    Mirrors the columns of the per-(method, flow) parquet:
    ``code, name, categories, sp_cf, ef_cf, sp_match_provenance``.
    """

    method_key: tuple[str, str, str, str]
    df: pd.DataFrame

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "code",
        "name",
        "categories",
        "sp_cf",
        "ef_cf",
        "sp_match_provenance",
    )


@dataclass(frozen=True)
class FlowLevelCfJoiner:
    """Join one method's registry CFs against SimaPro at the flow level.

    Stateless once constructed: the catalog, normaliser, SP index, and
    short-name → CAS cross-reference are built once per
    ``dds-compare-cfs`` run and reused across the 19 methods.
    """

    catalog: pd.DataFrame
    normaliser: ContextNormaliser
    sp_index: SimaProCfIndex

    _catalog_by_code: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _cas_by_short_name: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        catalog_by_code: dict[str, dict] = {}
        cas_by_short_name: dict[str, str] = {}
        for _, r in self.catalog.iterrows():
            code = str(r["code"])
            cas = str(r["cas"]).strip() if pd.notna(r["cas"]) else ""
            synonyms = (
                list(r["synonyms"]) if r["synonyms"] is not None and len(r["synonyms"]) > 0 else []
            )
            catalog_by_code.setdefault(
                code,
                {
                    "name": str(r["name"]),
                    "categories": (list(r["categories"]) if r["categories"] is not None else []),
                    "cas": cas,
                    "synonyms": synonyms,
                },
            )
            if cas:
                short = SimaProCfIndex._short_name(str(r["name"]))
                if short:
                    cas_by_short_name.setdefault(short, cas)
        object.__setattr__(self, "_catalog_by_code", catalog_by_code)
        object.__setattr__(self, "_cas_by_short_name", cas_by_short_name)

    def join_method(
        self,
        method_key: tuple[str, str, str, str],
        ef_cfs: pd.DataFrame,
    ) -> JoinedFlowFrame:
        """``ef_cfs`` has cols ``(database, code, amount)``.

        Returns a JoinedFlowFrame whose DataFrame has one row per
        ``(method_key, code)`` pair, with ``sp_cf`` populated when the SP
        index has a hit (and ``sp_match_provenance`` recording how it was
        matched) or ``NaN`` otherwise.
        """
        if ef_cfs.empty:
            empty = pd.DataFrame(columns=list(JoinedFlowFrame.COLUMNS))
            return JoinedFlowFrame(method_key=method_key, df=empty)

        rows: list[dict] = []
        for _, r in ef_cfs.iterrows():
            code = str(r["code"])
            ef_cf = float(r["amount"])
            meta = self._catalog_by_code.get(code)
            if meta is not None:
                name = meta["name"]
                categories = meta["categories"]
                cas = meta["cas"]
                synonyms = meta["synonyms"]
            else:
                name = ""
                categories = []
                cas = ""
                synonyms = []
            # JRC EF flows have no CAS in the catalog. Cross-reference via
            # the short-name index built from the rest of the catalog so
            # we can still hit the SP CAS index.
            if not cas and name:
                short = SimaProCfIndex._short_name(name)
                if short:
                    cas = self._cas_by_short_name.get(short, "")
            sp_context = self.normaliser.normalise(categories)
            sp_cf, prov = self.sp_index.lookup(
                registry_method_key=method_key,
                name=name,
                context=sp_context,
                synonyms=synonyms,
                cas=cas,
            )
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "categories": categories,
                    "sp_cf": sp_cf,
                    "ef_cf": ef_cf,
                    "sp_match_provenance": prov,
                }
            )
        df = pd.DataFrame(rows, columns=list(JoinedFlowFrame.COLUMNS))
        return JoinedFlowFrame(method_key=method_key, df=df)


# ---------------------------------------------------------------------------
# Joined-parquet writer.


@dataclass(frozen=True)
class JoinedFlowParquetWriter:
    """Write all joined per-method frames to a single parquet for future drill-down."""

    out_path: Path

    _METHOD_COLS: ClassVar[tuple[str, ...]] = (
        "method_database",
        "method_ef_version",
        "method_category",
        "method_indicator",
    )

    def write(self, frames: list[JoinedFlowFrame]) -> Path:
        all_rows: list[pd.DataFrame] = []
        for jf in frames:
            if jf.df.empty:
                continue
            chunk = jf.df.copy()
            for col, value in zip(self._METHOD_COLS, jf.method_key, strict=True):
                chunk[col] = value
            all_rows.append(chunk)
        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=list(self._METHOD_COLS) + list(JoinedFlowFrame.COLUMNS))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.out_path, index=False)
        return self.out_path
