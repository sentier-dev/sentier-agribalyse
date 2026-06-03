"""``BiosphereMatcher`` — registry-driven, deterministic, OOP.

Walks the ``MappingRegistry`` tier ladder for every AGB biosphere
exchange. Closes fixes 1.a–1.l and 1.o:

* tier-ordered candidates from the registry (no procedural pyramid);
* CAS-based lookup as a tier-5-equivalent fallback (1.o);
* unit equality enforced; rescale via ``UnitConverter`` or reject (1.e, 1.k);
* fill-only tiers cannot displace existing links (1.l);
* deterministic tie-break: tier ASC, provenance ASC, target_code ASC (1.f);
* every override goes into ``AuditLog``; every drop goes into ``DropTallyTracker``;
* ``--no-llm`` symmetric (1.j) via ``Settings.apply_llm_overrides``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

from config import Settings
from core.logging import Logging
from domain import Bucket, Tier
from matching.audit import AuditLog, DropTallyTracker
from matching.bio_catalog import BioFlowRef, BiosphereCatalog
from matching.regional_suffix import RegionalSuffixParser
from registry import MappingRegistry


@dataclass(frozen=True)
class SubCompartmentNormaliser:
    """Map heterogeneous sub-compartment labels to a small canonical key set.

    Sources mix AGB-style labels (``ocean``, ``low. pop.``, ``river``),
    biosphere3 labels (``surface water``, ``urban air close to ground``),
    and ILCD/EF labels (``Emissions to sea water``, ``Emissions to
    non-urban air or from high stacks``). When a registry candidate has
    multiple target codes covering different sub-compartments — typical
    of the harmonised-flows-simple ingestion which produces N water-bucket
    entries for ``Phosphorus, Total`` (one each for fresh/sea/unspecified)
    — the matcher uses this normaliser to pick the candidate whose target
    sub-compartment matches the exchange's. Without that tie-break ocean
    P silently routes to ``Emissions to water, unspecified`` (CF=1.0 for
    freshwater eutrophication) instead of ``Emissions to sea water``
    (CF=0), inflating sea-cage fish products by ~160× on freshwater
    eutrophication. See FIX_DATA.md § 3.
    """

    _GROUPS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            label.lower(): group
            for group, members in {
                "sea": ("ocean", "sea water", "emissions to sea water", "marine"),
                "fresh": (
                    "fresh water",
                    "river",
                    "surface water",
                    "emissions to fresh water",
                ),
                "ground": (
                    "ground-",
                    "ground water",
                    "groundwater",
                    "emissions to ground water",
                    "ground-, long-term",
                    "emissions to ground water, long-term",
                    "ground water, long-term",
                ),
                "urban_air": (
                    "urban air close to ground",
                    "high. pop.",
                    "high. pop., long-term",
                    "urban",
                ),
                "non_urban_air": (
                    "non-urban air or from high stacks",
                    "low. pop.",
                    "low. pop., long-term",
                    "low population density, long-term",
                ),
                "stratosphere": (
                    "lower stratosphere + upper troposphere",
                    "lower stratosphere and upper troposphere",
                    "stratosphere",
                ),
                "agricultural_soil": (
                    "agricultural",
                    "emissions to agricultural soil",
                ),
                "industrial_soil": ("industrial", "emissions to industrial soil"),
                "forest_soil": ("forest", "forestry"),
                "natural_soil": (
                    "non-agricultural",
                    "emissions to non-agricultural soil",
                ),
                "unspecified": (
                    "",
                    "(unspecified)",
                    "unspecified",
                    "emissions to water, unspecified",
                    "emissions to air, unspecified",
                    "emissions to soil, unspecified",
                ),
            }.items()
            for label in members
        }
    )

    @classmethod
    def group(cls, label: str | None) -> str:
        """Return the canonical group key for a raw sub-compartment label.

        Returns ``"unspecified"`` for empty / missing labels and the empty
        string for unrecognised labels (caller can treat that as "no
        opinion" and fall back to the lex tie-break).
        """
        if not label:
            return "unspecified"
        return cls._GROUPS.get(str(label).strip().lower(), "")


@dataclass(frozen=True)
class BiosphereMatchStats:
    n_total: int = 0
    n_linked: int = 0
    by_tier: dict[Tier, int] = field(default_factory=dict)
    n_unit_rejected: int = 0
    n_ambiguous_skipped: int = 0
    n_unmatchable_recognised: int = 0
    n_regional_suffixed: int = 0


@dataclass
class BiosphereMatcher:
    """Apply registry-driven biosphere linking to a SimaPro importer's exchanges."""

    settings: Settings
    registry: MappingRegistry
    catalog: BiosphereCatalog
    audit: AuditLog
    drops: DropTallyTracker
    regional_parser: RegionalSuffixParser = field(default_factory=RegionalSuffixParser)

    # Databases whose codes are allowed to carry the synthetic ``@<region>``
    # suffix. The EF database stays untouched because its CF rows live in
    # a separate keyed lookup and don't benefit from regionalisation.
    REGIONAL_DB_ALLOWLIST: ClassVar[frozenset[str]] = frozenset(
        {"ecoinvent-3.9.1-biosphere", "biosphere3"}
    )

    # Only water-class targets get the synthetic regional suffix. Other
    # regionally-suffixed AGB flows (``Nitrogen oxides, DE``,
    # ``BOD5, FR``) get characterised on the BASE bio3 UUID via the
    # inherited bw2io snapshot CFs — moving them onto synthetic rows
    # silently zeroes their contribution to PM / acidification /
    # eutrophication. The check runs on the target's catalog name
    # rather than the exchange's because the canonical flow type is
    # what determines whether SimaPro / JRC characterises regional
    # values at all.
    REGIONAL_TARGET_NAME_ALLOWLIST_PREFIXES: ClassVar[tuple[str, ...]] = ("water",)

    @property
    def _log(self):
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Public entrypoint.

    def match(self, sp_data: list[dict]) -> BiosphereMatchStats:
        n_total = 0
        n_linked = 0
        n_unit_rejected = 0
        n_ambiguous = 0
        n_unmatchable = 0
        n_regional_suffixed = 0
        per_tier: dict[Tier, int] = {}

        for proc in sp_data:
            proc_name = proc.get("name", "")
            for exc in proc.get("exchanges", []):
                if exc.get("type") != "biosphere":
                    continue
                n_total += 1

                exc_name = (exc.get("name") or "").strip()
                exc_unit = (exc.get("unit") or "").strip()
                exc_categories = tuple(exc.get("categories") or ())
                exc_bucket = Bucket.from_categories(exc_categories)
                exc_sub = self._extract_sub_compartment(exc_categories)
                exc_cas = (exc.get("CAS number") or exc.get("cas") or "").strip() or None

                outcome = self._best_candidate(
                    proc_name=proc_name,
                    exc_name=exc_name,
                    exc_unit=exc_unit,
                    exc_bucket=exc_bucket,
                    exc_sub=exc_sub,
                    exc_cas=exc_cas,
                    prior_input=exc.get("input"),
                    prior_tier=exc.get("_match_tier"),
                )
                if outcome is None:
                    # The unmatchable list (placeholder Neither + randonneur unlinked-list)
                    # is a FALLBACK signal, not a preempt: the whole point of sentier_agribalyse
                    # is that flows the placeholder couldn't classify get rescued via LLM /
                    # harmonised-flows / etc. Only treat the flow as "known unmatchable" if
                    # no real-mapping tier produced a candidate.
                    if self.registry.unmatchable_index.is_unmatchable(exc_name, exc_bucket):
                        n_unmatchable += 1
                    continue

                tier, target_db, target_code, target_unit, multiplier, provenance = outcome

                # Preserve the country-of-extraction encoded in the source
                # name (``Water, well, CN``) as part of the flow identity:
                # the matched bio3 code is shared across regional variants,
                # so without this step every regional water flow collapses
                # onto a single matrix row keyed by (db, base_code) and
                # loses access to the per-region AWARE deprivation CFs
                # SimaPro carries for AGB's reference method. See
                # docs/superpowers/specs/2026-05-22-regional-flow-mapping-design.md.
                #
                # ``RemoveBiosphereLocationPrefixIfFlowInSameLocation``
                # (the bw2io strategy run earlier in the pipeline)
                # strips the ``, FR`` suffix from ``exc.name`` when the
                # activity's own location equals the flow's region —
                # but it preserves the original spelling under
                # ``exc.get("simapro name")``. We re-read that field
                # first so a French activity emitting ``Water, well, FR``
                # still resolves the regional suffix here.
                regional_source_name = exc.get("simapro name") or exc_name
                _base_name, region = self.regional_parser.parse(str(regional_source_name).strip())
                if region and target_db in self.REGIONAL_DB_ALLOWLIST:
                    target_ref = self.catalog.get(target_db, target_code)
                    target_name_lc = target_ref.name.strip().lower() if target_ref else ""
                    if target_name_lc.startswith(self.REGIONAL_TARGET_NAME_ALLOWLIST_PREFIXES):
                        target_code = f"{target_code}@{region}"
                        n_regional_suffixed += 1

                prior = exc.get("input")
                if prior:
                    self.audit.record_override(
                        process_name=proc_name,
                        exchange_name=exc_name,
                        exchange_unit=exc_unit,
                        exchange_bucket=exc_bucket.value,
                        new_tier=tier,
                        new_target_db=target_db,
                        new_target_code=target_code,
                        new_provenance=provenance,
                        prior_tier=exc.get("_match_tier"),
                        prior_target_db=prior[0] if isinstance(prior, tuple) else "",
                        prior_target_code=prior[1] if isinstance(prior, tuple) else "",
                        unit_conversion=multiplier,
                    )
                else:
                    self.audit.record_new_link(
                        process_name=proc_name,
                        exchange_name=exc_name,
                        exchange_unit=exc_unit,
                        exchange_bucket=exc_bucket.value,
                        new_tier=tier,
                        new_target_db=target_db,
                        new_target_code=target_code,
                        new_provenance=provenance,
                        unit_conversion=multiplier,
                    )

                exc["input"] = (target_db, target_code)
                exc["_match_tier"] = int(tier)
                exc["_match_provenance"] = provenance
                if multiplier != 1.0:
                    exc["amount"] = exc.get("amount", 0) * multiplier
                    exc["unit"] = target_unit or exc_unit

                n_linked += 1
                per_tier[tier] = per_tier.get(tier, 0) + 1

        # Cleanup drops, reported (fix 1.i)
        n_dropped_zero = self._drop_zero_amount_unlinked(sp_data)
        n_dropped_waste = self._drop_final_waste_flows(sp_data)
        if n_dropped_zero:
            self.drops.record("drop_zero_amount_unlinked_biosphere", n_dropped_zero)
        if n_dropped_waste:
            self.drops.record("drop_final_waste_flows", n_dropped_waste)

        return BiosphereMatchStats(
            n_total=n_total,
            n_linked=n_linked,
            by_tier=per_tier,
            n_unit_rejected=n_unit_rejected,
            n_ambiguous_skipped=n_ambiguous,
            n_unmatchable_recognised=n_unmatchable,
            n_regional_suffixed=n_regional_suffixed,
        )

    # ------------------------------------------------------------------
    # Candidate selection.

    def _best_candidate(
        self,
        *,
        proc_name: str,
        exc_name: str,
        exc_unit: str,
        exc_bucket: Bucket,
        exc_sub: str,
        exc_cas: str | None,
        prior_input,
        prior_tier: int | None,
    ) -> tuple[Tier, str, str, str, float, str] | None:
        """Walk tiers ascending. Return the first valid (tier, db, code, unit, mult, prov) tuple."""
        # Combine name+bucket lookup with CAS lookup; merge and re-sort by tier.
        candidates = list(
            self.registry.biosphere_index.lookup("agb_flow", exc_name.lower(), exc_bucket)
        )
        if exc_cas:
            # CAS-based candidates must share the source name with the
            # exchange. The placeholder/curated mappings filed under one CAS
            # are scoped to a specific source (e.g. ``Methane, peat oxidation``
            # / ``Methane, fossil`` both have CAS 74-82-8 and target
            # ``Methane, fossil``); without the name filter, *every* methane
            # variant — biogenic, non-fossil, soil/biomass stock — would
            # inherit the fossil-methane target via the shared CAS, which
            # silently routes their inventory through the climate-change
            # fossil method instead of the biogenic one. The filter restores
            # CAS to its intended role: tie-breaker on top of name-bucket,
            # not a name-substitution shortcut.
            exc_name_lc = exc_name.lower()
            for c in self.registry.biosphere_cas_index.lookup(exc_cas):
                if c.source_name and c.source_name.lower() != exc_name_lc:
                    continue
                candidates.append(c)

        # Sort key:
        #   1. tier ASC          — strict, preserves the priority ladder.
        #   2. provenance ASC    — deterministic.
        #   3. _sub_rank         — within a (tier, provenance) round, prefer
        #                          candidates whose target sub-compartment
        #                          matches the exchange's. The harmonised-flows-simple
        #                          ingestion produces multiple candidates per
        #                          (name, bucket) covering different EF sub-
        #                          compartments (sea / fresh / unspecified for P
        #                          to water); the exchange's sub-compartment is
        #                          the only signal that disambiguates them.
        #   4. target_code ASC   — final lex tie-break.
        exc_sub_norm = (exc_sub or "").strip().lower()
        exc_sub_group = SubCompartmentNormaliser.group(exc_sub)
        candidates.sort(
            key=lambda c: (
                int(c.tier),
                c.provenance,
                self._sub_rank(c, exc_sub_norm, exc_sub_group),
                c.target_code,
                c.target_name,
            )
        )

        for cand in candidates:
            if cand.is_unmatchable:
                # Hit on the unmatchable list → stop looking, leave unlinked.
                return None
            if cand.tier.is_llm_gated() and not self.settings.apply_llm_overrides:
                continue
            if cand.tier.is_fill_only() and prior_input:
                continue

            target_db, target_code, target_unit = self._resolve_target(cand, exc_bucket)
            if not target_db or not target_code:
                continue

            multiplier = self._resolve_candidate_multiplier(
                cand=cand, exc_unit=exc_unit, target_unit=target_unit
            )
            if multiplier is None:
                self.audit.record_unit_mismatch(
                    process_name=proc_name,
                    exchange_name=exc_name,
                    exchange_unit=exc_unit,
                    exchange_bucket=exc_bucket.value,
                    candidate_tier=cand.tier,
                    candidate_target_db=target_db,
                    candidate_target_code=target_code,
                    candidate_provenance=cand.provenance,
                    candidate_unit=target_unit,
                )
                continue

            return (
                cand.tier,
                target_db,
                target_code,
                target_unit,
                multiplier,
                cand.provenance,
            )

        return None

    def _resolve_target(self, cand, exc_bucket: Bucket) -> tuple[str, str, str]:
        """Return ``(db, code, unit)``. Resolves missing codes via the catalog."""
        # Direct hit (registry already carries db+code).
        if cand.target_db and cand.target_code:
            ref = self.catalog.get(cand.target_db, cand.target_code)
            unit = ref.unit if ref is not None else cand.target_unit
            return cand.target_db, cand.target_code, unit

        prefs = self._target_db_prefs(cand)

        # UUID-only hit: ``AGB32FlowmapperBiosphereSource`` and similar
        # ingesters emit ``target_db=""`` because the flowmap JSON only
        # carries ``target.identifier``. The UUID still pins the exact
        # sub-compartment-scoped flow the author chose; walk the preferred
        # DBs and look it up directly. Without this branch, the matcher
        # falls through to name-based lookup which lex-tie-breaks across
        # sub-compartments — collapsing every Chloride emission of a
        # K2SO4 market activity (river / groundwater / long-term / ocean)
        # onto whichever bio3 chloride flow has the alphabetically
        # smallest code (``5e050fab`` = ``[water, surface water]``),
        # mis-characterising 67% of the chloride mass at the freshwater-
        # ecotox CF instead of the long-term groundwater CF=0.
        #
        # CROSS-BUCKET GUARD: refuse if the resolved flow lives in a
        # different top-level compartment than the exchange. The flowmapper
        # ingestion files Bifenazate/Boscalid/Kaolin/etc. under their bio3
        # ``[soil, agricultural]`` UUIDs (the only bucket those substances
        # have in bio3) but AGB also emits them as ``[air, non-urban air
        # or from high stacks]``. Without this guard the air-bucket
        # exchange silently lands on the soil flow — an air-emission of
        # a pesticide is materially different from a soil-emission and
        # would be characterised under the wrong CFs. The reviewer policy
        # is "no cross-compartment matches"; an explicit cross-compartment
        # pin must come through ``curated_overrides`` (which sets
        # ``target_db`` + ``target_code`` and goes through the direct-hit
        # branch above). Targets in the bucket-agnostic ``UNSPECIFIED``
        # compartment are accepted because they are bucket-agnostic by
        # design.
        if cand.target_code:
            for db in prefs:
                ref = self.catalog.get(db, cand.target_code)
                if ref is None:
                    continue
                if ref.bucket != exc_bucket and ref.bucket != Bucket.UNSPECIFIED:
                    continue
                return ref.db, ref.code, ref.unit

        # Resolve by name across the configured biosphere DBs.
        target_name = (cand.target_name or "").strip()
        if not target_name:
            return "", "", ""

        for db in prefs:
            hits = self.catalog.lookup_name_bucket(db, target_name, exc_bucket)
            if hits:
                ref = self._deterministic_pick(hits)
                return ref.db, ref.code, ref.unit
        # UNSPECIFIED-bucket target fallback. SAFE — bio3 / EF flows in the
        # ``unspecified`` compartment are bucket-agnostic by design (generic
        # targets such as "Phosphate"). LCIA computes the impact independently
        # of compartment for these.
        if exc_bucket != Bucket.UNSPECIFIED:
            for db in prefs:
                hits = self.catalog.lookup_name_bucket(db, target_name, Bucket.UNSPECIFIED)
                if hits:
                    ref = self._deterministic_pick(hits)
                    return ref.db, ref.code, ref.unit
        return "", "", ""

    @staticmethod
    def _deterministic_pick(refs: list[BioFlowRef]) -> BioFlowRef:
        """Pick the lexicographically-smallest code (replaces ``[0]`` non-determinism, fix 1.f)."""
        return sorted(refs, key=lambda r: (r.code,))[0]

    def _target_db_prefs(self, cand) -> list[str]:
        """Preference order for catalog lookups: explicit ``target_db`` first
        (if set), then configured biosphere DB, EF, biosphere3.
        Deduplicated, order-preserving."""
        prefs: list[str] = []
        if cand.target_db:
            prefs.append(cand.target_db)
        prefs.extend(
            d
            for d in (
                self.settings.biosphere_db_name,
                self.settings.ef_db_name,
                "biosphere3",
            )
            if d not in prefs
        )
        return prefs

    def _resolve_candidate_ref(self, cand) -> BioFlowRef | None:
        """Return the ``BioFlowRef`` a candidate's ``(target_db, target_code)``
        points at, or ``None`` if neither field resolves a flow.

        Mirrors the lookup branches of ``_resolve_target`` but returns the
        full ``BioFlowRef`` so ``_sub_rank`` can consult its ``categories``
        for the sub-compartment-aware tie-break. UUID-only candidates
        (``target_db=""`` + ``target_code=<uuid>``, the
        ``AGB32FlowmapperBiosphereSource`` shape) walk the same preferred
        DBs as ``_resolve_target`` so rank and resolution agree.
        """
        if not cand.target_code:
            return None
        if cand.target_db:
            return self.catalog.get(cand.target_db, cand.target_code)
        for db in self._target_db_prefs(cand):
            ref = self.catalog.get(db, cand.target_code)
            if ref is not None:
                return ref
        return None

    @staticmethod
    def _extract_sub_compartment(categories: tuple[str, ...]) -> str:
        """Pull the sub-compartment out of a normalised ``categories`` tuple.

        Biosphere3 categories typically have shape ``(top, sub)`` (e.g.
        ``("water", "ocean")``); ILCD/EF categories have three layers
        ``("Emissions", "Emissions to water", "Emissions to sea water")``
        — we want the most specific layer (``categories[-1]``) when
        multiple layers exist, otherwise ``categories[1]``.
        """
        if not categories or len(categories) < 2:
            return ""
        return str(categories[-1])

    def _sub_rank(self, cand, exc_sub_norm: str, exc_sub_group: str) -> int:
        """Rank a candidate's target sub-compartment match against an
        exchange's sub-compartment. Lower ranks sort first.

        4-tier scheme:

        * **0** — exact match (case-insensitive) between the exchange
          sub-compartment and the candidate target's last category. This
          preserves the distinction between sibling labels that
          collapse to the same coarse group, in particular ``ground-``
          vs ``ground-, long-term``: both are "ground" group but
          characterise differently in EF v3.1 (CF=301 vs CF=0 for
          chloride freshwater ecotox). Without this rank, chloride
          emissions to ``groundwater, long-term`` (1.66 kg/kg on K2SO4
          market) drift to the lex-smallest ``ground-`` UUID and
          inflate Sardine ecotox by ~31 CTUe/kg.

          Also returned (rank 0) when the exchange has no
          sub-compartment AND the candidate's target has none either —
          a single-element ``categories`` tuple identifies the bio3 /
          ecoinvent ``[air]`` / ``[water]`` "unspecified" variant.
          Regression for the fish-PM −51% cluster: AGB authors
          ``Diesel combustion in marine engines {FR}`` with 78.5 g of
          ``Nitrogen oxides`` per kg diesel under a bare
          ``Emissions to air`` header. Five tier-4 NOx candidates exist,
          one per sub-compartment variant; without this preference
          every candidate scored rank 3 and the final tie-break became
          ``target_code ASC``, lex-routing 79 g NOx/kg diesel to
          ``[air, lower stratosphere + upper troposphere]`` (CF=2.1E-7)
          instead of ``[air]`` (CF=1.6E-6). 7.6× CF gap × 79 g NOx ≈
          the entire −51% PM gap across 72 wild-fish SKUs.
        * **1** — same coarse group, different exact label. Preferred
          over a different-group target.
        * **2** — different group (e.g. ``ground`` vs ``fresh``).
        * **3** — unknown (no resolvable target ref or no exchange
          sub-comp signal that matches the candidate).

        Resolves the candidate's ``BioFlowRef`` through
        ``_resolve_candidate_ref`` so UUID-only candidates
        (``target_db=""`` + ``target_code=<uuid>``, the
        ``AGB32FlowmapperBiosphereSource`` shape) get ranked by their
        actual flow's compartment.
        """
        ref = self._resolve_candidate_ref(cand)
        if not exc_sub_norm:
            # Empty exchange sub-compartment — prefer the target whose
            # ``categories`` is single-element (bio3 / ecoinvent
            # "unspecified" variant, e.g. ``('air',)``). ``exc_sub_group``
            # is ``'unspecified'`` (not empty) in this branch because
            # ``SubCompartmentNormaliser.group('')`` returns
            # ``'unspecified'`` by design, so we cannot key off it here.
            if ref is not None and ref.categories and len(ref.categories) == 1:
                return 0
            return 3
        if ref is None or not ref.categories:
            return 3
        cand_sub = str(ref.categories[-1] or "").strip().lower()
        if cand_sub == exc_sub_norm:
            return 0
        cand_sub_group = SubCompartmentNormaliser.group(cand_sub)
        if not cand_sub_group or not exc_sub_group:
            return 3
        return 1 if cand_sub_group == exc_sub_group else 2

    def _resolve_unit_multiplier(self, source_unit: str, target_unit: str) -> float | None:
        """Return multiplier or ``None`` if no conversion exists (fix 1.e + 1.k)."""
        if not target_unit:
            return 1.0  # registry row didn't constrain the unit
        return self.registry.unit_converter.multiplier(source_unit, target_unit)

    def _resolve_candidate_multiplier(
        self, *, cand, exc_unit: str, target_unit: str
    ) -> float | None:
        """Pick the source→target multiplier for a candidate.

        Resolution order:

        1. **Global converter on (exc_unit → target_unit) first.** If a
           pure unit-scaling path exists (Bq→kBq, kg→g, m²→ha, …), use
           it. This is the only path that respects the *actual* exchange
           unit — critical when an upstream transform (e.g.
           ``BiosphereFlowmapApplier``) has already rescaled the
           exchange (e.g. Bq → kBq) before the matcher sees it. The
           registry's ``unit_conversion`` is keyed to the registry row's
           ``source_unit`` (the *original* unit the row was authored
           for), not to whatever the live exchange now carries; blindly
           preempting with ``cand.unit_conversion`` re-applies a scaling
           that's already been applied and gives a 1000× under-count on
           every Bq→kBq radioactivity edge that the flowmap touched —
           that was the dominant cause of the post-refactor backtest's
           30–60× under-score on ionising radiation.
        2. **Substance-specific override** (``cand.unit_conversion``)
           only when the global converter has *no* path, e.g. kg→m³
           water (density), MJ→kg uranium (heat content), or any other
           cross-dimensional conversion the global table can't express.

        Returns ``None`` if neither resolves.
        """
        via_global = self._resolve_unit_multiplier(exc_unit, target_unit)
        if via_global is not None:
            return via_global
        if cand.unit_conversion not in (None, 1.0):
            return float(cand.unit_conversion)
        return None

    # ------------------------------------------------------------------
    # Cleanup strategies — counted, not silent (fix 1.i).

    @staticmethod
    def _drop_zero_amount_unlinked(sp_data: list[dict]) -> int:
        n = 0
        for ds in sp_data:
            if "exchanges" not in ds:
                continue
            kept = []
            for exc in ds["exchanges"]:
                if exc.get("input") or exc.get("type") != "biosphere" or exc.get("amount"):
                    kept.append(exc)
                else:
                    n += 1
            ds["exchanges"] = kept
        return n

    @staticmethod
    def _drop_final_waste_flows(sp_data: list[dict]) -> int:
        n = 0
        for ds in sp_data:
            if "exchanges" not in ds:
                continue
            kept = []
            for exc in ds["exchanges"]:
                if exc.get("input") or exc.get("categories") != ("Final waste flows",):
                    kept.append(exc)
                else:
                    n += 1
            ds["exchanges"] = kept
        return n
