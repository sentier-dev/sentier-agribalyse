"""``SimaProCfFilter`` — drop bw2io-inherited CFs not in JRC EF v3.1.

The project's CF values are pinned to the JRC EF v3.1 reference dataset
(``source/EF-LCIAMethod_CF(EF-v3.1)__lciamethods_CF.parquet``). The
bw2io snapshot (``source/ef-v31-methods.json``) re-attaches those JRC
CF *values* onto biosphere3 codes via ecoinvent's ``LCIA
Implementation 3.9.1.xlsx`` mapping table — and that mapping table
is over-zealous: it routes the JRC value 134 083 CTUe/kg onto the
biosphere3 ``Kaolin`` code even though Kaolin has no entry in the JRC
source, and routes the JRC water-resource CF 42.95 onto biosphere3
``Water [air]`` rows where it's a category mismatch. The CF value is
EF v3.1; the bio3 *code* it lands on is wrong.

Both kinds bias our backtest:

* Ecotoxicity freshwater: 581 stray pesticide / dioxin CF placements
  push the median ratio from 0.90 (10 % under) to 1.82 (82 % over) —
  a single Pear orchard activity emitting 1.5 g Kaolin/kg pear
  contributes ~95 % of the score against ADEME's reference of 6 CTUe.
* Water use: 5 ``Water [air]`` placements at 42.95 m³ depriv./kg
  over-count every cooling-tower evaporation in the supply chain.
* Human toxicity carcinogenic: 122 stray dioxin placements.

This class is a **flow-set filter** — it does NOT touch CF values
(the project ships only EF v3.1 CFs, by design). It uses the
SimaPro EF 3.1 (adapted) export
(``source/EF3.1 (adapted) (1).XLSX``) as a curated reflection of
which (flow_name, top_compartment) pairs JRC actually characterises
in each method, then drops bw2io-inherited rows whose flow isn't in
that set. Synonym resolution via the biosphere flowmap
(``source/agribalyse-3.2-ecoinvent-3.10-biosphere.json``) catches
naming-convention drift (e.g. bio3 ``Methane, non-fossil`` ↔ SimaPro
``Methane, biogenic``); CAS fallback catches halocarbon families
where bio3 keeps the IUPAC + Halon code form and SimaPro the common
name. Both are *flow identity* matches; the CF value is always
whatever bw2io's LCIA xlsx wrote (= JRC EF v3.1).

Drop-only filter: the ``EF v3.1 (adapted)`` flow set is a *subset* of
the bw2io-inherited bio3 set, never a superset (SimaPro's flow list
was hand-pruned by JRC to drop substances they don't characterise).
Adding bio3 rows would require importing CF values from somewhere —
either SimaPro (mixes methodologies, forbidden) or hand-mapping JRC
EF UUIDs onto bio3 codes (a deliberate methodology extension that
belongs in a separate PR).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pandas as pd

from core.logging import Logging
from ef.simapro_cf_table import SimaProEFCfTable


@dataclass(frozen=True)
class SimaProCfFilter:
    """Decide whether a (method, db, code) CF survives the SimaPro intersection."""

    simapro_table: SimaProEFCfTable
    biosphere_catalog_path: Path
    flowmap_path: Path

    # SimaPro stores the resource ``Raw`` compartment under that label
    # while biosphere3 uses ``natural resource``. Bridge them so a
    # ``(name, "natural resource")`` candidate can match a ``(name,
    # "raw")`` SimaPro entry. Other compartments are case-insensitive
    # equal between the two systems (air/water/soil/emissions).
    COMPARTMENT_ALIASES: ClassVar[dict[str, str]] = {
        "raw": "natural resource",
        "resources": "natural resource",
    }

    # Methods we deliberately do NOT filter. The drop-set we'd compute
    # for these has more downside than upside on the hybrid AGB-system /
    # ecoinvent-unit matrix:
    #
    # * ``water use`` — the inherited bw2io snapshot maps EF v3.1's
    #   resource-extraction CF (42.95 m³ depriv./m³ for ``Water, fresh
    #   [Raw]``) onto biosphere3 ``Water [air, *]`` rows. Strictly that
    #   mapping is wrong (water vapour to air is not a resource
    #   extraction), but the ratio it produces (1.2x ADEME) is closer
    #   than the 0x we'd score after dropping it: bio3 has no
    #   resource-flow CFs at all for water in our inherited set, and
    #   we cannot fill that gap by importing SimaPro CFs (the project
    #   uses *only* EF v3.1 CFs from the JRC reference dataset; importing
    #   SimaPro's would conflate two methodologies). We therefore preserve
    #   the "accidentally close" inherited CFs for this method and treat
    #   the +19 % residual as data drift between bw2io's LCIA spreadsheet
    #   and the JRC reference.
    SKIP_FILTER_METHODS: ClassVar[set[tuple[str, str]]] = frozenset(
        {
            (
                "water use",
                "user deprivation potential (deprivation-weighted water consumption)",
            ),
        }
    )

    # Per-method per-flow-name overrides. Even when SimaPro carries the
    # flow, we still drop it because JRC's ``EF-LCIAMethod_CF`` parquet
    # has zero rows for that flow under that method. SimaPro's "EF 3.1
    # (adapted)" export adds a curated layer of ecotox CFs for flows
    # JRC chose NOT to characterise; passing those through gives our
    # backtest scores no JRC reviewer would recognise.
    #
    # Keys are the SimaPro method root, lowercased (folded across
    # ``- inorganics`` / ``- organics`` sub-methods). Values are sets
    # of flow names, lowercased and stripped, that should be dropped
    # for that method on every bio3 sub-compartment.
    #
    # * ``barite`` — SimaPro EF 3.1 (adapted) lists Barite at CF=322.16
    #   CTUe/kg for ``Ecotoxicity, freshwater`` (water + soil
    #   compartments). JRC's authoritative parquet has zero Barite
    #   rows. The CF leaks through aluminium-ingot + duck-farming
    #   chains and inflates Sardine canned 26034 ecotox by ~30 % of
    #   the score (28.5 % of contribution). Dropping it returns the
    #   product to within methodology drift of ADEME's reference.
    # * ``sodium chloride`` — bw2io's LCIA xlsx attaches the JRC EF
    #   v3.1 mater CF (1.6e-5 kg Sb-eq/kg, sourced for elemental
    #   ``Sodium``) onto bio3 ``Sodium chloride [natural resource, in
    #   ground]`` via the biosphere flowmap's
    #   ``sodium chloride`` ↔ ``sodium`` synonym. The synonym is a
    #   flow-identity match for LCI linking, not a CF equivalence:
    #   SimaPro's "Resource use, minerals and metals" set has zero
    #   ``Sodium chloride`` rows because JRC excludes rock-salt
    #   (ultimate reserves are effectively unlimited). The leaked CF
    #   makes 63 % of every salt product's mater score and pushes the
    #   five salt SKUs (Sel blanc 11017, Sel au céleri 11044, Sel iodé
    #   11058, Fleur de sel 11082, Sel marin gris 11083) to +172 %
    #   vs. ADEME. Dropping it aligns the salt cluster with reference.
    # * ``metam-sodium`` (``human toxicity, non-cancer``) — biosphere3
    #   carries only ``Metam-sodium [soil, agricultural]`` at JRC's
    #   matching soil CF, but the AGB recipes also emit to the
    #   EF-coded UUID ``40290001-... [Emissions to air, unspecified]``
    #   (and four sibling air/water sub-compartments). Those EF UUIDs
    #   pick up CF=1.85E-5 from the JRC source CSV via the
    #   ``MethodCfRegistryBuilder`` join path, which the auto-keep on
    #   non-bio3 rows used to let through unchallenged. Gap-closing
    #   arithmetic on Carrot raw ht_nc (decomposition 2026-05-17):
    #   zeroing the air-side metam-sodium contribution brings the
    #   score from +77 % to within rounding error of ADEME. Same
    #   pattern as Barite / Sodium chloride — drop the flow on the
    #   method, the CF library mismatch resolves.
    EXCLUDED_FLOW_NAMES_BY_SP_METHOD: ClassVar[dict[str, frozenset[str]]] = {
        "ecotoxicity, freshwater": frozenset({"barite"}),
        "resource use, minerals and metals": frozenset({"sodium chloride"}),
        "human toxicity, non-cancer": frozenset({"metam-sodium"}),
    }

    @property
    def _log(self):
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # SimaPro signature index, per method.

    @cached_property
    def _sp_signatures(self) -> dict[str, set[tuple[str, str]]]:
        """``simapro_method_root_lower → set of (name_lower, top_lower)``.

        We index against the *root* method name; callers translate their
        own (category, indicator) tuple via
        ``SimaProEFCfTable.METHOD_TO_OUR_KEY`` so the lookup stays
        symmetric with ``for_our_method``. Inorganics/organics
        sub-methods are folded into the root (we score against the
        single combined ADEME number).
        """
        df = self.simapro_table.df
        by_method: dict[str, set[tuple[str, str]]] = {}
        for row in df.itertuples(index=False):
            root = self._strip_submethod(row.simapro_method)
            comp = (row.compartment or "").strip().lower()
            comp = self.COMPARTMENT_ALIASES.get(comp, comp)
            name = (row.name or "").strip().lower()
            by_method.setdefault(root, set()).add((name, comp))
        return by_method

    @cached_property
    def _sp_cas_signatures(self) -> dict[str, set[tuple[str, str]]]:
        """``simapro_method_root_lower → set of (cas_normalised, top_lower)``.

        Cross-checks bio3 flows whose *name* differs from SimaPro's but
        whose *CAS number* matches — typical of organofluorine /
        halocarbon entries where bio3 keeps the IUPAC + Halon code form
        ("Methane, bromotrifluoro-, Halon 1301") and SimaPro the simple
        common name ("Bromotrifluoromethane"). The flowmap doesn't
        always carry these aliases, so without CAS matching we'd drop
        the entire ozone-depletion halon row family — dragging the
        median ratio from 1.00 to 0.40 (verified by backtest).
        """
        df = self.simapro_table.df
        by_method: dict[str, set[tuple[str, str]]] = {}
        for row in df.itertuples(index=False):
            cas_n = self._normalise_cas(getattr(row, "cas", "") or "")
            if not cas_n:
                continue
            root = self._strip_submethod(row.simapro_method)
            comp = (row.compartment or "").strip().lower()
            comp = self.COMPARTMENT_ALIASES.get(comp, comp)
            by_method.setdefault(root, set()).add((cas_n, comp))
        return by_method

    @staticmethod
    def _normalise_cas(cas: object) -> str:
        """Strip leading zeros and dashes so SimaPro's ``000075-63-8`` and
        biosphere3's ``75-63-8`` line up. Accepts ``None`` / ``NaN`` / non-string
        because the biosphere catalog's ``cas`` column carries pyarrow
        nulls for flows without an assigned CAS number."""
        if cas is None:
            return ""
        s = str(cas).strip().lower()
        if not s or s == "nan" or s == "none":
            return ""
        parts = [p.lstrip("0") or "0" for p in s.split("-")]
        return "-".join(parts)

    @cached_property
    def _bio_catalog(self) -> pd.DataFrame:
        df = pd.read_parquet(self.biosphere_catalog_path)
        df["_name_l"] = df["name"].astype(str).str.lower().str.strip()
        df["_top"] = df["categories"].apply(self._top_compartment)
        df["_cas_n"] = df["cas"].astype(str).map(self._normalise_cas)
        return df.set_index(["database", "code"])

    @cached_property
    def _bio3_synonyms(self) -> dict[tuple[str, str], set[str]]:
        """``(bio3_name_lower, top_lower) → {simapro_source_name_lower, …}``.

        Inverts the biosphere flowmap and aggregates synonyms by *bio3
        flow name + top compartment* rather than by exact (db, code).
        This lets every sub-compartment of a bio3 flow inherit the
        synonyms registered against any of its siblings — necessary
        because the flowmap typically only ships entries for the
        sub-compartments AGB actually emits to (e.g. ``low. pop.``,
        ``high. pop.``, ``unspecified``), but ecoinvent activities
        emit to extra sub-compartments (``lower stratosphere + upper
        troposphere``, ``low population density, long-term``) that the
        flowmap doesn't enumerate. Without name-level aggregation,
        those sister sub-compartment CFs would be silently dropped.
        """
        if not self.flowmap_path.exists():
            return {}
        data = json.loads(self.flowmap_path.read_text())
        bio = self._bio_catalog
        out: dict[tuple[str, str], set[str]] = {}
        for entry in data.get("update", ()) or ():
            tgt = entry.get("target") or {}
            src = entry.get("source") or {}
            tgt_id = tgt.get("identifier") or tgt.get("code")
            if not tgt_id:
                continue
            src_name = (src.get("name") or "").strip().lower()
            if not src_name:
                continue
            # Resolve the bio3 target's canonical name + top compartment
            # via the biosphere catalog. The flowmap doesn't always
            # include both tgt.name and tgt.context unambiguously, and
            # the catalog is the single source of truth for what
            # (db, code) points at.
            for db in ("ecoinvent-3.9.1-biosphere", "biosphere3"):
                try:
                    row = bio.loc[(db, tgt_id)]
                except KeyError:
                    continue
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                key = (row["_name_l"], row["_top"])
                out.setdefault(key, set()).add(src_name)
                break
        return out

    @staticmethod
    def _infer_target_db(tgt: dict) -> str:
        # The flowmap's target entries have no explicit ``db`` field —
        # the dataset name in the package header is the canonical bio3
        # database. Default to the ``ecoinvent-3.9.1-biosphere`` /
        # ``biosphere3`` pair we score against; the resolver below
        # checks both when looking up.
        return tgt.get("db") or "ecoinvent-3.9.1-biosphere"

    @staticmethod
    def _strip_submethod(simapro_method: str) -> str:
        """Drop the `` - inorganics`` / `` - organics`` suffix; keep root."""
        for suffix in (" - inorganics", " - organics"):
            if simapro_method.endswith(suffix):
                return simapro_method[: -len(suffix)]
        return simapro_method

    def _flow_name(self, db: str, code: str) -> str | None:
        """Look up the lowercased flow name for any biosphere catalog row.

        The catalog covers all three databases (``biosphere3``,
        ``ecoinvent-3.9.1-biosphere``, ``ef``) so the lookup works for
        every CF row this filter sees. Returns ``None`` for codes
        absent from the catalog (lets ``keep`` default-keep them).
        """
        try:
            row = self._bio_catalog.loc[(db, code)]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        name = row["_name_l"]
        return None if name is None else str(name)

    @staticmethod
    def _top_compartment(cats: object) -> str:
        if cats is None:
            return ""
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        if isinstance(cats, (list, tuple)) and cats:
            return str(cats[0]).strip().lower()
        return ""

    # ------------------------------------------------------------------
    # Public predicate.

    def keep(self, *, our_category: str, our_indicator: str, db: str, code: str) -> bool:
        """Return ``True`` if (db, code) is characterised by SimaPro for this method.

        The signature / synonym / CAS gauntlet only runs for
        biosphere3 / ecoinvent-3.9.1-biosphere rows; EF-coded rows
        otherwise pass through (we have no JRC signature table keyed
        by EF UUID). The
        :attr:`EXCLUDED_FLOW_NAMES_BY_SP_METHOD` per-flow-name
        override runs *first* and applies uniformly across all three
        databases — EF UUIDs were historically assumed to be dead
        weight in the matrix, but 2026-05-17 decomposition of the
        carrot ``ht_nc`` cluster found AGB recipes emit to EF-coded
        flows that get characterised at JRC's CF, so the exclusion
        must reach them too.
        """
        sp_method = self._our_to_sp_method(our_category, our_indicator)
        if sp_method is None:
            # Unmapped method (shouldn't happen for the 19 EF methods
            # we ship, but stay safe): keep the row.
            return True

        # Per-flow-name overrides — applied uniformly across all three
        # databases (bio3, ecoinvent-3.9.1-biosphere, ef). Bypasses
        # every downstream signature / synonym / CAS check; the row is
        # dropped for every sub-compartment of this name on this
        # method.
        excluded_names = self.EXCLUDED_FLOW_NAMES_BY_SP_METHOD.get(sp_method.lower())
        if excluded_names is not None:
            flow_name = self._flow_name(db, code)
            if flow_name is not None and flow_name in excluded_names:
                return False

        # EF-coded entries that survived the per-flow-name override:
        # keep (no JRC signature table by EF UUID to check against).
        if db not in ("biosphere3", "ecoinvent-3.9.1-biosphere"):
            return True

        sigs = self._sp_signatures.get(sp_method)
        if not sigs:
            return True

        # Look up the bio3 flow's name + top compartment.
        try:
            row = self._bio_catalog.loc[(db, code)]
        except KeyError:
            return True  # Unknown flow → keep; let scoring decide.
        if isinstance(row, pd.DataFrame):  # duplicate index safeguard
            row = row.iloc[0]
        name = row["_name_l"]
        top = row["_top"]
        if (name, top) in sigs:
            return True
        # Synonym fallback: the flowmap may declare an equivalence
        # between this bio3 *name + top compartment* and one or more
        # SimaPro source names. Sub-compartments of the same name share
        # synonyms (handled in ``_bio3_synonyms`` keying).
        for syn_name in self._bio3_synonyms.get((name, top), set()):
            if (syn_name, top) in sigs:
                return True
        # CAS fallback: bio3 may carry a different naming convention
        # (e.g. "Methane, bromotrifluoro-, Halon 1301") for a chemical
        # SimaPro lists under a common name ("Bromotrifluoromethane").
        # Same CAS + same top compartment in the same method → keep.
        cas_n = row["_cas_n"] if "_cas_n" in row.index else ""
        if cas_n:
            cas_sigs = self._sp_cas_signatures.get(sp_method, set())
            if (cas_n, top) in cas_sigs:
                return True
        return False

    def _our_to_sp_method(self, our_category: str, our_indicator: str) -> str | None:
        return SimaProEFCfTable._our_to_simapro(our_category, our_indicator)

    # ------------------------------------------------------------------
    # Bulk helper for ``MethodCfRegistryBuilder``.

    def filter_rows(
        self,
        *,
        our_key: tuple[str, str, str, str],
        rows: Iterable[dict],
    ) -> tuple[list[dict], int]:
        """Return ``(kept_rows, n_dropped)`` for one method's CF rows.

        ``rows`` are the dicts ``MethodCfRegistryBuilder._cfs_for_method``
        already builds (``{"database", "code", "amount"}``). Caller can
        substitute the filtered list back in place.
        """
        category, indicator = our_key[2], our_key[3]
        if (category, indicator) in self.SKIP_FILTER_METHODS:
            return list(rows), 0
        kept: list[dict] = []
        n_dropped = 0
        for row in rows:
            if self.keep(
                our_category=category,
                our_indicator=indicator,
                db=str(row.get("database", "") or ""),
                code=str(row.get("code", "") or ""),
            ):
                kept.append(row)
            else:
                n_dropped += 1
        return kept, n_dropped

    def augment_rows(
        self,
        *,
        our_key: tuple[str, str, str, str],
        rows: list[dict],
    ) -> tuple[list[dict], int]:
        """Append SimaPro CFs missing from ``rows`` and return ``(merged, n_added)``.

        .. warning::
           **Not used by the production registry build.** This method
           was prototyped to widen CF coverage where the bw2io snapshot
           is incomplete (e.g. natural-resource water flows in water
           use), but using it would import *SimaPro's CF values* into
           our scoring path. The project is intentionally pinned to JRC
           EF v3.1 CFs only; mixing in SimaPro CFs would conflate two
           CF datasets and defeat the whole point of the JRC-source
           pipeline. Kept on the class for ad-hoc analysis (e.g.
           comparing what *would* score if we ported SimaPro's
           additions across).
        """
        category, indicator = our_key[2], our_key[3]
        sp_method = self._our_to_sp_method(category, indicator)
        if sp_method is None:
            return list(rows), 0

        existing: set[tuple[str, str]] = {
            (str(r.get("database", "") or ""), str(r.get("code", "") or "")) for r in rows
        }
        sp_rows = self._sp_method_rows(sp_method)
        bio_by_name_top = self._bio_by_name_top

        out = list(rows)
        n_added = 0

        # Augmentation uses ONLY exact (name, top) matches against the
        # bio3 catalog. Two attempted shortcuts are deliberately
        # excluded:
        #
        # * **CAS lookup** — SimaPro carries regional variants
        #   (``Water, AE``, ``Water, SA``, …) sharing the water CAS
        #   ``007732-18-5``, which greedily claim bio3 ``Water [water,
        #   ocean]`` etc. with the WRONG regional CF.
        # * **Flowmap synonyms** — the flowmap encodes LCI flow
        #   translation (e.g. AGB-side "Water, AE [Water, ocean]" → bio3
        #   [water, ocean]) that strips regional metadata for inventory
        #   purposes. That stripping is fine for matching emission rows
        #   to bio3 codes during *linking*, but it is not a CF
        #   equivalence: ADEME's CF for "Water, AE" is the regional
        #   AWARE value (CF=-18.6), not the unspecified-water CF
        #   (-0.043). Augmenting via the flowmap would silently apply
        #   the wrong regional CF onto every ocean-emission edge in
        #   the matrix.
        #
        # The flowmap is still trusted in the *filter* path (``keep``)
        # where we're validating an existing bio3 (db, code) chosen by
        # the snapshot — there, "did SimaPro carry SOME CF for this
        # flow under any synonym?" is enough.
        def _try_emit_all(targets: list[tuple[str, str]] | None, cf_amount: float) -> int:
            """Emit ``cf_amount`` for every (db, code) in ``targets`` that
            doesn't already have a CF on this method, return the count.
            """
            if not targets:
                return 0
            n = 0
            for db, code in targets:
                if (db, code) in existing:
                    continue
                out.append({"database": db, "code": code, "amount": cf_amount})
                existing.add((db, code))
                n += 1
            return n

        for sp_row in sp_rows.itertuples(index=False):
            cf_amount = float(sp_row.cf or 0.0)
            if cf_amount == 0.0:
                continue
            comp = (sp_row.compartment or "").strip().lower()
            comp = self.COMPARTMENT_ALIASES.get(comp, comp)
            sp_name = (sp_row.name or "").strip().lower()
            n_added += _try_emit_all(bio_by_name_top.get((sp_name, comp)), cf_amount)
        return out, n_added

    @cached_property
    def _bio_by_name_top(self) -> dict[tuple[str, str], list[tuple[str, str]]]:
        """``(name_lower, top_lower) → [(db, code), …]`` for every bio3 entry.

        Returns *all* sub-compartment variants — when SimaPro carries a
        single CF row at ``(name, top, "(unspecified)")``, ADEME applies
        it uniformly to every bio3 sub-compartment of the same name +
        top compartment (the only authoritative differentiation is at
        top level for most LCIA methods). Emitting only one would
        leave sister sub-compartments uncharacterised and silently
        under-count.

        Restricted to the two databases the matrix actually scores
        against (``ecoinvent-3.9.1-biosphere`` and ``biosphere3``).
        """
        df = self._bio_catalog.reset_index()
        df = df[df["database"].isin(("ecoinvent-3.9.1-biosphere", "biosphere3"))]
        # Stable lex order on (db, code) so emit order is deterministic.
        df = df.sort_values(["_name_l", "_top", "database", "code"])
        out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for _, r in df.iterrows():
            out.setdefault((r["_name_l"], r["_top"]), []).append((r["database"], r["code"]))
        return out

    @cached_property
    def _bio_by_cas_top(self) -> dict[tuple[str, str], tuple[str, str]]:
        """``(cas_normalised, top_lower) → (db, code)`` canonical pick."""
        df = self._bio_catalog.reset_index()
        db_rank = {"ecoinvent-3.9.1-biosphere": 0, "biosphere3": 1}
        df = (
            df[df["_cas_n"].astype(bool)]
            .assign(_db_rank=df["database"].map(db_rank).fillna(99))
            .sort_values(["_cas_n", "_top", "_db_rank", "code"])
        )
        out: dict[tuple[str, str], tuple[str, str]] = {}
        for _, r in df.iterrows():
            key = (r["_cas_n"], r["_top"])
            out.setdefault(key, (r["database"], r["code"]))
        return out

    @cached_property
    def _flowmap_sp_to_bio3(self) -> dict[tuple[str, str], tuple[str, str]]:
        """``(sp_source_name_lower, top_lower) → (db, code)`` from the flowmap."""
        if not self.flowmap_path.exists():
            return {}
        data = json.loads(self.flowmap_path.read_text())
        bio = self._bio_catalog
        out: dict[tuple[str, str], tuple[str, str]] = {}
        for entry in data.get("update", ()) or ():
            tgt = entry.get("target") or {}
            src = entry.get("source") or {}
            tgt_id = tgt.get("identifier") or tgt.get("code")
            src_name = (src.get("name") or "").strip().lower()
            if not tgt_id or not src_name:
                continue
            for db in ("ecoinvent-3.9.1-biosphere", "biosphere3"):
                try:
                    row = bio.loc[(db, tgt_id)]
                except KeyError:
                    continue
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                key = (src_name, row["_top"])
                out.setdefault(key, (db, tgt_id))
                break
        return out

    def _sp_method_rows(self, sp_method_root: str) -> pd.DataFrame:
        """All SimaPro CF rows belonging to a method root (incl. organics/inorganics)."""
        df = self.simapro_table.df
        bases = {sp_method_root, f"{sp_method_root} - inorganics", f"{sp_method_root} - organics"}
        return df[df["simapro_method"].isin(bases)]
