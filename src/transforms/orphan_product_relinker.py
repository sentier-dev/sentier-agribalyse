"""``OrphanProductRelinker`` — re-route consumers of un-produced AGB products to ecoinvent.

bw_simapro_csv materialises every distinct ``input`` reference in the
CSV as an ``ActivityDataset`` with ``type='product'``. When the SimaPro
name has the canonical ecoinvent shape (``Display {LOC}| activity name |
Cut-off, S/U - Source``), the ``simapro-ecoinvent-3.9.1-cutoff``
randonneur datapackage normally rewrites the consumer-side exchange
name to the ecoinvent canonical so ``TechnosphereMatcher`` can link it.

A few dozen rows fall through that datapackage. The result is **orphan
products**: ``type='product'`` rows that nothing produces but are still
referenced by AGB activities. In the technosphere matrix each orphan
becomes a row with no diagonal entry, which silently NaNs the LU
factorisation under pardiso (and "failed to factorize" under scipy).

This transform parses each orphan's SimaPro-form name, looks it up in
``EcoinventCatalog`` (parquet-backed; no bw2data), and:

* When matched: rewrites every consumer ``input`` to point at the
  ecoinvent activity (and copies name / location / unit / reference
  product so downstream linking doesn't undo it).
* When unmatched: drops the consumer's ``input`` field so
  ``drop_unlinked`` removes the dangling exchange.
* In both cases: removes the now-unreferenced orphan product entry
  from ``sp.data`` before the ``ScoringPackage`` emit so it doesn't
  materialise as a singular matrix row.

Runs AFTER ``RestoreSimaproNamesTransform`` (so the simapro→ecoinvent
datapackage has already had its chance) and BEFORE
``TechnosphereMatcher`` (whose catalog match would otherwise try to
match against orphan names that no ecoinvent activity has).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

from config import Settings
from core.logging import Logging
from matching.ecoinvent_catalog import EcoinventActivityRef, EcoinventCatalog

# "Display name {LOC}| ecoinvent-style suffix | Cut-off, S/U - Source" → (display, loc, suffix)
_SIMAPRO_NAME = re.compile(r"^(.+?)\s*\{([^}]+)\}\s*\|\s*(.+?)\s*\|\s*")


@dataclass(frozen=True)
class OrphanProductRelinker:
    settings: Settings
    catalog: EcoinventCatalog

    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(self, sp) -> dict[str, int]:
        agb_db_name = self.settings.resolved_agribalyse_db_name
        ei_db_name = self.settings.ecoinvent_db_name

        orphans, products_by_code = self._find_orphans(sp.data, agb_db_name)
        if not orphans:
            self._log.info("transforms.orphan_products.none")
            return {"orphan_products": 0, "relinked": 0, "dropped": 0}

        ei_lookup = self._build_ei_lookup(ei_db_name)
        agb_lookup = self._build_agb_sibling_index(sp.data, agb_db_name, products_by_code)
        orphan_to_target: dict[str, tuple[str, str, str, str | None, str | None, str | None]] = {}
        unmapped: list[str] = []
        n_agb_siblings = 0
        for code in orphans:
            target = self._match_orphan(
                products_by_code[code], ei_lookup, agb_lookup, ei_db_name, agb_db_name
            )
            if target is None:
                unmapped.append(products_by_code[code]["name"])
            else:
                if target[0] == agb_db_name:
                    n_agb_siblings += 1
                orphan_to_target[code] = target

        n_relinked, n_dropped = self._rewrite_consumer_exchanges(
            sp.data, agb_db_name, orphans, orphan_to_target
        )
        sp.data = [
            ds for ds in sp.data if not (ds.get("type") == "product" and ds.get("code") in orphans)
        ]

        self._log.info(
            "transforms.orphan_products.relinked",
            orphan_products=len(orphans),
            relinked=n_relinked,
            dropped=n_dropped,
            unmapped=len(unmapped),
            agb_sibling_hits=n_agb_siblings,
            unmapped_samples=unmapped[:5],
        )
        return {
            "orphan_products": len(orphans),
            "relinked": n_relinked,
            "dropped": n_dropped,
            "unmapped": len(unmapped),
            "agb_sibling_hits": n_agb_siblings,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _find_orphans(sp_data: list[dict], agb_db_name: str) -> tuple[list[str], dict[str, dict]]:
        """Return ``(orphan_codes, code → product_dict)``."""
        products_by_code: dict[str, dict] = {}
        produced: set[str] = set()
        for ds in sp_data:
            if ds.get("type") == "product":
                code = ds.get("code")
                if code:
                    products_by_code[code] = ds
                continue
            for exc in ds.get("exchanges", []):
                inp = exc.get("input")
                if not inp:
                    continue
                db, code = inp
                if db != agb_db_name:
                    continue
                if exc.get("type") == "production" or (
                    exc.get("type") == "technosphere" and exc.get("functional") is True
                ):
                    produced.add(code)
        return [c for c in products_by_code if c not in produced], products_by_code

    def _build_ei_lookup(
        self,
        ei_db_name: str,
    ) -> dict[tuple[str, str | None], list[EcoinventActivityRef]]:
        """``(name_lower, location) → [activities]``. RoW / GLO entries are reused as fallbacks."""
        lookup: dict[tuple[str, str | None], list[EcoinventActivityRef]] = defaultdict(list)
        for a in self.catalog.activities:
            if a.db != ei_db_name:
                continue
            key = (a.name.lower(), a.location or None)
            lookup[key].append(a)
        return dict(lookup)

    # Geographic-fallback chain used when an orphan's exact location has no
    # producer. Order reflects ecoinvent's own substitution priority:
    # original → European region → global → rest-of-world.
    _LOCATION_FALLBACK: ClassVar[tuple[str, ...]] = (
        "RER",
        "Europe without Switzerland",
        "EU",
        "GLO",
        "RoW",
    )

    def _build_agb_sibling_index(
        self,
        sp_data: list[dict],
        agb_db_name: str,
        products_by_code: dict[str, dict],
    ) -> dict[tuple[str, str], list[tuple[str, str, dict]]]:
        """``(display_lc, suffix_lc) → [(loc, code, product), …]`` for produced AGB siblings.

        An AGB sibling is a same-SimaPro-shape product in a different
        location that some activity in ``sp.data`` actually produces (or
        consumes as a ``functional=True`` waste-treatment input that the
        ``WasteTreatmentFunctionalPromoter`` has already flipped to
        production). We exclude orphans themselves — pointing at an
        orphan would only kick the can down the road.
        """
        produced_codes = self._produced_codes(sp_data, agb_db_name)
        out: dict[tuple[str, str], list[tuple[str, str, dict]]] = defaultdict(list)
        for code, product in products_by_code.items():
            if code not in produced_codes:
                continue
            parsed = _SIMAPRO_NAME.match(product.get("name") or "")
            if not parsed:
                continue
            display, loc, suffix = (g.strip() for g in parsed.groups())
            key = (display.lower(), suffix.lower())
            out[key].append((loc, code, product))
        # Deterministic ordering: sort each bucket by (location, code) so
        # repeated runs against the same data pick the same sibling.
        for _k, v in out.items():
            v.sort(key=lambda r: (r[0], r[1]))
        return dict(out)

    @staticmethod
    def _produced_codes(sp_data: list[dict], agb_db_name: str) -> set[str]:
        produced: set[str] = set()
        for ds in sp_data:
            if ds.get("type") == "product":
                continue
            for exc in ds.get("exchanges", []):
                inp = exc.get("input")
                if not inp:
                    continue
                db, code = inp
                if db != agb_db_name:
                    continue
                if exc.get("type") == "production" or (
                    exc.get("type") == "technosphere" and exc.get("functional") is True
                ):
                    produced.add(code)
        return produced

    def _match_orphan(
        self,
        product: dict,
        ei_lookup: dict[tuple[str, str | None], list[EcoinventActivityRef]],
        agb_lookup: dict[tuple[str, str], list[tuple[str, str, dict]]],
        ei_db_name: str,
        agb_db_name: str,
    ) -> tuple[str, str, str, str | None, str | None, str | None] | None:
        """Return ``(db, code, name, location, unit, reference_product)`` or ``None``.

        Priority order:

        1. AGB-internal geographic sibling at the original location.
        2. AGB-internal geographic sibling at a fallback location
           (``RER`` → ``Europe without Switzerland`` → ``EU`` → ``GLO`` →
           ``RoW``). Closes the gap for the common case where AGB ships
           a ``{RoW}`` variant of the same waste-treatment / market
           activity but no ``{FR}`` (or other country-specific) copy.
        3. Ecoinvent catalog with the same fallback chain.
        """
        parsed = _SIMAPRO_NAME.match(product.get("name") or "")
        if not parsed:
            return None
        display, loc, suffix = (g.strip() for g in parsed.groups())
        display_lc = display.lower()
        suffix_lc = suffix.lower()

        agb_hit = self._match_agb_sibling(agb_lookup, display_lc, suffix_lc, loc)
        if agb_hit is not None:
            sibling_loc, sibling_code, sibling_product = agb_hit
            return (
                agb_db_name,
                sibling_code,
                sibling_product.get("name") or "",
                sibling_loc or None,
                sibling_product.get("unit") or None,
                sibling_product.get("reference product") or None,
            )

        # Possible ecoinvent activity name shapes seen in agb source:
        # 1. "<suffix> <display>"   — e.g. "market for tap water" + "Tap water"
        # 2. "<suffix>"              — e.g. "calendering, rigid sheets" (display = ref product)
        # 3. "<display>"             — fallback when suffix is empty/generic
        # 4. "<display lower> <suffix>" — when SimaPro inverts the convention.
        # Lower-case the display itself, since ecoinvent names are all lowercase.
        candidates = [
            f"{suffix_lc} {display_lc}",
            f"{display_lc} {suffix_lc}",
            suffix_lc,
            display_lc,
        ]
        for cand in candidates:
            hit = self._lookup_with_fallback(ei_lookup, cand, loc)
            if hit is not None:
                return (
                    ei_db_name,
                    hit.code,
                    hit.name,
                    hit.location or None,
                    hit.unit or None,
                    hit.reference_product or None,
                )
        return None

    @classmethod
    def _match_agb_sibling(
        cls,
        agb_lookup: dict[tuple[str, str], list[tuple[str, str, dict]]],
        display_lc: str,
        suffix_lc: str,
        loc: str,
    ) -> tuple[str, str, dict] | None:
        """Find a same-name AGB product produced in a fallback geography."""
        siblings = agb_lookup.get((display_lc, suffix_lc), [])
        if not siblings:
            return None
        for try_loc in cls._fallback_locations(loc):
            for sibling_loc, sibling_code, sibling_product in siblings:
                if sibling_loc == try_loc:
                    return (sibling_loc, sibling_code, sibling_product)
        # As a last resort, accept any produced sibling (deterministic by
        # the sort order applied during index construction).
        return siblings[0]

    @classmethod
    def _fallback_locations(cls, loc: str) -> tuple[str, ...]:
        """Yield locations to try in priority order (original first, dedup)."""
        seen: set[str] = set()
        chain: list[str] = []
        for candidate in (loc, *cls._LOCATION_FALLBACK):
            if candidate and candidate not in seen:
                chain.append(candidate)
                seen.add(candidate)
        return tuple(chain)

    @classmethod
    def _lookup_with_fallback(
        cls,
        lookup: dict[tuple[str, str | None], list[EcoinventActivityRef]],
        name: str,
        loc: str,
    ) -> EcoinventActivityRef | None:
        for try_loc in cls._fallback_locations(loc):
            hits = lookup.get((name, try_loc), [])
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _rewrite_consumer_exchanges(
        sp_data: list[dict],
        agb_db_name: str,
        orphans: list[str],
        orphan_to_target: dict[str, tuple[str, str, str, str | None, str | None, str | None]],
    ) -> tuple[int, int]:
        """Rewrite or unlink every exchange whose input points at an orphan product.

        Returns ``(n_relinked, n_dropped)``.
        """
        orphan_set = set(orphans)
        n_relinked = 0
        n_dropped = 0
        for ds in sp_data:
            if ds.get("type") == "product":
                continue
            for exc in ds.get("exchanges", []):
                inp = exc.get("input")
                if not inp:
                    continue
                db, code = inp
                if db != agb_db_name or code not in orphan_set:
                    continue
                target = orphan_to_target.get(code)
                if target is not None:
                    new_db, new_code, new_name, new_loc, new_unit, new_refprod = target
                    exc["input"] = (new_db, new_code)
                    exc["name"] = new_name
                    if new_loc is not None:
                        exc["location"] = new_loc
                    if new_unit is not None:
                        exc["unit"] = new_unit
                    if new_refprod:
                        exc["reference product"] = new_refprod
                    n_relinked += 1
                else:
                    # Drop the link so ``drop_unlinked`` removes the dangling exchange.
                    exc.pop("input", None)
                    n_dropped += 1
        return n_relinked, n_dropped
