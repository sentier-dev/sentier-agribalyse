"""``SimaproFormResolver`` — SimaPro-form exchange name → ecoinvent activity.

SimaPro writes a reference to an ecoinvent dataset as::

    Display name {LOC}| activity name | Cut-off, U - Copied from Ecoinvent U

Most of these are rewritten to ecoinvent's own labels by the edge-label
and SimaPro-names datapackages. The ones those packages do not list (the
embedded copies outside the aggregated-process deletion list, and names
SimaPro truncated to ``market for`` / ``production``) reach the
technosphere matcher in SimaPro form. This class resolves them against
the user's own ecoinvent catalog: a naming rule, no shipped mapping.

A name resolves only when exactly one catalog activity carries the
candidate name at the location. When several do (a multi-output activity
such as ``soybean meal and crude oil production``, one dataset per
co-product), the SimaPro display name picks the one whose reference
product it is. Names that are still ambiguous stay unlinked.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from matching.ecoinvent_catalog import EcoinventActivityRef, EcoinventCatalog


@dataclass(frozen=True)
class SimaproFormResolver:
    catalog: EcoinventCatalog
    ei_db: str

    # "Display {LOC}| activity | Cut-off, …" → (display, loc, activity)
    _NAME: ClassVar[re.Pattern[str]] = re.compile(r"^(.+?)\s*\{([^}]+)\}\s*\|\s*(.*?)\s*\|\s*")

    # Tried after the name's own location, in ecoinvent's substitution order.
    _LOCATION_FALLBACK: ClassVar[tuple[str, ...]] = (
        "RER",
        "Europe without Switzerland",
        "GLO",
        "RoW",
    )

    def resolve(self, simapro_name: str) -> EcoinventActivityRef | None:
        parsed = self._NAME.match(simapro_name or "")
        if parsed is None:
            return None
        display, location, activity = (g.strip().lower() for g in parsed.groups())
        location = parsed.group(2).strip()
        for candidate in self._candidates(display, activity):
            hit = self._unique(candidate, location, display)
            if hit is not None:
                return hit
        return None

    @cached_property
    def _by_name_location(self) -> dict[tuple[str, str], tuple[EcoinventActivityRef, ...]]:
        index: dict[tuple[str, str], list[EcoinventActivityRef]] = defaultdict(list)
        for activity in self.catalog.activities:
            if activity.db == self.ei_db:
                index[(activity.name.lower(), activity.location)].append(activity)
        return {key: tuple(refs) for key, refs in index.items()}

    @staticmethod
    def _candidates(display: str, activity: str) -> tuple[str, ...]:
        """ecoinvent name shapes behind a SimaPro label, most specific first.

        * ``activity`` — the label is complete (``carrot production``).
        * ``activity display`` — truncated market (``market for`` + product).
        * ``display activity`` — truncated transformer (product + ``production``).
        """
        if not activity:
            return ()
        return (activity, f"{activity} {display}", f"{display} {activity}")

    def _unique(self, name: str, location: str, product: str) -> EcoinventActivityRef | None:
        seen: set[str] = set()
        for try_location in (location, *self._LOCATION_FALLBACK):
            if try_location in seen:
                continue
            seen.add(try_location)
            hits = self._by_name_location.get((name, try_location), ())
            if len(hits) > 1:
                hits = tuple(h for h in hits if h.reference_product.lower() == product)
                return hits[0] if len(hits) == 1 else None
            if hits:
                return hits[0]
        return None
