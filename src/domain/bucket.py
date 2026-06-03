"""Compartment bucket classification — air / water / soil / resource / unspecified.

Lives as a class-with-classmethods rather than a free function. The bucket
collapses both SimaPro-style cats (``Emissions to air``, ``Resources``) and
ecoinvent-style cats (``air``, ``natural resource``) onto a single vocabulary
so ``(name, bucket)`` keys align across sources.
"""

from __future__ import annotations

from enum import StrEnum


class Bucket(StrEnum):
    """Top-level compartment bucket. Values are the canonical lowercase strings."""

    AIR = "air"
    WATER = "water"
    SOIL = "soil"
    RESOURCE = "resource"
    UNSPECIFIED = "unspecified"

    @classmethod
    def from_categories(cls, cats: tuple[str, ...] | list[str] | None) -> Bucket:
        """Map any compartment tuple to its top-level bucket.

        Walks every element so EF's ``('Emissions', 'Emissions to water', ...)``
        and ``('Resources', 'Resources from ground', ...)`` schemas resolve
        correctly — the compartment word lives in element 1, not 0. Earlier
        elements take priority so ecoinvent's ``('natural resource', 'in air')``
        still maps to ``RESOURCE`` (cat[0]) rather than ``AIR`` (cat[1]).
        """
        if not cats:
            return cls.UNSPECIFIED
        for c in cats:
            s = str(c).strip().lower()
            if "air" in s:
                return cls.AIR
            if "water" in s:
                return cls.WATER
            if "soil" in s:
                return cls.SOIL
            if "resource" in s or "raw" in s:
                return cls.RESOURCE
        return cls.UNSPECIFIED

    @classmethod
    def from_harmonised_iri(cls, context_iri: str) -> Bucket:
        """Map a harmonised-flows ``context_iri`` to its bucket.

        IRIs encode the compartment as the last path segment, e.g.
        ``envi-air-indr-unkn``, ``envi-wate-suwa``, ``reso-grou``.
        """
        if not context_iri:
            return cls.UNSPECIFIED
        last = context_iri.rstrip("/").split("/")[-1].lower()
        if last.startswith("envi-air"):
            return cls.AIR
        if last.startswith("envi-wate"):
            return cls.WATER
        if last.startswith("envi-grou"):
            return cls.SOIL
        if last.startswith(("reso", "laus")):
            return cls.RESOURCE
        return cls.UNSPECIFIED
