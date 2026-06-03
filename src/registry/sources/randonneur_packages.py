"""Source ingesters for bundled randonneur datapackages.

One class per package. Each takes a ``RandonneurDataLoader`` and produces
the appropriate registry rows on ``read()``.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain import (
    Bucket,
    ContextNorm,
    Mapping,
    SourceKind,
    Tier,
    UnitAlias,
    UnitConversion,
)
from readers import RandonneurDataLoader


@dataclass(frozen=True)
class RandonneurAgribalyseBiosphereSource:
    """``agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches`` (96 rows).

    Replaces the residual hardcoded ``BIOSPHERE_SYNONYMS`` (fix 1.d).
    Wired to tier 2.
    """

    loader: RandonneurDataLoader
    label: str = "agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches"
    tier: Tier = Tier.RANDONNEUR_AGB_SPECIFIC
    provenance: str = "randonneur.agribalyse-3.1.1-biosphere-manual-matches"

    def read(self) -> list[Mapping]:
        pkg = self.loader.load(self.label)
        out: list[Mapping] = []
        for i, entry in enumerate(pkg.get("replace", [])):
            src = entry.get("source", {})
            tgt = entry.get("target", {})
            src_ctx = tuple(src.get("context", ()))
            out.append(
                Mapping(
                    source_kind=SourceKind.AGB_FLOW,
                    source_name=str(src.get("name", "")).strip(),
                    source_unit=str(src.get("unit", "")).strip(),
                    source_context=src_ctx,
                    source_top_bucket=Bucket.from_categories(src_ctx),
                    target_db="",
                    target_code=str(tgt.get("identifier", "")).strip(),
                    target_name=str(tgt.get("name", "")).strip(),
                    target_unit=str(tgt.get("unit", "")).strip(),
                    unit_conversion=float(entry.get("conversion_factor") or 1.0),
                    priority_tier=self.tier,
                    provenance=self.provenance,
                    provenance_row=str(i),
                    notes=str(entry.get("comment", "")).strip(),
                )
            )
        return out


@dataclass(frozen=True)
class RandonneurSimaproBiosphereSource:
    """``SimaPro-9-ecoinvent-3.9-biosphere-manual-matches`` (580 rows). Tier 3."""

    loader: RandonneurDataLoader
    label: str = "SimaPro-9-ecoinvent-3.9-biosphere-manual-matches"
    tier: Tier = Tier.RANDONNEUR_SIMAPRO_BIO
    provenance: str = "randonneur.simapro-9-ecoinvent-3.9-biosphere-manual-matches"

    def read(self) -> list[Mapping]:
        pkg = self.loader.load(self.label)
        out: list[Mapping] = []
        for i, entry in enumerate(pkg.get("replace", [])):
            src = entry.get("source", {})
            tgt = entry.get("target", {})
            src_ctx = tuple(src.get("context", ()))
            out.append(
                Mapping(
                    source_kind=SourceKind.AGB_FLOW,
                    source_name=str(src.get("name", "")).strip(),
                    source_unit=str(src.get("unit", "")).strip(),
                    source_context=src_ctx,
                    source_top_bucket=Bucket.from_categories(src_ctx),
                    target_db="",
                    target_code=str(tgt.get("identifier", "")).strip(),
                    target_name=str(tgt.get("name", "")).strip(),
                    target_unit=str(tgt.get("unit", "")).strip(),
                    unit_conversion=float(entry.get("conversion_factor") or 1.0),
                    priority_tier=self.tier,
                    provenance=self.provenance,
                    provenance_row=str(i),
                    notes=str(entry.get("comment", "")).strip(),
                )
            )
        return out


@dataclass(frozen=True)
class RandonneurWaterSlashM3Source:
    """``simapro-9-ecoinvent-3-water-slash-m3`` (~39 675 rows). Tier 5 (fix 1.c)."""

    loader: RandonneurDataLoader
    label: str = "simapro-9-ecoinvent-3-water-slash-m3"
    tier: Tier = Tier.RANDONNEUR_WATER_M3
    provenance: str = "randonneur.simapro-9-ecoinvent-3-water-slash-m3"

    def read(self) -> list[Mapping]:
        pkg = self.loader.load(self.label)
        out: list[Mapping] = []
        for i, entry in enumerate(pkg.get("replace", [])):
            src = entry.get("source", {})
            tgt = entry.get("target", {})
            src_ctx = tuple(src.get("context", ()))
            out.append(
                Mapping(
                    source_kind=SourceKind.AGB_FLOW,
                    source_name=str(src.get("name", "")).strip(),
                    source_unit=str(src.get("unit", "")).strip(),
                    source_context=src_ctx,
                    source_top_bucket=Bucket.from_categories(src_ctx),
                    target_db="",
                    target_code=str(tgt.get("identifier", "")).strip(),
                    target_name=str(tgt.get("name", "")).strip(),
                    target_unit=str(tgt.get("unit", "")).strip(),
                    unit_conversion=float(entry.get("conversion_factor") or 1.0),
                    priority_tier=self.tier,
                    provenance=self.provenance,
                    provenance_row=str(i),
                    notes=str(entry.get("location", "")).strip(),
                )
            )
        return out


@dataclass(frozen=True)
class RandonneurSimaproContextSource:
    """``simapro-9-ecoinvent-3-context`` (101 rows). Feeds context_normalisation."""

    loader: RandonneurDataLoader
    label: str = "simapro-9-ecoinvent-3-context"
    provenance: str = "randonneur.simapro-9-ecoinvent-3-context"

    def read(self) -> list[ContextNorm]:
        pkg = self.loader.load(self.label)
        out: list[ContextNorm] = []
        for entry in pkg.get("replace", []):
            src = entry.get("source", {})
            tgt = entry.get("target", {})
            out.append(
                ContextNorm(
                    source_context=tuple(src.get("context", ())),
                    target_context=tuple(tgt.get("context", ())),
                    provenance=self.provenance,
                )
            )
        return out


@dataclass(frozen=True)
class RandonneurUnitAliasSource:
    """``Flowmapper-standard-units-harmonization`` — case-insensitive unit aliases."""

    loader: RandonneurDataLoader
    label: str = "Flowmapper-standard-units-harmonization"
    provenance: str = "randonneur.flowmapper-standard-units-harmonization"

    def read(self) -> list[UnitAlias]:
        pkg = self.loader.load(self.label)
        out: list[UnitAlias] = []
        for entry in pkg.get("update", []):
            src_unit = str(entry.get("source", {}).get("unit", "")).strip()
            tgt_unit = str(entry.get("target", {}).get("unit", "")).strip()
            if src_unit and tgt_unit:
                out.append(
                    UnitAlias(alias=src_unit, canonical=tgt_unit, provenance=self.provenance)
                )
        return out


@dataclass(frozen=True)
class RandonneurUnitConversionSource:
    """``generic-brightway-unit-conversions`` — replaces hardcoded ``UNIT_CONVERSIONS`` (fix 1.k)."""

    loader: RandonneurDataLoader
    label: str = "generic-brightway-unit-conversions"
    provenance: str = "randonneur.generic-brightway-unit-conversions"

    def read(self) -> list[UnitConversion]:
        pkg = self.loader.load(self.label)
        out: list[UnitConversion] = []
        for entry in pkg.get("replace", []):
            src_unit = str(entry.get("source", {}).get("unit", "")).strip()
            tgt = entry.get("target", {})
            tgt_unit = str(tgt.get("unit", "")).strip()
            multiplier = tgt.get("allocation")
            if not src_unit or not tgt_unit or multiplier is None:
                continue
            out.append(
                UnitConversion(
                    source_unit=src_unit,
                    target_unit=tgt_unit,
                    multiplier=float(multiplier),
                    provenance=self.provenance,
                )
            )
        return out


@dataclass(frozen=True)
class RandonneurUnitNormalisationSource:
    """``generic-brightway-units-normalization`` (46 rows) — also unit aliases."""

    loader: RandonneurDataLoader
    label: str = "generic-brightway-units-normalization"
    provenance: str = "randonneur.generic-brightway-units-normalization"

    def read(self) -> list[UnitAlias]:
        pkg = self.loader.load(self.label)
        out: list[UnitAlias] = []
        for entry in pkg.get("replace", []):
            src_unit = str(entry.get("source", {}).get("unit", "")).strip()
            tgt_unit = str(entry.get("target", {}).get("unit", "")).strip()
            if src_unit and tgt_unit:
                out.append(
                    UnitAlias(alias=src_unit, canonical=tgt_unit, provenance=self.provenance)
                )
        return out


@dataclass(frozen=True)
class RandonneurUnlinkedListSource:
    """``agribalyse-3.1.1-biosphere-ecoinvent-3.8-biosphere`` — known-unmatchable list.

    Same logical content as the on-disk
    ``agribalyse-3.1.1-unlinked-ecoinvent-3.8-biosphere.json`` (the
    Registry uses a different label).
    """

    loader: RandonneurDataLoader
    label: str = "agribalyse-3.1.1-biosphere-ecoinvent-3.8-biosphere"
    tier: Tier = Tier.UNMATCHABLE
    provenance: str = "randonneur.agribalyse-3.1.1-unlinked-ecoinvent-3.8-biosphere"

    def read(self) -> list[Mapping]:
        if not self.loader.has(self.label):
            return []
        pkg = self.loader.load(self.label)
        out: list[Mapping] = []
        for i, entry in enumerate(pkg.get("update", []) + pkg.get("replace", [])):
            src = entry.get("source", {})
            src_ctx = tuple(src.get("context", ()))
            out.append(
                Mapping(
                    source_kind=SourceKind.AGB_FLOW,
                    source_name=str(src.get("name", "")).strip(),
                    source_unit=str(src.get("unit", "")).strip(),
                    source_context=src_ctx,
                    source_top_bucket=Bucket.from_categories(src_ctx),
                    priority_tier=self.tier,
                    provenance=self.provenance,
                    provenance_row=str(i),
                    is_unmatchable=True,
                    notes=str(entry.get("comment", "")).strip(),
                )
            )
        return out
