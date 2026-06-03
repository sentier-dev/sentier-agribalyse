"""``CuratedOverridesSource`` and ``LlmReviewedSource``.

The curated overrides JSON is the small, dated file that replaces the
residual portion of the legacy hardcoded ``BIOSPHERE_SYNONYMS``. Each
entry chooses its own tier (typically 1 or 11). The LLM-reviewed xlsx
contributes tier-10 fill-only rows.

``ExtraUnitConversionSource`` carries the small set of land/transport unit
conversions (m² → hectare, m → km) the upstream randonneur datapackage
doesn't ship; without these the GLO market-for tillage / fertilising tech
edges can't match ecoinvent (unit field mismatch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.parquet_cache import ParquetCache
from domain import Bucket, Mapping, SourceKind, Tier, UnitConversion
from readers import JsonReader, XlsxReader


@dataclass(frozen=True)
class CuratedOverridesSource:
    """Read ``source/curated_overrides.json``.

    Schema (each entry)::

        {
            "source_name": "1,2-Dichloropropane",
            "source_unit": "kg",
            "source_context": ["Emissions to water", "river"],
            "target_db": "biosphere3",
            "target_code": "<uuid>",          # optional, else resolved by name
            "target_name": "Propane, 1,2-dichloro-",
            "target_unit": "kg",
            "unit_conversion": 1.0,
            "tier": "CURATED_TARGETED",       # or "CURATED_SYNONYM_FALLBACK"
            "is_unmatchable": false,          # optional; true → flow is acknowledged dead-end
            "notes": "fix 1,2-DCP → Propane regression",
        }

    Set ``is_unmatchable: true`` for AGB flows that have no defensible target
    in any biosphere DB (e.g. AGB-only resource concepts like "Inert rock").
    Those rows are routed to ``unmatchable.parquet`` so the matcher counts
    them as recognised dead-ends instead of unexplained residuals.

    The file is allowed to not exist (then the source contributes zero rows).
    The same JSON is read twice by the builder — once with
    ``select_unmatchable=False`` for ``mappings_biosphere``, once with
    ``select_unmatchable=True`` for ``unmatchable``.
    """

    path: Path
    json: JsonReader = field(default_factory=JsonReader)
    provenance: str = "curated_overrides"
    select_unmatchable: bool = False

    def read(self) -> list[Mapping]:
        if not Path(self.path).exists():
            return []
        data = self.json.read(self.path)
        out: list[Mapping] = []
        for i, entry in enumerate(data.get("entries", [])):
            is_unmatchable = bool(entry.get("is_unmatchable", False))
            if is_unmatchable != self.select_unmatchable:
                continue
            tier_name = str(entry.get("tier", "CURATED_TARGETED")).strip()
            try:
                tier = Tier[tier_name]
            except KeyError:
                tier = Tier.CURATED_TARGETED
            ctx = tuple(entry.get("source_context", ()))
            out.append(
                Mapping(
                    source_kind=SourceKind.AGB_FLOW,
                    source_name=str(entry["source_name"]).strip(),
                    source_unit=str(entry.get("source_unit", "")).strip(),
                    source_context=ctx,
                    source_top_bucket=Bucket.from_categories(ctx),
                    source_cas=str(entry.get("source_cas") or "") or None,
                    target_db=str(entry.get("target_db", "")).strip(),
                    target_code=str(entry.get("target_code", "")).strip(),
                    target_name=str(entry.get("target_name", "")).strip(),
                    target_unit=str(entry.get("target_unit", "")).strip(),
                    unit_conversion=float(entry.get("unit_conversion") or 1.0),
                    priority_tier=Tier.UNMATCHABLE if is_unmatchable else tier,
                    provenance=self.provenance,
                    provenance_row=str(i),
                    is_unmatchable=is_unmatchable,
                    notes=str(entry.get("notes", "")).strip(),
                )
            )
        return out


@dataclass(frozen=True)
class ExtraUnitConversionSource:
    """Read ``source/agribalyse-3.2-extra-unit-conversions.json``.

    Plugs the gap left by upstream ``generic-brightway-unit-conversions``: AGB
    SimaPro CSV emits tech edges in m²/m for land/transport markets that
    ecoinvent stores in hectare/km. The legacy linker hardcoded these; here
    they live next to the other registry sources.

    Schema::

        {
          "entries": [
            {"source_unit": "square meter", "target_unit": "hectare", "multiplier": 0.0001}
          ]
        }
    """

    path: Path
    json: JsonReader = field(default_factory=JsonReader)
    provenance: str = "agribalyse-3.2-extra-unit-conversions"

    def read(self) -> list[UnitConversion]:
        if not Path(self.path).exists():
            return []
        data = self.json.read(self.path)
        out: list[UnitConversion] = []
        for entry in data.get("entries", []):
            src = str(entry.get("source_unit", "")).strip()
            tgt = str(entry.get("target_unit", "")).strip()
            mult = entry.get("multiplier")
            if not src or not tgt or mult is None:
                continue
            out.append(
                UnitConversion(
                    source_unit=src,
                    target_unit=tgt,
                    multiplier=float(mult),
                    provenance=self.provenance,
                )
            )
        return out


@dataclass(frozen=True)
class LlmReviewedSource:
    """Read the LLM-reviewed xlsx (only ``decision='accept'`` rows).

    Tier 10, fill-only, gated by ``Settings.apply_llm_overrides``. The
    matcher applies the gate at match time; the registry always carries
    the rows so the data is auditable.
    """

    path: Path
    cache: ParquetCache
    target_db: str = "biosphere3"
    tier: Tier = Tier.LLM_OVERRIDES
    provenance: str = "llm.biosphere-residuals-reviewed"

    def read(self) -> list[Mapping]:
        if not Path(self.path).exists():
            return []
        xlsx = XlsxReader(cache=self.cache)
        df = xlsx.read(self.path)
        accepted = df[df.get("decision", "") == "accept"]
        out: list[Mapping] = []
        for idx, r in accepted.iterrows():
            target_name = str(r.get("target_name", "")).strip()
            if not target_name:
                continue
            source_top = str(r.get("source_top_cat") or "").strip()
            source_sub = str(r.get("source_sub_cat") or "").strip()
            source_cats = tuple(c for c in (source_top, source_sub) if c)
            multiplier = r.get("multiplier")
            try:
                conv = float(multiplier) if multiplier else 1.0
            except (TypeError, ValueError):
                conv = 1.0
            out.append(
                Mapping(
                    source_kind=SourceKind.AGB_FLOW,
                    source_name=str(r.get("source_name", "")).strip(),
                    source_unit=str(r.get("source_unit", "")).strip(),
                    source_context=source_cats,
                    # LLM rows are name synonyms — they apply regardless of the
                    # exchange's compartment. Storing them under UNSPECIFIED lets
                    # TieredNameBucketIndex.lookup surface them for any actual
                    # source bucket; _resolve_target then locates the target in
                    # the exchange's actual compartment.
                    source_top_bucket=Bucket.UNSPECIFIED,
                    target_db=self.target_db,
                    target_code="",  # resolved at match time against bio3 by (name, bucket)
                    target_name=target_name,
                    target_unit=str(r.get("target_unit", "")).strip(),
                    unit_conversion=conv,
                    priority_tier=self.tier,
                    provenance=self.provenance,
                    provenance_row=str(idx),
                    notes=str(r.get("reviewer_note", "")).strip(),
                )
            )
        return out
