"""Source ingesters for the AGB-specific JSON files in ``source/``.

These are the agribalyse-3.2-* files we ship locally:

* ``agribalyse-3.2-ecoinvent-3.10-biosphere.json`` → biosphere mappings (tier 4)
* ``agribalyse-3.2-correct-ecoinvent-edge-labels.json`` → edge_label_corrections
* ``agribalyse-3.2-delete-aggregated-ecoinvent-{processes,products}.json`` → deletions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from domain import Bucket, Deletion, EdgeLabelCorrection, Mapping, SourceKind, Tier
from readers import JsonOrGzJsonReader


@dataclass(frozen=True)
class AGB32FlowmapperBiosphereSource:
    """``agribalyse-3.2-ecoinvent-3.10-biosphere.json`` — Flowmapper-generated
    SimaPro-9 → ecoinvent-3.10-biosphere mappings (~4 039 rows, ``update`` key).

    UUID-pinned (``target.identifier`` always present); ``target_db`` is left
    empty because the JSON only declares the identifier. ``BiosphereMatcher``
    walks the configured biosphere DBs (configured bio + EF + biosphere3) to
    resolve the UUID, both in ``_resolve_target`` (final pick) and in
    ``_sub_rank`` (sub-compartment-aware tie-break). Wired to tier
    ``AGRIBALYSE_EI_BIOSPHERE`` (4).
    """

    path: Path
    json: JsonOrGzJsonReader = None  # type: ignore[assignment]
    tier: Tier = Tier.AGRIBALYSE_EI_BIOSPHERE
    provenance: str = "agribalyse.ecoinvent-3.10-biosphere-flowmapper"

    def __post_init__(self) -> None:
        if self.json is None:
            object.__setattr__(self, "json", JsonOrGzJsonReader())

    def read(self) -> list[Mapping]:
        if not Path(self.path).exists():
            return []
        data = self.json.read(self.path)
        out: list[Mapping] = []
        for i, entry in enumerate(data.get("update", [])):
            src = entry.get("source", {})
            tgt = entry.get("target", {})
            src_ctx = tuple(src.get("context", ()))
            target_code = str(tgt.get("identifier", "")).strip()
            target_name = str(tgt.get("name", "")).strip()
            if not target_code and not target_name:
                continue
            out.append(
                Mapping(
                    source_kind=SourceKind.AGB_FLOW,
                    source_name=str(src.get("name", "")).strip(),
                    source_unit=str(src.get("unit", "")).strip(),
                    source_context=src_ctx,
                    source_top_bucket=Bucket.from_categories(src_ctx),
                    target_db="",
                    target_code=target_code,
                    target_name=target_name,
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
class AgbDeleteAggregatedSource:
    """Combines the two ``delete-aggregated-ecoinvent-*.json`` files into ``Deletion`` rows.

    Fix 1.m: the original linker matches by name only, which can collide.
    The registry stores both ``name`` and ``code`` so the matcher can
    tighten matching when both are present.
    """

    processes_path: Path
    products_path: Path
    json: JsonOrGzJsonReader = None  # type: ignore[assignment]
    provenance: str = "agribalyse.delete-aggregated-ecoinvent"

    def __post_init__(self) -> None:
        if self.json is None:
            object.__setattr__(self, "json", JsonOrGzJsonReader())

    def read(self) -> list[Deletion]:
        out: list[Deletion] = []
        for path, kind in (
            (self.processes_path, "process"),
            (self.products_path, "product"),
        ):
            if not Path(path).exists():
                continue
            data = self.json.read(path)
            for entry in data.get("delete", []):
                src = entry.get("source", {})
                name = str(src.get("name", "")).strip()
                code = str(src.get("identifier", "")).strip() or str(src.get("code", "")).strip()
                if not name and not code:
                    continue
                out.append(
                    Deletion(
                        name=name,
                        code=code or None,
                        kind=kind,
                        provenance=f"{self.provenance}-{kind}s",
                    )
                )
        return out


@dataclass(frozen=True)
class AgbEdgeLabelsSource:
    """``agribalyse-3.2-correct-ecoinvent-edge-labels.json`` (~2 123 rows)."""

    path: Path
    json: JsonOrGzJsonReader = None  # type: ignore[assignment]
    provenance: str = "agribalyse.correct-ecoinvent-edge-labels"

    def __post_init__(self) -> None:
        if self.json is None:
            object.__setattr__(self, "json", JsonOrGzJsonReader())

    def read(self) -> list[EdgeLabelCorrection]:
        if not Path(self.path).exists():
            return []
        data = self.json.read(self.path)
        out: list[EdgeLabelCorrection] = []
        for entry in data.get("replace", []):
            src = entry.get("source", {})
            tgt = entry.get("target", {})
            src_name = str(src.get("name", "")).strip()
            tgt_name = str(tgt.get("name", "")).strip()
            if not src_name or not tgt_name:
                continue
            out.append(
                EdgeLabelCorrection(
                    source_name=src_name,
                    target_name=tgt_name,
                    edge_type=str(src.get("type", "")).strip(),
                    categories=tuple(src.get("categories", ())),
                    provenance=self.provenance,
                )
            )
        return out
