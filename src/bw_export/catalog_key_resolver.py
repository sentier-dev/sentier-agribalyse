"""``CatalogKeyResolver`` — integer node id → ``(database, code)`` + labels.

The technosphere/biosphere matrices carry only integer ids (63-bit SHA-256
hashes of ``(database, code)`` — see
:meth:`scoring.exchange_frame_builder.ExchangeFrameBuilder.flow_id_for`). A
``bw2data`` project keys every node by ``(database, code)``, so the bw2 export
must recover those string keys. We do it by hashing each catalog row's
``(database, code)`` forward and indexing by the resulting id — the same join
the metadata emitter performs.

Ids absent from the catalogs (the label join is best-effort) fall back to a
synthetic ``(fallback_db, str(id))`` key so every matrix node stays addressable
and the project remains importable; only the human-readable label is lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from bw_export.bw_node_types import ActivityMeta, BioMeta
from scoring.exchange_frame_builder import ExchangeFrameBuilder

_ACTIVITY_FALLBACK_DB = "agribalyse-ef31"
_BIOSPHERE_FALLBACK_DB = "agribalyse-ef31-biosphere"
_CORRECTION_DB = "agribalyse-ef31-correction"


@dataclass
class CatalogKeyResolver:
    """Builds id→key maps once, then resolves O(1) per node.

    ``synthetic_flow_codes`` maps a synthetic AWARE correction flow id to a
    stable code (e.g. ``"aware-<method-slug>"``) so those flows get meaningful,
    collision-free keys instead of an opaque integer.
    """

    product_catalog: pd.DataFrame
    biosphere_catalog: pd.DataFrame
    ecoinvent_catalog: pd.DataFrame
    synthetic_flow_codes: dict[int, str] = field(default_factory=dict)
    # Preferred, divergence-free source: one row per technosphere column
    # keyed by ``activity_id`` (the column id itself), covering agribalyse,
    # ecoinvent, and Allocator multifunctional-split columns. When present
    # it is authoritative and the ``(database, code)``-hash fallback below is
    # skipped. Absent for bundles built before the activity_catalog fix.
    activity_catalog: pd.DataFrame | None = None

    _activities: dict[int, ActivityMeta] = field(default_factory=dict, init=False)
    _bioflows: dict[int, BioMeta] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._activities = self._index_activities()
        self._bioflows = self._index_bioflows()

    def activity(self, col_id: int) -> ActivityMeta:
        meta = self._activities.get(col_id)
        if meta is not None:
            return meta
        return ActivityMeta(
            key=(_ACTIVITY_FALLBACK_DB, str(col_id)),
            name=f"activity {col_id}",
            unit="",
            location="",
            reference_product="",
        )

    def biosphere(self, bioflow_id: int) -> BioMeta:
        meta = self._bioflows.get(bioflow_id)
        if meta is not None:
            return meta
        code = self.synthetic_flow_codes.get(bioflow_id)
        if code is not None:
            return BioMeta(
                key=(_CORRECTION_DB, code),
                name=code,
                categories=("correction",),
                unit="",
                is_synthetic_correction=True,
            )
        return BioMeta(
            key=(_BIOSPHERE_FALLBACK_DB, str(bioflow_id)),
            name=f"flow {bioflow_id}",
            categories=(),
            unit="",
            is_synthetic_correction=False,
        )

    def _index_activities(self) -> dict[int, ActivityMeta]:
        out: dict[int, ActivityMeta] = {}
        # Authoritative path: the activity_catalog is keyed by the column id
        # itself, so every technosphere column resolves directly — including
        # Allocator synthetic splits, whose ids are not ``flow_id_for`` hashes
        # and therefore can't be recovered by the ``(database, code)`` join.
        ac = self.activity_catalog
        if ac is not None and not ac.empty and "activity_id" in ac.columns:
            for row in ac.itertuples(index=False):
                out[int(row.activity_id)] = ActivityMeta(
                    key=(
                        self._clean(getattr(row, "database", "")),
                        self._clean(getattr(row, "code", "")),
                    ),
                    name=self._clean(getattr(row, "name", "")),
                    unit=self._clean(getattr(row, "unit", "")),
                    location=self._clean(getattr(row, "location", "")),
                    reference_product=self._clean(getattr(row, "reference_product", "")),
                )
            return out
        # Fallback for bundles built before the activity_catalog fix:
        # Foreground (Agribalyse) products carry their own name + unit in the
        # product catalog; enrich location/reference_product from ecoinvent by
        # code where available.
        cat = self.product_catalog
        if cat is not None and not cat.empty:
            eco = self._ecoinvent_by_code()
            for row in cat.itertuples(index=False):
                database = self._clean(getattr(row, "database", ""))
                code = self._clean(getattr(row, "code", ""))
                if not database or not code:
                    continue
                cid = ExchangeFrameBuilder.flow_id_for((database, code))
                enrich = eco.get(code, {})
                out[cid] = ActivityMeta(
                    key=(database, code),
                    name=self._clean(getattr(row, "name", "")),
                    unit=self._clean(getattr(row, "unit", "")),
                    location=self._clean(enrich.get("location", "")),
                    reference_product=self._clean(enrich.get("reference_product", "")),
                )
        # Background (ecoinvent) activities are absent from the product catalog
        # — their rows there are null — so index them straight from the
        # ecoinvent catalog. Without this, every technosphere column that points
        # at an ecoinvent process falls back to an opaque "activity <id>" with
        # no unit (and AB then shows empty Unit/Product on the exchange rows,
        # since it derives those from the linked activity). Keyed by the same
        # forward hash of (database, code); ``setdefault`` lets any real product
        # catalog entry win over the background.
        eco_cat = self.ecoinvent_catalog
        if eco_cat is not None and not eco_cat.empty:
            for row in eco_cat.itertuples(index=False):
                database = self._clean(getattr(row, "database", ""))
                code = self._clean(getattr(row, "code", ""))
                if not database or not code:
                    continue
                cid = ExchangeFrameBuilder.flow_id_for((database, code))
                out.setdefault(
                    cid,
                    ActivityMeta(
                        key=(database, code),
                        name=self._clean(getattr(row, "name", "")),
                        unit=self._clean(getattr(row, "unit", "")),
                        location=self._clean(getattr(row, "location", "")),
                        reference_product=self._clean(getattr(row, "reference_product", "")),
                    ),
                )
        return out

    def _index_bioflows(self) -> dict[int, BioMeta]:
        cat = self.biosphere_catalog
        if cat is None or cat.empty:
            return {}
        out: dict[int, BioMeta] = {}
        for row in cat.itertuples(index=False):
            database = str(getattr(row, "database", ""))
            code = str(getattr(row, "code", ""))
            if not database or not code:
                continue
            bid = ExchangeFrameBuilder.flow_id_for((database, code))
            out[bid] = BioMeta(
                key=(database, code),
                name=str(getattr(row, "name", "") or ""),
                categories=self._as_categories(getattr(row, "categories", None)),
                unit=str(getattr(row, "unit", "") or ""),
                is_synthetic_correction=False,
            )
        return out

    def _ecoinvent_by_code(self) -> dict[str, dict[str, str]]:
        eco = self.ecoinvent_catalog
        if eco is None or eco.empty or "code" not in eco.columns:
            return {}
        keep = [c for c in ("code", "location", "reference_product") if c in eco.columns]
        slim = eco[keep].drop_duplicates("code")
        return {str(r["code"]): r for _, r in slim.iterrows()}

    @staticmethod
    def _clean(value: object) -> str:
        """Coerce a catalog cell to a plain string, mapping pandas nulls to ``""``.

        ``str(nan)``/``str(pd.NA)`` would otherwise yield the truthy strings
        ``"nan"``/``"<NA>"`` and slip past the emptiness guards.
        """
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass  # non-scalar (list/tuple) — not null
        return str(value)

    @staticmethod
    def _as_categories(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        text = str(value).strip()
        if not text:
            return ()
        # Catalogs sometimes store categories as a "/"- or "::"-joined string.
        for sep in ("::", "/", "|"):
            if sep in text:
                return tuple(p.strip() for p in text.split(sep) if p.strip())
        return (text,)
