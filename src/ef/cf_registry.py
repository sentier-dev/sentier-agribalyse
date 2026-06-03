"""``MethodCfRegistryBuilder`` / ``MethodCfRegistryLoader`` — per-method CF parquets.

Reads two source-file inputs (REFACTOR_FINAL F6):

(a) augmented CFs from ``EfCfTable.global_cfs`` for the EF method name
    in ``EF_METHOD_MAP`` — keyed on ``(ef_db_name, FLOW_uuid)``.
(b) inherited CFs from ``source/ef-v31-methods.json`` — pre-snapshotted
    biosphere3 entries (db != ef_db_name) the legacy
    ``EfMethodAugmenter`` preserved via its ``keep`` filter.

No bw2data, no SQLite. Determinism + atomic write per file
(``<path>.partial`` + ``os.replace``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pandas as pd

from config import Settings
from core.logging import Logging
from core.parquet_io import ParquetAtomicWriter
from ef.cf_simapro_filter import SimaProCfFilter
from ef.cf_table import EfCfTable
from ef.regional_cf import RegionalCfRegistryBuilder
from ef.regional_water_cf import RegionalWaterCfRegistryBuilder
from ef.sp_regional_water_cf_loader import SpRegionalWaterCfLoader
from ef.water_resource_augmenter import WaterResourceCfAugmenter
from ef.water_use_bio3_bridge import WaterUseBio3Bridge
from scoring.method_slug import MethodSlug


@dataclass(frozen=True)
class MethodCfRegistryBuilder:
    """Build per-method CF parquets from the EF source CSV + the bw2data snapshot.

    Optionally intersects the inherited CFs with the SimaPro EF 3.1
    (adapted) reference method ADEME used for the AGRIBALYSE 3.2
    synthesis (``simapro_filter`` argument). When set, every
    biosphere3-coded inherited CF is dropped if SimaPro doesn't carry
    a matching ``(name, top compartment)`` for that method, with a
    biosphere-flowmap synonym fallback. See ``SimaProCfFilter`` and
    FIX_DATA.md § 1.1 for why this is needed (Kaolin / Water[air] /
    pesticide CFs were inherited from ecoinvent's LCIA xlsx but are
    *not* in JRC's authoritative reference dataset).
    """

    settings: Settings
    cf_table: EfCfTable
    simapro_filter: SimaProCfFilter | None = None
    water_resource_augmenter: WaterResourceCfAugmenter | None = None
    water_use_bio3_bridge: WaterUseBio3Bridge | None = None
    regional_water_cf_builder: RegionalWaterCfRegistryBuilder | None = None
    regional_cf_builder: RegionalCfRegistryBuilder | None = None
    sp_regional_water_cf_loader: SpRegionalWaterCfLoader | None = None

    # Method key (last two components: category, indicator) that the
    # ``water_resource_augmenter`` applies to. The other 17 methods do
    # not receive augmentation rows.
    WATER_USE_KEY: ClassVar[tuple[str, str]] = (
        "water use",
        "user deprivation potential (deprivation-weighted water consumption)",
    )

    EF_METHOD_MAP: ClassVar[Mapping[tuple[str, str], str]] = MappingProxyType(
        {
            ("climate change", "global warming potential (GWP100)"): "Climate change",
            (
                "climate change: biogenic",
                "global warming potential (GWP100)",
            ): "Climate change-Biogenic",
            (
                "climate change: fossil",
                "global warming potential (GWP100)",
            ): "Climate change-Fossil",
            (
                "climate change: land use and land use change",
                "global warming potential (GWP100)",
            ): "Climate change-Land use and land use change",
            ("ozone depletion", "ozone depletion potential (ODP)"): "Ozone depletion",
            (
                "ionising radiation: human health",
                "human exposure efficiency relative to u235",
            ): "Ionising radiation, human health",
            (
                "photochemical oxidant formation: human health",
                "tropospheric ozone concentration increase",
            ): "Photochemical ozone formation - human health",
            ("particulate matter formation", "impact on human health"): "EF-particulate Matter",
            (
                "human toxicity: non-carcinogenic",
                "comparative toxic unit for human (CTUh)",
            ): "Human toxicity, non-cancer",
            (
                "human toxicity: carcinogenic",
                "comparative toxic unit for human (CTUh)",
            ): "Human toxicity, cancer",
            ("acidification", "accumulated exceedance (AE)"): "Acidification",
            (
                "eutrophication: freshwater",
                "fraction of nutrients reaching freshwater end compartment (P)",
            ): "Eutrophication, freshwater",
            (
                "eutrophication: marine",
                "fraction of nutrients reaching marine end compartment (N)",
            ): "Eutrophication marine",
            (
                "eutrophication: terrestrial",
                "accumulated exceedance (AE)",
            ): "Eutrophication, terrestrial",
            (
                "ecotoxicity: freshwater",
                "comparative toxic unit for ecosystems (CTUe)",
            ): "Ecotoxicity, freshwater",
            ("land use", "soil quality index"): "Land use",
            (
                "water use",
                "user deprivation potential (deprivation-weighted water consumption)",
            ): "Water use",
            (
                "energy resources: non-renewable",
                "abiotic depletion potential (ADP): fossil fuels",
            ): "Resource use, fossils",
            (
                "material resources: metals/minerals",
                "abiotic depletion potential (ADP): elements (ultimate reserves)",
            ): "Resource use, minerals and metals",
        }
    )

    INDEX_VERSION: ClassVar[int] = 1

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Public entry point.

    def build(self) -> Path:
        s = self.settings
        s.paths.ensure_runtime_dirs()
        out_dir = s.paths.registry_method_cfs_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ef_cfs_by_method = self._ef_cfs_by_method_name()
        snapshot = self._load_methods_snapshot()
        ef_db = snapshot.get("ef_db_name", s.ef_db_name)

        index_entries: list[dict] = []
        n_methods_skipped_unmapped = 0
        n_dropped_total = 0
        for _slug_label, entry in sorted(snapshot.get("methods", {}).items()):
            m_key = tuple(entry["key"])
            method_name = self.EF_METHOD_MAP.get((m_key[2], m_key[3]))
            if method_name is None:
                n_methods_skipped_unmapped += 1
                continue
            cfs = self._cfs_for_method(
                m_key=m_key,
                method_name=method_name,
                ef_cfs_by_method=ef_cfs_by_method,
                inherited_cfs=entry.get("inherited_cfs", []),
                ef_db=ef_db,
            )
            if (
                self.water_resource_augmenter is not None
                and (m_key[2], m_key[3]) == self.WATER_USE_KEY
            ):
                cfs = self._merge_augmenter_rows(cfs, self.water_resource_augmenter.cf_rows())
            if (
                self.water_use_bio3_bridge is not None
                and (m_key[2], m_key[3]) == self.WATER_USE_KEY
            ):
                # Curated EF v3.1 (adapted) bio3 CFs for water-use.
                # Mirrors ADEME's SimaPro reference: +42.95 on natural-
                # resource water inputs, -42.95 on water-compartment
                # releases. Replaces the disabled augment_rows path
                # whose name-match unit mismatch caused 337x overshoot.
                cfs = self._merge_augmenter_rows(cfs, self.water_use_bio3_bridge.cf_rows())
            if (
                self.sp_regional_water_cf_loader is not None
                and (m_key[2], m_key[3]) == self.WATER_USE_KEY
            ):
                # Per-region AWARE deprivation CFs for the synthetic
                # ``<base_uuid>@<region>`` codes the matcher emits.
                # Implements step 2 of the regional-flow-mapping spec:
                # each linked regional water flow now has its own matrix
                # row and its own CF, matching SimaPro's per-country
                # deprivation values. See
                # docs/superpowers/specs/2026-05-22-regional-flow-mapping-design.md.
                cfs = self._merge_augmenter_rows(
                    cfs,
                    self._regional_water_cf_rows(),
                )
            n_dropped = 0
            if self.simapro_filter is not None:
                cfs, n_dropped = self.simapro_filter.filter_rows(our_key=m_key, rows=cfs)
                # NB: ``augment_rows`` is intentionally NOT called. Adding
                # SimaPro CFs to bio3 codes by name match misbehaves on
                # our hybrid AGB-system + ecoinvent-unit matrix: SimaPro's
                # water-use CF of 42.95 m³ depriv./m³ for "Water, lake
                # [Raw]" applied to ecoinvent's identically-named
                # extraction flow over-counts catastrophically (water-use
                # ratio 1.2x -> 337x when augmentation was on, because
                # ecoinvent activities consume huge cooling-water volumes
                # whose CFs are normally baked into AGB system processes
                # with regional scaling). The filter remains: drop CFs
                # SimaPro doesn't characterise (Kaolin, Water[air], stray
                # pesticides). Augmentation is preserved as ``augment_rows``
                # on the class for ad-hoc analysis but not invoked here.
                n_dropped_total += n_dropped
            slug = MethodSlug.encode(m_key)
            self._write_method_parquet(out_dir / slug / "cfs.parquet", cfs)
            n_regional = 0
            if (m_key[2], m_key[3]) == self.WATER_USE_KEY:
                # Water-use keeps its specialised |CF| + sign(global)
                # convention — see ``RegionalWaterCfRegistryBuilder``.
                if self.regional_water_cf_builder is not None:
                    regional_rows = self.regional_water_cf_builder.build_rows(cfs)
                    n_regional = len(regional_rows)
                    self._write_regional_parquet(
                        out_dir / slug / "regional_cfs.parquet",
                        regional_rows,
                    )
            elif self.regional_cf_builder is not None:
                # All other methods consult the builder's allowlist. Per-(flow,
                # location) JRC CF is applied verbatim with the bw2io sign
                # preserved. Skip the write when the method has no regional
                # rows so we don't litter the registry with empty sidecars.
                regional_rows = self.regional_cf_builder.build_rows(
                    method_key=(m_key[2], m_key[3]),
                    method_name=method_name,
                    global_cfs=cfs,
                )
                n_regional = len(regional_rows)
                if n_regional > 0:
                    self._write_regional_parquet(
                        out_dir / slug / "regional_cfs.parquet",
                        regional_rows,
                    )
            index_entries.append(
                {
                    "key": list(m_key),
                    "slug": slug,
                    "n_cfs": len(cfs),
                    "n_dropped_by_simapro_filter": n_dropped,
                    "n_regional_cfs": n_regional,
                }
            )

        index_entries.sort(key=lambda e: e["slug"])
        self._write_index(s.paths.registry_method_cfs_index, index_entries)
        self._log.info(
            "ef.method_cfs.registry.built",
            dir=str(out_dir),
            n_methods=len(index_entries),
            n_unmapped=n_methods_skipped_unmapped,
            n_dropped_by_simapro_filter=n_dropped_total,
        )
        return out_dir

    # ------------------------------------------------------------------
    # Snapshot loader.

    def _load_methods_snapshot(self) -> dict[str, Any]:
        path = self.settings.paths.ef_methods_snapshot_json
        if not path.exists():
            raise FileNotFoundError(
                f"EF v3.1 method snapshot not found: {path}. "
                f"Snapshot via REFACTOR_FINAL phase F6 (one-time bootstrap)."
            )
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # CF assembly.

    def _ef_cfs_by_method_name(self) -> dict[str, list[tuple[str, str, float]]]:
        """``method_name → [(database, code, amount), ...]`` from the source CSV."""
        ef_db = self.settings.ef_db_name
        out: dict[str, list[tuple[str, str, float]]] = {}
        for _, r in self.cf_table.global_cfs.iterrows():
            uuid = r["FLOW_uuid"]
            if not isinstance(uuid, str) or not uuid:
                continue
            out.setdefault(r["LCIAMethod_name"], []).append(
                (ef_db, str(uuid), float(r["CF_global"]))
            )
        return out

    def _cfs_for_method(
        self,
        *,
        m_key: tuple[str, str, str, str],
        method_name: str,
        ef_cfs_by_method: dict[str, list[tuple[str, str, float]]],
        inherited_cfs: list[dict],
        ef_db: str,
    ) -> list[dict]:
        rows: list[dict] = []
        for db, code, amount in ef_cfs_by_method.get(method_name, []):
            rows.append({"database": db, "code": code, "amount": amount})

        for cf in inherited_cfs:
            db = str(cf.get("db", "") or "")
            code = str(cf.get("code", "") or "")
            if not db or not code or db == ef_db:
                continue
            try:
                amount = float(cf.get("amount", 0.0))
            except (TypeError, ValueError):
                continue
            rows.append({"database": db, "code": code, "amount": amount})

        rows.sort(key=lambda r: (r["database"], r["code"]))
        return rows

    # ------------------------------------------------------------------
    # Regional water-use augmenter.

    # Names that bio3 / ecoinvent carry with a categorical suffix
    # (``Water, well, in ground``) but SimaPro's CF table indexes
    # without (``Water, well``). Strip the suffix at lookup time so the
    # (catalog_name, region) → SimaPro CF join hits.
    _BIO3_NAME_TO_SP_BASE: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "Water, well, in ground": "Water, well",
        }
    )

    # bio3 / ecoinvent top compartment → SimaPro compartment label.
    _BIO3_TOP_TO_SP_COMPARTMENT: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "natural resource": "Raw",
            "water": "Water",
            "air": "Air",
            "soil": "Soil",
        }
    )

    # bio3 / ecoinvent water-release sub-compartment → SimaPro sub.
    # The water-use method's SimaPro CF table only covers three release
    # sub-compartments (``(unspecified)``, ``ocean``, ``groundwater,
    # long-term``) plus a partial ``river`` set. Bio3 codes outside
    # these (``[water, fossil well]``, ``[water, ground-]``) have no
    # SimaPro regional CF and resolve to ``None`` (no row emitted).
    _BIO3_SUB_TO_SP_SUB_WATER_RELEASE: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "": "(unspecified)",
            "ocean": "ocean",
            "ground-, long-term": "groundwater, long-term",
            "surface water": "river",
        }
    )

    def _regional_water_cf_rows(self) -> list[dict]:
        """Emit CF rows for every synthetic ``<base>@<region>`` code the
        augmented catalog carries.

        The catalog is read from
        ``settings.paths.registry_biosphere_catalog`` which after
        ``BiosphereCatalogAugmenter`` has run carries one row per
        ``(database, base_code, region)`` actually used by the matrix.
        For each synthetic row we look up SimaPro's CF for ``(catalog
        name, region, top compartment, sub compartment)`` and emit
        ``{"database", "code", "amount"}``. The downstream merge in
        :meth:`_merge_augmenter_rows` keeps the synthetic-side override
        idempotent.
        """
        loader = self.sp_regional_water_cf_loader
        if loader is None:
            return []
        catalog_path = self.settings.paths.registry_biosphere_catalog
        if not catalog_path.exists():
            self._log.warning(
                "ef.method_cfs.regional_water.no_catalog",
                path=str(catalog_path),
            )
            return []
        df = pd.read_parquet(catalog_path)
        if "region" not in df.columns:
            return []
        synth = df[df["region"].astype(str) != ""]
        if synth.empty:
            return []
        rows: list[dict] = []
        n_hit = 0
        n_miss = 0
        n_skipped_release = 0
        for r in synth.itertuples(index=False):
            cats = r.categories
            top_raw = (str(cats[0]) if cats is not None and len(cats) >= 1 else "").strip().lower()
            sp_compartment = self._BIO3_TOP_TO_SP_COMPARTMENT.get(top_raw)
            if sp_compartment is None:
                n_miss += 1
                continue
            # Only emit per-region CFs on the resource (``Raw``) side.
            # Emitting matching negative CFs on the water-release side
            # would let the ~1-5 % inventory imbalance noise on every
            # nominally balanced cooling/turbine activity blow the
            # water-use mean |Δ| to ~70 % (vs. baseline 15.98 %) — the
            # cancellation-amplifier the spec's risk #1 warned about
            # (and the same failure mode that retired the original
            # ``WaterUseBio3Bridge``). Resource-only emission scores
            # NET consumption directly: AGB foreground activities
            # without a return flow consume the resource volume * the
            # per-region CF; activities with both sides see only the
            # resource contribution (slight over-count vs. ADEME for
            # genuinely balanced cooling, but bounded).
            if sp_compartment != "Raw":
                n_skipped_release += 1
                continue
            # Resource side: SimaPro uses ``(unspecified)`` uniformly
            # across the bio3 ``in water`` / ``in ground`` / ``in
            # air`` / ``fossil well`` sub-compartment variants.
            sp_sub = "(unspecified)"
            sp_name = self._BIO3_NAME_TO_SP_BASE.get(str(r.name), str(r.name))
            cf = loader.cf_for(
                base_name=sp_name,
                region=str(r.region),
                compartment=sp_compartment,
                sub_compartment=sp_sub,
            )
            if cf is None:
                n_miss += 1
                continue
            n_hit += 1
            rows.append(
                {
                    "database": str(r.database),
                    "code": str(r.code),
                    "amount": float(cf),
                }
            )
        self._log.info(
            "ef.method_cfs.regional_water.augment",
            n_hit=n_hit,
            n_miss=n_miss,
            n_skipped_release=n_skipped_release,
            n_synthetic_catalog_rows=len(synth),
        )
        return rows

    @staticmethod
    def _merge_augmenter_rows(existing: list[dict], augmented: list[dict]) -> list[dict]:
        """Append augmenter rows that don't already have a CF in ``existing``.

        Augmentation only fills GAPS — if the bw2io snapshot already
        carries a CF for a (database, code) the augmenter would emit,
        the inherited value wins. This protects against double-counting
        the same flow under regional vs global CFs.
        """
        seen: set[tuple[str, str]] = {
            (str(r.get("database", "") or ""), str(r.get("code", "") or "")) for r in existing
        }
        merged = list(existing)
        for row in augmented:
            key = (
                str(row.get("database", "") or ""),
                str(row.get("code", "") or ""),
            )
            if key in seen:
                continue
            merged.append(dict(row))
            seen.add(key)
        merged.sort(key=lambda r: (r["database"], r["code"]))
        return merged

    # ------------------------------------------------------------------
    # I/O helpers — atomic everywhere.

    @staticmethod
    def _write_method_parquet(target: Path, rows: list[dict]) -> None:
        df = pd.DataFrame(rows, columns=["database", "code", "amount"])
        df = df.astype({"database": "string", "code": "string"})
        ParquetAtomicWriter.write(df, target)

    @staticmethod
    def _write_regional_parquet(target: Path, rows: list[dict]) -> None:
        df = RegionalWaterCfRegistryBuilder.to_dataframe(rows)
        ParquetAtomicWriter.write(df, target)

    @classmethod
    def _write_index(cls, target: Path, entries: list[dict]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": cls.INDEX_VERSION, "methods": entries}
        partial = target.with_suffix(target.suffix + ".partial")
        partial.write_text(json.dumps(payload, sort_keys=True, indent=2))
        os.replace(partial, target)


@dataclass(frozen=True)
class MethodCfRegistryLoader:
    """Read ``registry/method_cfs/`` back into ``dict[method_tuple, DataFrame]``."""

    registry_dir: Path

    INDEX_VERSION: ClassVar[int] = 1
    REGIONAL_FILENAME: ClassVar[str] = "regional_cfs.parquet"

    def load_all(self) -> dict[tuple[str, ...], pd.DataFrame]:
        index_path = self.registry_dir / "_index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"method-CFs index not found: {index_path}. "
                f"Run `dds-build-method-cfs-registry` first."
            )
        index = json.loads(index_path.read_text())
        version = index.get("version")
        if version != self.INDEX_VERSION:
            raise ValueError(
                f"unsupported method-CFs index version {version!r} at {index_path}; "
                f"expected {self.INDEX_VERSION}. Re-run `dds-build-method-cfs-registry`."
            )
        out: dict[tuple[str, ...], pd.DataFrame] = {}
        for entry in index.get("methods", []):
            slug = entry["slug"]
            key = tuple(entry["key"])
            cf_path = self.registry_dir / slug / "cfs.parquet"
            if not cf_path.exists():
                raise FileNotFoundError(f"method-CFs parquet missing for {key!r}: {cf_path}")
            out[key] = pd.read_parquet(cf_path)
        return out

    def load_regional_all(self) -> dict[tuple[str, ...], pd.DataFrame]:
        """Return per-method regional CF dataframes — keyed identically to
        ``load_all``. Methods without a ``regional_cfs.parquet`` sidecar
        are simply absent from the returned dict (caller treats them as
        "no regional correction"). Skipping callers can avoid loading
        what doesn't exist by checking the ``n_regional_cfs`` count in
        ``_index.json``.
        """
        index_path = self.registry_dir / "_index.json"
        if not index_path.exists():
            return {}
        index = json.loads(index_path.read_text())
        out: dict[tuple[str, ...], pd.DataFrame] = {}
        for entry in index.get("methods", []):
            slug = entry["slug"]
            key = tuple(entry["key"])
            reg_path = self.registry_dir / slug / self.REGIONAL_FILENAME
            if not reg_path.exists():
                continue
            df = pd.read_parquet(reg_path)
            if not df.empty:
                out[key] = df
        return out
