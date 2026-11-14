"""``TechnosphereMatcher`` — registry-driven, parquet-backed, no SQLite.

Replaces the legacy bw2io ``match_database(ei_db, ...)`` call chain with
direct lookups into ``EcoinventCatalog``. The match passes are:

1. Apply ``ChangeElectricityUnitMjToKwh`` to harmonise electricity units.
2. Strict 4-field match (name, unit, location, reference product).
3. Registry-driven unit conversions for unlinked technosphere edges.
4. Strict 4-field match again (now that units have been harmonised).
5. Custom-fixes JSON migration via ``randonneur``, then strict match.
6. Relaxed 3-field match (drop ``reference product``) — catches stale or
   capitalised refprod fields that the strict pass missed.
7. SimaPro-form names (``Display {LOC}| activity | Cut-off, U - Copied
   from Ecoinvent U``) resolved on the catalog by ``SimaproFormResolver``:
   the references to embedded ecoinvent copies that no datapackage
   relabels. Same unit required, unique activity required.

Each strict / relaxed match resolves to ``(database, code)`` only when
the catalog returns exactly one activity for the key — same conservative
behaviour as ``bw2io.match_database``, which leaves ambiguous keys
unlinked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import randonneur as rn

from config import Settings
from core.logging import Logging
from matching.audit import DropTallyTracker
from matching.ecoinvent_catalog import EcoinventCatalog
from matching.simapro_form_resolver import SimaproFormResolver
from readers import JsonOrGzJsonReader
from registry import MappingRegistry
from transforms.strategies.units import ChangeElectricityUnitMjToKwh, RescaleExchange


@dataclass(frozen=True)
class TechnosphereMatchStats:
    n_total: int
    n_linked: int
    n_unit_conversions_applied: int
    custom_fixes_applied: int
    n_simapro_form_linked: int = 0
    n_simapro_form_unit_mismatch: int = 0


@dataclass
class TechnosphereMatcher:
    settings: Settings
    registry: MappingRegistry
    catalog: EcoinventCatalog
    drops: DropTallyTracker
    json: JsonOrGzJsonReader = None  # type: ignore[assignment]
    redirects: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    """ecoinvent activity → rebuilt unit copy (``UnitCopyRebuilder``). Applies to
    SimaPro-form references only: those are the inputs that named the copy."""
    name_redirects: dict[str, tuple[str, str]] = field(default_factory=dict)
    """SimaPro name → rebuilt dataset, for stripped blocks that are not a whole
    ecoinvent activity (ADEME's combustion extracts)."""

    def __post_init__(self) -> None:
        if self.json is None:
            object.__setattr__(self, "json", JsonOrGzJsonReader())

    @property
    def _log(self):
        return Logging.get(__name__)

    def match(self, sp) -> TechnosphereMatchStats:
        """Apply the four-stage technosphere link chain via ``EcoinventCatalog``."""
        ei_db = self.settings.ecoinvent_db_name

        # 1. Electricity units → kWh.
        sp.data = ChangeElectricityUnitMjToKwh()(sp.data)

        # 2. First strict pass.
        self._link_strict(sp.data, ei_db)

        # 3. Registry-driven unit conversions for unlinked technosphere flows.
        n_unit = self._apply_unit_conversions(sp)

        # 4. Strict pass again now that units may have shifted.
        self._link_strict(sp.data, ei_db)

        # 5. Custom-fixes datapackage authored by ``RandonneurPackagesExporter``.
        n_custom = 0
        custom_path = self.settings.paths.custom_technosphere_fixes_json
        if custom_path.exists():
            data = self.json.read(custom_path)
            sp.data = rn.migrate_edges(
                graph=sp.data,
                migrations={"replace": data.get("replace", [])},
                config=rn.MigrationConfig(
                    edges_label="exchanges",
                    edge_filter=lambda x: not x.get("input"),
                ),
            )
            self._link_strict(sp.data, ei_db)
            n_custom = len(data.get("replace", []))

        # 6. Relaxed match (drop reference_product) — catches stale labels.
        self._link_relaxed(sp.data, ei_db)

        # 7. SimaPro-form names resolved on the catalog.
        n_form, n_form_unit = self._link_simapro_form(sp.data, ei_db)

        n_total, n_linked = self._counts(sp.data)
        self._log.info(
            "technosphere.matched",
            n_total=n_total,
            n_linked=n_linked,
            unit_conversions=n_unit,
            custom_fixes=n_custom,
            simapro_form_linked=n_form,
            simapro_form_unit_mismatch=n_form_unit,
        )
        return TechnosphereMatchStats(
            n_total=n_total,
            n_linked=n_linked,
            n_unit_conversions_applied=n_unit,
            custom_fixes_applied=n_custom,
            n_simapro_form_linked=n_form,
            n_simapro_form_unit_mismatch=n_form_unit,
        )

    # ------------------------------------------------------------------
    # Catalog-backed match passes.

    def _link_strict(self, sp_data: list[dict], ei_db: str) -> int:
        """Strict 4-field match: (name, unit, location, reference product)."""
        n = 0
        for ds in sp_data:
            for exc in ds.get("exchanges", []):
                if not self._is_unlinked_technosphere(exc):
                    continue
                hit = self.catalog.match_full(
                    ei_db,
                    exc.get("name", ""),
                    exc.get("unit", ""),
                    exc.get("location", ""),
                    exc.get("reference product", ""),
                )
                if hit is not None:
                    exc["input"] = (hit.db, hit.code)
                    n += 1
        return n

    def _link_relaxed(self, sp_data: list[dict], ei_db: str) -> int:
        """3-field match without ``reference product``."""
        n = 0
        for ds in sp_data:
            for exc in ds.get("exchanges", []):
                if not self._is_unlinked_technosphere(exc):
                    continue
                hit = self.catalog.match_relaxed(
                    ei_db,
                    exc.get("name", ""),
                    exc.get("unit", ""),
                    exc.get("location", ""),
                )
                if hit is not None:
                    exc["input"] = (hit.db, hit.code)
                    n += 1
        return n

    def _link_simapro_form(self, sp_data: list[dict], ei_db: str) -> tuple[int, int]:
        """Link exchanges still named in SimaPro form. Returns ``(linked, unit_mismatch)``."""
        resolver = SimaproFormResolver(catalog=self.catalog, ei_db=ei_db)
        n_linked = 0
        n_unit_mismatch = 0
        for ds in sp_data:
            for exc in ds.get("exchanges", []):
                if not self._is_unlinked_technosphere(exc):
                    continue
                named = self.name_redirects.get(exc.get("name", ""))
                if named is not None:
                    exc["input"] = named
                    n_linked += 1
                    continue
                hit = resolver.resolve(exc.get("name", ""))
                if hit is None:
                    continue
                if (exc.get("unit") or "").strip().lower() != hit.unit.strip().lower():
                    n_unit_mismatch += 1
                    continue
                exc["input"] = self.redirects.get((hit.db, hit.code), (hit.db, hit.code))
                exc["name"] = hit.name
                exc["location"] = hit.location
                exc["reference product"] = hit.reference_product
                n_linked += 1
        return n_linked, n_unit_mismatch

    @staticmethod
    def _is_unlinked_technosphere(exc: dict) -> bool:
        return exc.get("type") == "technosphere" and not exc.get("input")

    # ------------------------------------------------------------------
    # Unit conversion (registry-driven).

    def _apply_unit_conversions(self, sp) -> int:
        """Rescale unlinked technosphere exchanges via the registry's unit_conversions."""
        conversions = self.registry.unit_converter
        n = 0
        for proc in sp.data:
            for exc in proc.get("exchanges", []):
                if exc.get("input"):
                    continue
                if exc.get("type") != "technosphere":
                    continue
                src_unit = (exc.get("unit") or "").strip()
                if not src_unit:
                    continue
                for row in conversions.conversions_df.itertuples(index=False):
                    if row.source_unit.lower() != src_unit.lower():
                        continue
                    multiplier = conversions.multiplier(src_unit, row.target_unit)
                    if multiplier is None:
                        continue
                    RescaleExchange.apply(exc, multiplier)
                    exc["unit"] = row.target_unit
                    n += 1
                    break
        return n

    @staticmethod
    def _counts(sp_data: list[dict]) -> tuple[int, int]:
        total = 0
        linked = 0
        for p in sp_data:
            for e in p.get("exchanges", []):
                if e.get("type") == "technosphere":
                    total += 1
                    if "input" in e:
                        linked += 1
        return total, linked
