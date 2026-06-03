"""Biosphere normalisation strategies — lifted from ``bw2io.strategies``.

Pure dict transforms. No bw2data, no bw2io. Each is callable
(``__call__(data)``) so the existing ``StrategyRunner.apply`` can drive it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from transforms.strategies.migrations import MigrationApplier, MigrationStore

# SimaPro → ecoinvent top-level categories. Lifted verbatim from
# ``bw2io.strategies.simapro.SIMAPRO_BIOSPHERE``.
_SIMAPRO_BIOSPHERE: ClassVar = MappingProxyType(
    {
        "Economic issues": "economic",
        "Emissions to air": "air",
        "Emissions to soil": "soil",
        "Emissions to water": "water",
        "Non material emissions": "non-material",
        "Non mat.": "non-material",
        "Resources": "natural resource",
        "Social issues": "social",
        "Economic": "economic",
        "Air": "air",
        "Soil": "soil",
        "Water": "water",
        "Raw": "natural resource",
        "Waste": "waste",
    }
)

# SimaPro → ecoinvent sub-categories.
_SIMAPRO_BIO_SUBCATEGORIES: ClassVar = MappingProxyType(
    {
        "groundwater": "ground-",
        "groundwater, long-term": "ground-, long-term",
        "high. pop.": "urban air close to ground",
        "low. pop.": "non-urban air or from high stacks",
        "low. pop., long-term": "low population density, long-term",
        "stratosphere + troposphere": "lower stratosphere + upper troposphere",
        "river": "surface water",
        "river, long-term": "surface water",
        "lake": "surface water",
    }
)

_UNSPECIFIED: frozenset = frozenset({"unspecified", "(unspecified)", "", None})


# ============================================================================
# Stateless strategies — no constructor args.


@dataclass(frozen=True)
class StripBiosphereExchangeLocations:
    """Drop the ``location`` field from biosphere exchanges (always non-spatial)."""

    name: str = "strip_biosphere_exc_locations"

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            for exc in ds.get("exchanges", []):
                if exc.get("type") == "biosphere" and "location" in exc:
                    del exc["location"]
        return data


@dataclass(frozen=True)
class DropUnspecifiedSubcategories:
    """Drop trailing 'unspecified' / '' / None entries from category tuples.

    Lifted from ``bw2io.strategies.drop_unspecified_subcategories``.
    Mutates dataset-level ``categories`` and exchange-level ``categories``.
    """

    name: str = "drop_unspecified_subcategories"

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            self._strip(ds)
            for exc in ds.get("exchanges", []):
                self._strip(exc)
        return data

    @staticmethod
    def _strip(obj: dict) -> None:
        cats = obj.get("categories")
        if not cats:
            return
        cats = list(cats)
        while cats and cats[-1] in _UNSPECIFIED:
            cats.pop()
        if isinstance(obj.get("categories"), tuple):
            obj["categories"] = tuple(cats)
        else:
            obj["categories"] = cats


@dataclass(frozen=True)
class NormalizeSimaproBiosphereCategories:
    """Map SimaPro top-level categories ('Air', 'Resources', ...) to ecoinvent."""

    name: str = "normalize_simapro_biosphere_categories"

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            for exc in ds.get("exchanges", []):
                if exc.get("type") != "biosphere":
                    continue
                cats = exc.get("categories")
                if not cats:
                    continue
                top = _SIMAPRO_BIOSPHERE.get(cats[0], cats[0])
                if len(cats) > 1:
                    sub = _SIMAPRO_BIO_SUBCATEGORIES.get(cats[1], cats[1])
                    exc["categories"] = (top, sub)
                else:
                    exc["categories"] = (top,)
        return data


# ============================================================================
# Strategies that need a path — load JSON / migration once at construction.


@dataclass(frozen=True)
class NormalizeSimaproBiosphereNames:
    """Map SimaPro flow names to ecoinvent names via the lifted JSON file."""

    json_path: Path
    name: str = "normalize_simapro_biosphere_names"

    def __call__(self, data: list[dict]) -> list[dict]:
        mapping = self._mapping()
        for ds in data:
            for exc in ds.get("exchanges", []):
                if exc.get("type") != "biosphere":
                    continue
                cats = exc.get("categories")
                exc_name = exc.get("name")
                if not cats or not exc_name:
                    continue
                key = (cats[0], exc_name)
                if key in mapping:
                    exc["name"] = mapping[key]
        return data

    def _mapping(self) -> dict[tuple[str, str], str]:
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"simapro-biosphere mapping not found at {self.json_path}. "
                f"Snapshot the bw2io data file via REFACTOR_FINAL phase F3."
            )
        rows = json.loads(self.json_path.read_text())
        return {(r[0], r[1]): r[2] for r in rows}


@dataclass(frozen=True)
class NormalizeBiosphereCategories:
    """Apply the ``biosphere-2-3-categories`` migration to exchanges + datasets."""

    store: MigrationStore
    name: str = "normalize_biosphere_categories"
    lcia: bool = False
    migration_name: str = "biosphere-2-3-categories"

    def __call__(self, data: list[dict]) -> list[dict]:
        applier = MigrationApplier.from_store(self.store, self.migration_name)
        applier.apply_to_exchanges(data)
        if not self.lcia:
            applier.apply_to_datasets(data)
        return data


@dataclass(frozen=True)
class NormalizeBiosphereNames:
    """Apply the ``biosphere-2-3-names`` migration to exchanges + datasets."""

    store: MigrationStore
    name: str = "normalize_biosphere_names"
    lcia: bool = False
    migration_name: str = "biosphere-2-3-names"

    def __call__(self, data: list[dict]) -> list[dict]:
        applier = MigrationApplier.from_store(self.store, self.migration_name)
        applier.apply_to_exchanges(data)
        if not self.lcia:
            applier.apply_to_datasets(data)
        return data


@dataclass(frozen=True)
class RemoveBiosphereLocationPrefixIfFlowInSameLocation:
    """Drop ``, AR`` from a SimaPro-regionalised flow when process is in 'AR'.

    Lifted from
    ``bw2io.strategies.remove_biosphere_location_prefix_if_flow_in_same_location``.
    """

    name: str = "remove_biosphere_location_prefix_if_flow_in_same_location"

    def __call__(self, data: list[dict]) -> list[dict]:
        for ds in data:
            loc = ds.get("location")
            if not isinstance(loc, str):
                continue
            finder = re.compile(rf"(?P<name>.+?)[\,/]* (?P<location>{re.escape(loc)})\s?$")
            for exc in ds.get("exchanges", []):
                if exc.get("type") != "biosphere":
                    continue
                exc_name = exc.get("name")
                if not exc_name:
                    continue
                m = finder.match(exc_name)
                if not m:
                    continue
                gd = m.groupdict()
                if gd["location"].strip() == loc:
                    if "simapro name" not in exc:
                        exc["simapro name"] = exc_name
                    exc["name"] = gd["name"].strip()
        return data


# ============================================================================
# Composite — the whole biosphere normalisation chain.


@dataclass
class BiosphereStrategyChain:
    """Run every biosphere normalisation strategy in canonical order.

    Replaces ``BiosphereLabelNormaliser`` (which iterated over bw2io
    callables). The chain is:

    1. ``NormalizeSimaproBiosphereCategories``
    2. ``NormalizeSimaproBiosphereNames``
    3. ``NormalizeBiosphereCategories``
    4. ``NormalizeBiosphereNames``
    5. ``StripBiosphereExchangeLocations``
    6. ``DropUnspecifiedSubcategories``

    Each step is wrapped by the runner so exceptions surface in the
    suppressed-strategy log rather than crashing the linker.
    """

    strategies: tuple[Any, ...] = field(default_factory=tuple)

    @classmethod
    def from_paths(
        cls,
        bw2io_data_dir: Path,
    ) -> BiosphereStrategyChain:
        """Build the canonical chain rooted at ``source/bw2io-data/``."""
        store = MigrationStore(directory=bw2io_data_dir)
        return cls(
            strategies=(
                NormalizeSimaproBiosphereCategories(),
                NormalizeSimaproBiosphereNames(json_path=bw2io_data_dir / "simapro-biosphere.json"),
                NormalizeBiosphereCategories(store=store),
                NormalizeBiosphereNames(store=store),
                StripBiosphereExchangeLocations(),
                DropUnspecifiedSubcategories(),
            )
        )

    def apply(self, sp: Any, runner: Any | None = None) -> None:
        """Run every strategy on ``sp.data``. ``runner`` is optional — when
        passed, suppression is recorded; otherwise exceptions propagate."""
        for s in self.strategies:
            label = getattr(s, "name", type(s).__name__)
            if runner is not None:
                runner.apply(self._apply_one, sp, s, label=label)
            else:
                self._apply_one(sp, s)

    @staticmethod
    def _apply_one(sp: Any, strategy: Any) -> None:
        sp.data = strategy(sp.data)


def _registered_strategies() -> Iterable[type]:
    """Helper used by tests / discovery — every strategy class in the module."""
    return (
        NormalizeSimaproBiosphereCategories,
        NormalizeSimaproBiosphereNames,
        NormalizeBiosphereCategories,
        NormalizeBiosphereNames,
        StripBiosphereExchangeLocations,
        DropUnspecifiedSubcategories,
        RemoveBiosphereLocationPrefixIfFlowInSameLocation,
    )
