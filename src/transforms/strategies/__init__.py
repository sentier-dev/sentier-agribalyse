"""In-house strategy classes that replace ``bw2io.strategies`` callables.

Each strategy is a small class with an ``apply(data) -> data`` (or
``__call__``) surface so the existing ``StrategyRunner.apply(...)`` pipeline
can drive them. They are pure dict transforms — no bw2data, no SQLite —
which is what makes the whole linker SQLite-free per REFACTOR_FINAL.
"""

from transforms.strategies.biosphere import (
    BiosphereStrategyChain,
    DropUnspecifiedSubcategories,
    NormalizeBiosphereCategories,
    NormalizeBiosphereNames,
    NormalizeSimaproBiosphereCategories,
    NormalizeSimaproBiosphereNames,
    RemoveBiosphereLocationPrefixIfFlowInSameLocation,
    StripBiosphereExchangeLocations,
)
from transforms.strategies.internal import (
    DropUnlinkedExchanges,
    LinkIterableByFields,
    SetMetadataUsingSingleFunctionalExchange,
    SplitSimaproNameGeo,
)
from transforms.strategies.labels import (
    NormalizeSimaproLabelsToBrightwayStandard,
)
from transforms.strategies.migrations import MigrationApplier, MigrationStore
from transforms.strategies.units import (
    ChangeElectricityUnitMjToKwh,
    RescaleExchange,
)

__all__ = [
    "BiosphereStrategyChain",
    "ChangeElectricityUnitMjToKwh",
    "DropUnlinkedExchanges",
    "DropUnspecifiedSubcategories",
    "LinkIterableByFields",
    "MigrationApplier",
    "MigrationStore",
    "NormalizeBiosphereCategories",
    "NormalizeBiosphereNames",
    "NormalizeSimaproBiosphereCategories",
    "NormalizeSimaproBiosphereNames",
    "NormalizeSimaproLabelsToBrightwayStandard",
    "RemoveBiosphereLocationPrefixIfFlowInSameLocation",
    "RescaleExchange",
    "SetMetadataUsingSingleFunctionalExchange",
    "SplitSimaproNameGeo",
    "StripBiosphereExchangeLocations",
]
