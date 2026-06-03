"""Coarse normalisation stages — now driven by in-house strategy classes.

Each class wraps a small group of dict transforms (lifted from bw2io
into ``transforms.strategies`` per REFACTOR_FINAL phase F3). The legacy
``BioStrategyChain``, which used to call ``sp.match_database(bio_db, ...)``
against ``bw2data``, is replaced by ``BiosphereCatalogPrelinker`` —
which reads the same biosphere flow universe from the parquet catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import Settings
from matching.bio_catalog import BiosphereCatalog
from matching.strategy_runner import StrategyRunner
from transforms.biosphere_prelinker import BiosphereCatalogPrelinker
from transforms.strategies.biosphere import BiosphereStrategyChain
from transforms.strategies.labels import NormalizeSimaproLabelsToBrightwayStandard


@dataclass(frozen=True)
class InternalAgbLinker:
    """Default-strategy chain + initial intra-AGB exchange linking.

    The default chain (``set_metadata_using_single_functional_exchange``,
    ``drop_unspecified_subcategories``, ``split_simapro_name_geo``) lives
    on ``ParsedSimaProCsv.strategies``. The internal match links AGB
    process exchanges to AGB product nodes via name+unit.
    """

    runner: StrategyRunner

    def apply(self, sp: Any) -> None:
        sp.apply_strategies()
        sp.match_database(fields=["name", "unit"], processes_to_products=True)


@dataclass
class BiosphereLabelNormaliser:
    """Run the lifted biosphere normalisation chain via ``StrategyRunner``."""

    runner: StrategyRunner
    settings: Settings
    chain: BiosphereStrategyChain | None = field(default=None)

    def apply(self, sp: Any) -> None:
        chain = self.chain or BiosphereStrategyChain.from_paths(
            bw2io_data_dir=self.settings.paths.bw2io_data
        )
        chain.apply(sp, runner=self.runner)


@dataclass(frozen=True)
class StandardLabelNormaliser:
    """Brightway-standard labels + units; the post-flowmap finishing pass.

    ``normalize_labels_to_brightway_standard`` is now driven by the
    lifted ``NormalizeSimaproLabelsToBrightwayStandard`` strategy. The
    randonneur datapackage is still applied via the wrapper (no bw2data
    needed — randonneur reads stored datapackages directly).
    """

    runner: StrategyRunner

    def apply(self, sp: Any) -> None:
        sp.apply_strategy(NormalizeSimaproLabelsToBrightwayStandard())
        sp.randonneur("generic-brightway-units-normalization")


@dataclass
class BioStrategyChain:
    """Pre-pass biosphere linking via the parquet catalog (no bw2data).

    Drives ``BiosphereCatalogPrelinker`` followed by the
    ``RemoveBiosphereLocationPrefixIfFlowInSameLocation`` strategy.
    Replaces the legacy class of the same name that called
    ``sp.match_database(bio_db, ...)`` against the SQLite biosphere DB.
    """

    runner: StrategyRunner
    bio_db_name: str
    catalog: BiosphereCatalog

    def apply(self, sp: Any) -> dict[str, int]:
        prelinker = BiosphereCatalogPrelinker(
            bio_db_name=self.bio_db_name,
            catalog=self.catalog,
        )
        return prelinker.apply(sp.data)


@dataclass(frozen=True)
class RestoreSimaproNamesTransform:
    """Apply ``simapro-ecoinvent-3.9.1-cutoff`` + restore-names datapackages.

    The restore-names datapackage may not be registered in older
    ``randonneur_data`` versions; record the suppression rather than
    silently swallow the failure.
    """

    runner: StrategyRunner

    def apply(self, sp: Any) -> None:
        self.runner.apply(
            sp.randonneur,
            "agribalyse-3.1.1-restore-simapro-ecoinvent-names",
            fields=["name"],
            label="randonneur.restore-simapro-ecoinvent-names",
        )
        sp.randonneur("simapro-ecoinvent-3.9.1-cutoff", fields=["name"])
