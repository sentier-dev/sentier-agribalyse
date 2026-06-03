"""LCIA scoring layer."""

from scoring.activity_location_overrides import ActivityLocationOverrides
from scoring.activity_location_parser import ActivityLocationParser
from scoring.allocator import Allocator
from scoring.aware_consumption_correction import AwareConsumptionCorrectionBuilder
from scoring.dangling_edge_auditor import (
    DanglingEdgeAuditor,
    FrameSnapshot,
    IdNameResolver,
)
from scoring.dangling_edge_pruner import DanglingEdgePruner
from scoring.ecoinvent_exchanges import EcoinventExchangesIngester
from scoring.exchange_frame import ExchangeFrame
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.matrix_builder import (
    BiosphereBuilder,
    BuiltMatrix,
    CharacterizationBuilder,
    TechnosphereBuilder,
)
from scoring.method_slug import MethodSlug
from scoring.native_scorer import NativeLciaScorer, NativeScoreWorker, NativeWorkerPayload
from scoring.perennial_lifecycle_filter import PerennialLifecycleFilter
from scoring.product_catalog import ProductCatalog, ProductCatalogBuilder
from scoring.product_deduplicator import ProductDeduplicator
from scoring.scorer import ScoringResult
from scoring.scoring_package import (
    ScoringPackage,
    ScoringPackageBuilder,
    ScoringPackageStore,
)

__all__ = [
    "ActivityLocationOverrides",
    "ActivityLocationParser",
    "Allocator",
    "AwareConsumptionCorrectionBuilder",
    "BiosphereBuilder",
    "BuiltMatrix",
    "CharacterizationBuilder",
    "DanglingEdgeAuditor",
    "DanglingEdgePruner",
    "EcoinventExchangesIngester",
    "ExchangeFrame",
    "ExchangeFrameBuilder",
    "FrameSnapshot",
    "IdNameResolver",
    "MethodSlug",
    "NativeLciaScorer",
    "NativeScoreWorker",
    "NativeWorkerPayload",
    "PerennialLifecycleFilter",
    "ProductCatalog",
    "ProductCatalogBuilder",
    "ProductDeduplicator",
    "ScoringPackage",
    "ScoringPackageBuilder",
    "ScoringPackageStore",
    "ScoringResult",
    "TechnosphereBuilder",
]
