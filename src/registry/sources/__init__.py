"""One class per registry source. ``RegistryBuilder`` orchestrates them."""

from registry.sources.agribalyse_assets import (
    AGB32FlowmapperBiosphereSource,
    AgbDeleteAggregatedSource,
    AgbEdgeLabelsSource,
)
from registry.sources.curated import (
    CuratedOverridesSource,
    ExtraUnitConversionSource,
    LlmReviewedSource,
)
from registry.sources.ef_target import EfCfTargetIndexSource
from registry.sources.harmonised import HarmonisedFlowsSource
from registry.sources.placeholder import (
    PlaceholderEcoinventSource,
    PlaceholderEfNativeSource,
    PlaceholderTransitiveSource,
    PlaceholderUnmatchableSource,
)
from registry.sources.randonneur_packages import (
    RandonneurAgribalyseBiosphereSource,
    RandonneurSimaproBiosphereSource,
    RandonneurSimaproContextSource,
    RandonneurUnitAliasSource,
    RandonneurUnitConversionSource,
    RandonneurUnitNormalisationSource,
    RandonneurUnlinkedListSource,
    RandonneurWaterSlashM3Source,
)

__all__ = [
    "AGB32FlowmapperBiosphereSource",
    "AgbDeleteAggregatedSource",
    "AgbEdgeLabelsSource",
    "CuratedOverridesSource",
    "EfCfTargetIndexSource",
    "ExtraUnitConversionSource",
    "HarmonisedFlowsSource",
    "LlmReviewedSource",
    "PlaceholderEcoinventSource",
    "PlaceholderEfNativeSource",
    "PlaceholderTransitiveSource",
    "PlaceholderUnmatchableSource",
    "RandonneurAgribalyseBiosphereSource",
    "RandonneurSimaproBiosphereSource",
    "RandonneurSimaproContextSource",
    "RandonneurUnitAliasSource",
    "RandonneurUnitConversionSource",
    "RandonneurUnitNormalisationSource",
    "RandonneurUnlinkedListSource",
    "RandonneurWaterSlashM3Source",
]
