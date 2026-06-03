"""End-to-end pipelines. Each is a class with a single ``.run()`` method."""

from pipelines.backtest import BacktestOptions, BacktestPipeline
from pipelines.end_to_end import EndToEndOptions, EndToEndPipeline
from pipelines.link_all import LinkAllOptions, LinkAllPipeline
from pipelines.registry_build import RegistryBuildPipeline

__all__ = [
    "BacktestOptions",
    "BacktestPipeline",
    "EndToEndOptions",
    "EndToEndPipeline",
    "LinkAllOptions",
    "LinkAllPipeline",
    "RegistryBuildPipeline",
]
