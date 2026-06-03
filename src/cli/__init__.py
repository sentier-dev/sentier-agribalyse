"""Command-line entrypoints. One class per CLI."""

from cli._base import BaseCli
from cli.backtest import BacktestCli
from cli.build_packages import BuildPackagesCli
from cli.build_registry import BuildRegistryCli
from cli.link_all import LinkAllCli
from cli.llm_suggest import LlmSuggestCli
from cli.mappings_comparison import MappingsComparisonCli
from cli.run_end_to_end import RunEndToEndCli

__all__ = [
    "BacktestCli",
    "BaseCli",
    "BuildPackagesCli",
    "BuildRegistryCli",
    "LinkAllCli",
    "LlmSuggestCli",
    "MappingsComparisonCli",
    "RunEndToEndCli",
]
