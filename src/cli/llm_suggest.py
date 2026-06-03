"""``dds-llm-suggest-mappings`` — auto-resolve residual unlinked AGB flows.

Runs after a full ``dds-link-all`` so ``unlinked/biosphere_unlinked.xlsx``
is fresh. The CLI loads the registry to know which flows already have a
deterministic mapping, asks the LLM (CLI by default, API opt-in) to pick a
target for each truly-missing residual, and appends accepted picks straight
into ``source/agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx`` with
``decision="accept"``. No sidecar file, no human gate.

To pick up the new mappings: re-run ``dds-build-registry`` then
``dds-link-all`` — they enter at tier 10 (``LLM_OVERRIDES``).
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

try:
    from anthropic import Anthropic
except ModuleNotFoundError:
    Anthropic = None  # optional: only required when --use-api is passed

from cli._base import BaseCli
from llm import (
    AnthropicApiClient,
    CandidatePool,
    ClaudeCliClient,
    LlmMappingSuggester,
    LlmReviewedXlsxAppender,
    UnlinkedReader,
)
from matching.bio_catalog import BiosphereCatalog
from registry import MappingRegistry


class LlmSuggestCli(BaseCli):
    PROG: ClassVar[str] = "cli.llm_suggest"
    DESCRIPTION: ClassVar[str] = (
        "Ask Claude (CLI by default) to auto-resolve residual unlinked AGB "
        "biosphere flows. Picks land in source/agribalyse-3.2-biosphere-residuals-"
        "llm-reviewed.xlsx with decision=accept."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--use-api",
            action="store_true",
            help="Use the Anthropic API (Sonnet by default) instead of the local claude CLI.",
        )
        p.add_argument(
            "--model",
            default=None,
            help=(
                "Override model. With --use-api defaults to claude-sonnet-4-6; "
                "ignored for the CLI (uses whatever the local claude binary is bound to)."
            ),
        )
        p.add_argument(
            "--max-candidates",
            type=int,
            default=20,
            help="Top-N candidate biosphere flows shown to the LLM per residual.",
        )
        p.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Process at most N residuals (smoke-test the pipeline).",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        s = self.settings
        registry = MappingRegistry.load(s)
        unlinked = UnlinkedReader(settings=s).read(
            registry_unmatchable=registry.unmatchable,
            already_reviewed_xlsx=s.paths.llm_reviewed_xlsx,
        )
        if args.limit is not None:
            unlinked = unlinked[: args.limit]
        if not unlinked:
            print("No residual unlinked flows to process — exiting.")
            return

        catalog = BiosphereCatalog.load(
            s.paths.registry_biosphere_catalog,
            db_names=(s.biosphere_db_name, "biosphere3", s.ef_db_name),
        )
        pool = CandidatePool(catalog=catalog, max_candidates=args.max_candidates)

        client = self._make_client(args)
        suggester = LlmMappingSuggester(
            settings=s,
            candidate_pool=pool,
            client=client,
            max_candidates=args.max_candidates,
        )
        suggestions = suggester.suggest(unlinked)

        LlmReviewedXlsxAppender(output_path=s.paths.llm_reviewed_xlsx).append(suggestions)

    @staticmethod
    def _make_client(args: argparse.Namespace):
        if args.use_api:
            if Anthropic is None:
                raise ImportError(
                    "--use-api requires the 'anthropic' package. "
                    "Install with: pip install -e '.[llm-api]'"
                )
            return AnthropicApiClient(
                client=Anthropic(),
                model=args.model or "claude-sonnet-4-6",
            )
        return ClaudeCliClient()


def main() -> int:
    return LlmSuggestCli().run()


if __name__ == "__main__":
    sys.exit(main())
