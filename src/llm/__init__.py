"""LLM-assisted mapping for residual unlinked AGB biosphere flows.

Runs LAST in the linking workflow — after every deterministic tier has had
its chance. The LLM's job is constrained: pick the best-matching existing
biosphere flow from a pre-filtered candidate pool, OR mark the flow as
unmappable. It never invents flow IDs.

Decisions are auto-accepted; new rows are appended to
``source/agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx`` with
``decision="accept"`` so the next ``dds-build-registry`` picks them up at
tier 10. No sidecar file, no human gate.
"""

from llm.candidates import CandidatePool
from llm.client import AnthropicApiClient, ClaudeCliClient, LlmClient
from llm.exporter import LlmReviewedXlsxAppender
from llm.suggester import (
    AgbUnlinkedFlow,
    LlmMappingSuggester,
    Suggestion,
    UnlinkedReader,
)

__all__ = [
    "AgbUnlinkedFlow",
    "AnthropicApiClient",
    "CandidatePool",
    "ClaudeCliClient",
    "LlmClient",
    "LlmMappingSuggester",
    "LlmReviewedXlsxAppender",
    "Suggestion",
    "UnlinkedReader",
]
