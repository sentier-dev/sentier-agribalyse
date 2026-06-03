"""``ScoringResult`` and ``split_chunks`` — shared primitives for LCIA scoring.

The legacy ``LciaScorer``, ``WorkerPayload``, ``run_score_worker``, and
``run_isolated_score_worker`` have been removed (REFACTOR_LINKING phase L4).
All runtime scoring now goes through ``NativeLciaScorer`` in
``scoring.native_scorer``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringResult:
    process_key: tuple[str, str]
    scores: Mapping[tuple[str, ...], float] | None
    skip_reason: str | None = None
    elapsed_s: float = 0.0
