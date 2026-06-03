"""``CandidatePool`` — pre-filter biosphere flows down to a short list per query.

The catalog holds ~80k flows across biosphere3 / ecoinvent-bio / ef. Sending
that many to an LLM per residual is wasteful and exceeds context. We pre-rank
by tokenized name overlap within the exchange's compartment bucket (the LLM
must respect compartment, so off-bucket hits are dropped up front).

Ranking is deterministic: token-overlap score descending, name length ascending
(prefer shorter, more generic names on ties), then lex by (db, code).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

from domain import Bucket
from matching.bio_catalog import BioFlowRef, BiosphereCatalog

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CandidatePool:
    """Score-and-rank biosphere flows from a catalog by lexical similarity."""

    catalog: BiosphereCatalog
    max_candidates: int = 20

    @cached_property
    def _by_bucket(self) -> dict[str, list[tuple[BioFlowRef, frozenset[str]]]]:
        """``bucket → [(flow, token_set), …]`` so we can rank within bucket fast."""
        out: dict[str, list[tuple[BioFlowRef, frozenset[str]]]] = {}
        for f in self.catalog.flows:
            tokens = self._tokenize(f.name)
            out.setdefault(f.bucket.value, []).append((f, tokens))
        return out

    def candidates_for(
        self,
        source_name: str,
        bucket: Bucket | str,
        max_n: int | None = None,
    ) -> list[BioFlowRef]:
        """Return up to ``max_n`` candidates ordered by similarity to ``source_name``."""
        b = bucket.value if isinstance(bucket, Bucket) else str(bucket)
        n = max_n if max_n is not None else self.max_candidates
        src_tokens = self._tokenize(source_name)
        if not src_tokens:
            return []
        pool = self._by_bucket.get(b, [])
        scored: list[tuple[int, int, str, str, BioFlowRef]] = []
        for flow, flow_tokens in pool:
            overlap = len(src_tokens & flow_tokens)
            if overlap == 0:
                continue
            # Sort key: -overlap (more is better), len(name) (shorter is better),
            # db, code — all deterministic.
            scored.append((-overlap, len(flow.name), flow.db, flow.code, flow))
        scored.sort()
        return [s[4] for s in scored[:n]]

    @staticmethod
    def _tokenize(text: str) -> frozenset[str]:
        """Lowercase alphanum tokens. Strips punctuation and chemistry junk."""
        if not text:
            return frozenset()
        return frozenset(_TOKEN_RE.findall(text.lower()))
