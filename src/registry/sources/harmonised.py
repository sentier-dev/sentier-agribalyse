"""``HarmonisedFlowsSource`` — Sentier/Brightway harmonised flow registry → tier 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from domain import Bucket, Mapping, SourceKind, Tier
from readers import GzJsonReader


@dataclass(frozen=True)
class HarmonisedFlowsSource:
    """Read ``harmonised-flows-simple.json.gz`` and emit one mapping per altLabel.

    When ``valid_uuids`` is provided, flows whose ``identifier`` is not in the
    set are skipped — these UUIDs have no CF data and would silently zero out
    at LCIA time, matching pre-refactor behavior (≈4k of 79k harmonised entries).
    """

    path: Path
    target_db: str = "ef"
    tier: Tier = Tier.HARMONISED_FLOWS
    provenance: str = "harmonised-flows-simple"
    valid_uuids: frozenset[str] | None = None
    gz: GzJsonReader = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gz is None:
            object.__setattr__(self, "gz", GzJsonReader())

    def read(self) -> list[Mapping]:
        if not Path(self.path).exists():
            return []
        data = self.gz.read(self.path)
        flows = [f for f in data.get("flows", []) if f.get("source") == "EF 3.1"]

        # Dedupe by (name_lower, bucket, uuid): the matcher only ever takes
        # the first candidate per key, so storing all altLabels separately
        # explodes the registry to 1.6M rows for no behavioural gain.
        seen: set[tuple[str, str, str]] = set()
        out: list[Mapping] = []
        for flow_idx, flow in enumerate(flows):
            uuid = str(flow.get("identifier") or "").strip()
            if not uuid:
                continue
            if self.valid_uuids is not None and uuid not in self.valid_uuids:
                continue
            bucket = Bucket.from_harmonised_iri(flow.get("context_iri", ""))
            pref = (flow.get("prefLabel") or "").strip()
            names: list[str] = [pref] if pref else []
            for alt in flow.get("altLabel") or []:
                if isinstance(alt, str) and "<" not in alt:
                    s = alt.strip()
                    if s:
                        names.append(s)
            for j, name in enumerate(names):
                key = (name.lower(), bucket.value, uuid)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Mapping(
                        source_kind=SourceKind.AGB_FLOW,
                        source_name=name,
                        source_unit="",
                        source_context=(),
                        source_top_bucket=bucket,
                        target_db=self.target_db,
                        target_code=uuid,
                        target_name=pref or name,
                        target_unit="",
                        priority_tier=self.tier,
                        provenance=self.provenance,
                        provenance_row=f"{flow_idx}.{j}",
                    )
                )
        return out
