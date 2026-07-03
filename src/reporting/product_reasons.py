"""``product_reasons.py`` — author per-product × outlier-impact explanations.

The backtest dashboard already carries *per-impact* outlier notes
(``outlier_reasons.json``, one paragraph per impact category). This module
adds the *per-product* layer: for every product × impact cell whose |%diff|
versus the ADEME reference clears a threshold, an LLM authors a one- to
two-sentence, product-specific explanation grounded in that product's flow
decomposition. The dashboard renders these above the impact-level note in the
cell tooltip.

The pipeline is three deterministic classes plus one LLM-backed orchestrator:

* :class:`ProductOutlierScanner` — reads ``backtest_pass1.csv`` and groups the
  outlier cells by product.
* :class:`DecompEvidence` — pulls the top contributing flows for one
  product × method out of ``dashboard/decomp/<code>.json``.
* :class:`ProductReasonPrompt` — turns a product's outliers + decomp evidence
  + the impact-level notes into a ``(system, user)`` prompt pair and parses the
  model's JSON reply.
* :class:`ProductReasonGenerator` — walks the products, calls an
  :class:`~llm.client.LlmClient`, and writes ``product_reasons.json``.

The deterministic classes are unit-tested with a stub client; no network or
CLI calls happen in CI.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, NamedTuple

from core.logging import Logging
from llm.client import LlmClient
from reporting.backtest_dashboard_csv import BacktestPass1Emitter

_LOG = Logging.get("reporting.product_reasons")


class OutlierImpact(NamedTuple):
    """One product × impact cell that cleared the outlier threshold."""

    short: str  # method short id, e.g. "ht_c"
    label: str  # human method name, e.g. "human toxicity: carcinogenic"
    pct: float  # signed %diff (model vs ADEME reference)

    @property
    def direction(self) -> str:
        return "over" if self.pct > 0 else "under"


@dataclass(frozen=True)
class ProductOutliers:
    """A product and the outlier impacts found for it."""

    code: str
    name: str
    impacts: tuple[OutlierImpact, ...]


@dataclass(frozen=True)
class FlowLine:
    """One contributing flow, distilled for the prompt."""

    flow_name: str
    compartment: str
    sub_compartment: str
    cf: float | None
    sp_cf: float | None
    share_pct: float
    provenance: str
    contribution_sign: str

    @property
    def cf_delta_pct(self) -> float | None:
        """``(EF CF − SimaPro CF) / |SimaPro CF|`` as a percent, or ``None``."""
        if self.cf is None or self.sp_cf is None or self.sp_cf == 0:
            return None
        if not math.isfinite(self.cf) or not math.isfinite(self.sp_cf):
            return None
        return ((self.cf - self.sp_cf) / abs(self.sp_cf)) * 100.0

    def render(self) -> str:
        comp = self.compartment + (f" / {self.sub_compartment}" if self.sub_compartment else "")
        delta = self.cf_delta_pct
        delta_str = "no SimaPro CF" if delta is None else f"EF CF {delta:+.0f}% vs SimaPro"
        return (
            f"- {self.flow_name} [{comp}] · {self.share_pct:.0f}% of |Σcontrib| · "
            f"{self.contribution_sign} contribution · {delta_str} · match: {self.provenance}"
        )


@dataclass(frozen=True)
class ProductOutlierScanner:
    """Group the backtest CSV's outlier cells by product.

    ``rows`` is the parsed ``backtest_pass1.csv`` (list of dict rows). Only
    ``resolution == 'mapped'`` rows are considered; cells whose |%diff| is at
    or above ``threshold_pct`` become outliers.
    """

    rows: Sequence[Mapping[str, str]]
    threshold_pct: float = 30.0

    _LABELS: ClassVar[Mapping[str, str]] = {
        short: long for long, short in BacktestPass1Emitter.LONG_TO_SHORT.items()
    }

    def scan(self) -> list[ProductOutliers]:
        out: list[ProductOutliers] = []
        for row in self.rows:
            if (row.get("resolution") or "mapped") != "mapped":
                continue
            impacts: list[OutlierImpact] = []
            for short in BacktestPass1Emitter.SHORT_ORDER:
                raw = row.get(short, "")
                try:
                    pct = float(raw)
                except (TypeError, ValueError):
                    continue
                if abs(pct) >= self.threshold_pct:
                    impacts.append(
                        OutlierImpact(short=short, label=self._LABELS.get(short, short), pct=pct)
                    )
            if impacts:
                out.append(
                    ProductOutliers(
                        code=row.get("code", ""),
                        name=row.get("name", ""),
                        impacts=tuple(impacts),
                    )
                )
        return out

    @staticmethod
    def parse_csv(text: str) -> list[dict[str, str]]:
        """Parse ``backtest_pass1.csv`` into row dicts using the stdlib reader."""
        import csv
        import io

        return list(csv.DictReader(io.StringIO(text)))


@dataclass(frozen=True)
class DecompEvidence:
    """Read top contributing flows for one product × method from its decomp JSON."""

    decomp_dir: Path
    top_n: int = 6
    min_share_pct: float = 1.0

    def for_product(self, code: str) -> dict | None:
        path = self.decomp_dir / f"{code}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("product_reasons.decomp_unreadable", code=code, error=str(exc))
            return None

    def flows(self, decomp: dict, short: str) -> list[FlowLine]:
        method = (decomp.get("methods") or {}).get(short)
        if not method:
            return []
        lines: list[FlowLine] = []
        for f in method.get("flows", []):
            share_pct = float(f.get("share", 0.0)) * 100.0
            if share_pct < self.min_share_pct:
                continue
            contribution = f.get("contribution", 0.0)
            lines.append(
                FlowLine(
                    flow_name=str(f.get("flow_name", "?")),
                    compartment=str(f.get("compartment", "")),
                    sub_compartment=str(f.get("sub_compartment", "") or ""),
                    cf=_finite_or_none(f.get("cf")),
                    sp_cf=_finite_or_none(f.get("sp_cf")),
                    share_pct=share_pct,
                    provenance=str(f.get("sp_match_provenance") or "—"),
                    contribution_sign=("negative" if contribution < 0 else "positive"),
                )
            )
            if len(lines) >= self.top_n:
                break
        return lines


def _finite_or_none(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclass(frozen=True)
class ProductReasonPrompt:
    """Build the ``(system, user)`` prompt for one product and parse the reply."""

    impact_notes: Mapping[str, dict]
    evidence: DecompEvidence

    SYSTEM: ClassVar[str] = (
        "You explain, in plain engineer-primitive terms, why a single food "
        "product's life-cycle impact score deviates from ADEME's published "
        "Agribalyse reference. Voice: calm, factual, no hype, no marketing "
        "words, no exclamation marks. Each explanation is ONE or TWO short "
        "sentences. Ground every claim strictly in the numbers you are given. "
        "name the dominant flow(s) and the mechanism. Never invent flows, CFs, "
        "or percentages. The deviation is not a calculation bug; it traces to "
        "inventory data or a documented CF/methodology difference with ADEME. "
        "Return ONLY a JSON object mapping each impact short-id to its "
        "explanation string. No prose outside the JSON."
    )

    def user(self, product: ProductOutliers, decomp: dict | None) -> str:
        blocks: list[str] = [
            f"Product: {product.name} (code {product.code})",
            "",
            "For each impact below, write a product-specific explanation.",
            "",
        ]
        for imp in product.impacts:
            note = self.impact_notes.get(imp.short) or {}
            blocks.append(f"## impact short-id: {imp.short}  ({imp.label})")
            blocks.append(
                f"Model is {imp.pct:+.0f}% vs ADEME ({imp.direction}-estimates)."
            )
            if note.get("short"):
                blocks.append(f"Impact-level mechanism (context, do not repeat verbatim): {note['short']}")
            flows = self.evidence.flows(decomp, imp.short) if decomp else []
            if flows:
                blocks.append("Top contributing flows (this product, this impact):")
                blocks.extend(line.render() for line in flows)
            else:
                blocks.append("No flow decomposition available for this product/impact.")
            blocks.append("")
        example_key = product.impacts[0].short
        blocks.append(f'Reply with JSON only, e.g. {{"{example_key}": "..."}}.')
        return "\n".join(blocks)

    @staticmethod
    def parse(reply: str, wanted: Sequence[str]) -> dict[str, str]:
        """Pull the JSON object out of the model reply, keeping only wanted keys."""
        obj = _extract_json_object(reply)
        if obj is None:
            return {}
        out: dict[str, str] = {}
        for key in wanted:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                out[key] = val.strip()
        return out


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class ProductReasonGenerator:
    """Author per-product notes for every outlier product and write the JSON.

    Resumable: existing entries in ``out_path`` are kept and their products
    skipped unless ``force`` is set.
    """

    client: LlmClient
    prompt: ProductReasonPrompt
    out_path: Path
    max_workers: int = 4
    force: bool = False
    _existing: dict[str, dict[str, str]] = field(default_factory=dict)

    def run(self, products: Sequence[ProductOutliers]) -> dict[str, dict[str, str]]:
        result = dict(self._load_existing())
        todo = [
            p
            for p in products
            if self.force or not result.get(p.code) or self._missing_impacts(p, result)
        ]
        _LOG.info(
            "product_reasons.start",
            products=len(products),
            to_generate=len(todo),
            already_done=len(products) - len(todo),
            workers=self.max_workers,
        )
        done = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._one, p): p for p in todo}
            for fut in as_completed(futures):
                product = futures[fut]
                notes = fut.result()
                done += 1
                if notes:
                    result.setdefault(product.code, {}).update(notes)
                    self._checkpoint(result)
                _LOG.info(
                    "product_reasons.progress",
                    done=done,
                    total=len(todo),
                    code=product.code,
                    got=len(notes),
                )
        self._write(result)
        return result

    def _one(self, product: ProductOutliers) -> dict[str, str]:
        decomp = self.prompt.evidence.for_product(product.code)
        wanted = [i.short for i in product.impacts]
        try:
            reply = self.client.ask(self.prompt.SYSTEM, self.prompt.user(product, decomp))
        except Exception as exc:  # one bad product must not kill the batch
            _LOG.warning("product_reasons.llm_failed", code=product.code, error=str(exc))
            return {}
        notes = self.prompt.parse(reply, wanted)
        if not notes:
            _LOG.warning("product_reasons.empty_parse", code=product.code)
        return notes

    @staticmethod
    def _missing_impacts(p: ProductOutliers, result: Mapping[str, dict]) -> bool:
        have = result.get(p.code, {})
        return any(i.short not in have for i in p.impacts)

    def _load_existing(self) -> dict[str, dict[str, str]]:
        if self.force or not self.out_path.exists():
            return {}
        try:
            data = json.loads(self.out_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}

    def _checkpoint(self, result: Mapping[str, dict]) -> None:
        # Write-through so a long run survives interruption.
        self._write(result)

    def _write(self, result: Mapping[str, dict]) -> None:
        payload = {
            "_about": (
                "Per-product explanations for impact categories where the model "
                "deviates from ADEME's Agribalyse reference beyond the dashboard "
                "outlier threshold. Authored by an LLM pass over each product's "
                "flow decomposition (dds-build-product-reasons). Keyed by product "
                "code, then impact short-id. Shown above the impact-level note in "
                "the dashboard cell tooltip."
            ),
            **{k: result[k] for k in sorted(result) if result[k]},
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
