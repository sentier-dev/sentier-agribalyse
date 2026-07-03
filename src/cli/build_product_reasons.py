"""``dds-build-product-reasons`` — author per-product outlier explanations.

Scans ``dashboard/backtest_pass1.csv`` for product × impact cells whose
|%diff| versus the ADEME reference clears ``--threshold``, then has an LLM
(local ``claude`` CLI by default) author a one- to two-sentence, product-
specific explanation for each, grounded in that product's flow decomposition
(``dashboard/decomp/<code>.json``) and the impact-level notes
(``dashboard/outlier_reasons.json``). Results are written to
``dashboard/product_reasons.json``.

Standalone and resumable — re-running skips products already covered (unless
``--force``); the output is checkpointed after every product so a long run
survives interruption. Re-run after a fresh backtest or decomposition.

Examples::

    dds-build-product-reasons --limit 5          # sample run, validate quality
    dds-build-product-reasons                    # full run (resumes)
    dds-build-product-reasons --use-api --force  # rebuild from scratch via API
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import ClassVar

from cli._base import BaseCli
from llm.client import AnthropicApiClient, ClaudeCliClient, LlmClient
from reporting.product_reasons import (
    DecompEvidence,
    ProductOutlierScanner,
    ProductReasonGenerator,
    ProductReasonPrompt,
)


@dataclass
class BuildProductReasonsCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_product_reasons"
    DESCRIPTION: ClassVar[str] = (
        "Author per-product, per-outlier-impact explanations into "
        "dashboard/product_reasons.json via an LLM pass over the flow "
        "decomposition. Resumable; rerun after a fresh backtest."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--threshold",
            type=float,
            default=30.0,
            help="Outlier threshold in percent |%%diff| (default: 30, matches the dashboard).",
        )
        p.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Only process the first N outlier products (for sample runs).",
        )
        p.add_argument(
            "--codes",
            nargs="*",
            default=None,
            help="Only process these product codes (space-separated).",
        )
        p.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Concurrent LLM calls (default: 4).",
        )
        p.add_argument(
            "--top-flows",
            type=int,
            default=6,
            help="Flows per impact handed to the model as evidence (default: 6).",
        )
        p.add_argument(
            "--force",
            action="store_true",
            help="Re-generate every product, ignoring existing product_reasons.json.",
        )
        p.add_argument(
            "--use-api",
            action="store_true",
            help="Use the Anthropic API (needs ANTHROPIC_API_KEY) instead of the claude CLI.",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        csv_path = self.settings.paths.dashboard_backtest_pass1_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run dds-backtest then the dashboard CSV emitter first."
            )
        rows = ProductOutlierScanner.parse_csv(csv_path.read_text())
        products = ProductOutlierScanner(rows=rows, threshold_pct=args.threshold).scan()

        if args.codes:
            wanted = set(args.codes)
            products = [p for p in products if p.code in wanted]
        if args.limit is not None:
            products = products[: args.limit]

        total_cells = sum(len(p.impacts) for p in products)
        log = self._log()
        log.info(
            "product_reasons.scanned",
            products=len(products),
            cells=total_cells,
            threshold=args.threshold,
        )
        if not products:
            log.warning("product_reasons.nothing_to_do")
            return

        impact_notes = self._load_impact_notes()
        prompt = ProductReasonPrompt(
            impact_notes=impact_notes,
            evidence=DecompEvidence(
                decomp_dir=self.settings.paths.dashboard_decomp_dir,
                top_n=args.top_flows,
            ),
        )
        generator = ProductReasonGenerator(
            client=self._client(args),
            prompt=prompt,
            out_path=self.settings.paths.dashboard_product_reasons,
            max_workers=args.workers,
            force=args.force,
        )
        result = generator.run(products)
        log.info(
            "product_reasons.done",
            products_with_notes=len(result),
            out=str(self.settings.paths.dashboard_product_reasons),
        )

    # ------------------------------------------------------------------

    def _log(self):
        from core.logging import Logging

        return Logging.get(self.PROG)

    def _load_impact_notes(self) -> dict[str, dict]:
        path = self.settings.paths.dashboard_outlier_reasons
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    def _client(self, args: argparse.Namespace) -> LlmClient:
        if not args.use_api:
            return ClaudeCliClient()
        import anthropic  # imported lazily; only needed with --use-api

        return AnthropicApiClient(client=anthropic.Anthropic())


def main() -> int:
    return BuildProductReasonsCli().run()


if __name__ == "__main__":
    sys.exit(main())
