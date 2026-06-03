"""``dds-decompose-score`` — explain why a product scores what it scores.

Loads the cached :class:`~scoring.scoring_package.ScoringPackage` (the
same one ``dds-backtest`` consumes) and prints the per-biosphere-flow
contributions for one ``(product_key, method)`` pair. Activity and
edge contributions are intentionally omitted: in practice the flow
view is the one diagnosis users reach for, and the others were noise.

Usage::

    dds-decompose-score \\
        --database agribalyse-3.2 \\
        --code 88b91d4e5a9d46b697fd350423bcd087 \\
        --method climate \\
        --top-n 15 \\
        [--inventory]   # also dump uncharacterised inventory

``--method`` accepts the short name from the backtest CSV (``climate``,
``cc_bio``, ``ecotox``, ``water`` …) and resolves to the full
4-tuple via :class:`MethodAliases` so the user doesn't have to type
the verbose ``("ecoinvent-3.9.1", "EF v3.1", "climate change",
"global warming potential (GWP100)")`` form.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from cli._base import BaseCli
from scoring.decomposer import ScoreDecomposer
from scoring.product_catalog import ProductCatalog
from scoring.scoring_package import ScoringPackageStore


@dataclass(frozen=True)
class MethodAliases:
    """Map short names (used in backtest CSV headers) → full method tuples.

    Single source of truth for the mapping ``BacktestPipeline`` already
    encodes via :data:`BacktestPipeline.METHOD_TO_ADEME`. Duplicating
    here keeps the decompose CLI from depending on the backtest
    pipeline; the alias list is short and changes rarely.
    """

    SHORT_TO_FULL: ClassVar[dict[str, tuple[str, ...]]] = {
        "climate": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        ),
        "ozone": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "ozone depletion",
            "ozone depletion potential (ODP)",
        ),
        "radiation": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "ionising radiation: human health",
            "human exposure efficiency relative to u235",
        ),
        "photo_ox": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "photochemical oxidant formation: human health",
            "tropospheric ozone concentration increase",
        ),
        "pm": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "particulate matter formation",
            "impact on human health",
        ),
        "ht_nc": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "human toxicity: non-carcinogenic",
            "comparative toxic unit for human (CTUh)",
        ),
        "ht_c": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "human toxicity: carcinogenic",
            "comparative toxic unit for human (CTUh)",
        ),
        "acid": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "acidification",
            "accumulated exceedance (AE)",
        ),
        "e_fw": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "eutrophication: freshwater",
            "fraction of nutrients reaching freshwater end compartment (P)",
        ),
        "e_m": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "eutrophication: marine",
            "fraction of nutrients reaching marine end compartment (N)",
        ),
        "e_t": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "eutrophication: terrestrial",
            "accumulated exceedance (AE)",
        ),
        "ecotox": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "ecotoxicity: freshwater",
            "comparative toxic unit for ecosystems (CTUe)",
        ),
        "land": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "land use",
            "soil quality index",
        ),
        "water": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "water use",
            "user deprivation potential (deprivation-weighted water consumption)",
        ),
        "energy": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "energy resources: non-renewable",
            "abiotic depletion potential (ADP): fossil fuels",
        ),
        "mater": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "material resources: metals/minerals",
            "abiotic depletion potential (ADP): elements (ultimate reserves)",
        ),
        "cc_bio": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change: biogenic",
            "global warming potential (GWP100)",
        ),
        "cc_fos": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change: fossil",
            "global warming potential (GWP100)",
        ),
        "cc_luc": (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change: land use and land use change",
            "global warming potential (GWP100)",
        ),
    }

    @classmethod
    def resolve(cls, short_or_full: str) -> tuple[str, ...]:
        """Resolve a short alias or accept a comma-separated full tuple."""
        if short_or_full in cls.SHORT_TO_FULL:
            return cls.SHORT_TO_FULL[short_or_full]
        # Allow ad-hoc full tuple via comma separation.
        if "," in short_or_full:
            parts = tuple(p.strip() for p in short_or_full.split(","))
            return parts
        raise ValueError(
            f"unknown method alias {short_or_full!r}. Known: {sorted(cls.SHORT_TO_FULL)}"
        )


@dataclass
class DecomposeScoreCli(BaseCli):
    PROG: ClassVar[str] = "cli.decompose_score"
    DESCRIPTION: ClassVar[str] = (
        "Decompose a product's LCIA score into per-biosphere-flow contributions. "
        "Useful for diagnosing why a backtest deviation looks the way it does."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument("--database", required=True, help="Activity database (e.g. agribalyse-3.2).")
        p.add_argument("--code", required=True, help="Activity code or hex product code.")
        p.add_argument(
            "--method",
            required=True,
            help=(
                "Short alias (climate, cc_bio, ecotox, water …) or comma-separated "
                "full 4-tuple for ad-hoc methods."
            ),
        )
        p.add_argument(
            "--top-n",
            type=int,
            default=None,
            help=(
                "Optional cap on flow rows to print (default: all non-zero "
                "characterised flows, sorted by |contribution|)."
            ),
        )
        p.add_argument(
            "--inventory",
            action="store_true",
            help=(
                "Also dump the full inventory (every non-zero flow by mass, "
                "regardless of CF). Reveals 'right amount, no CF' diagnostics."
            ),
        )
        p.add_argument(
            "--out",
            type=Path,
            default=None,
            help=(
                "Optional directory; writes decomp_<code>_<method>.json with "
                "the flow contributions for downstream tooling. Always prints "
                "the human-readable summary to stdout."
            ),
        )
        p.add_argument(
            "--solver",
            choices=("scipy", "pardiso"),
            default="scipy",
            help=(
                "Linear solver for A^-1 d. 'pardiso' requires pypardiso; needed "
                "for the post-refactor matrix because scipy SuperLU rejects the "
                "37 zero-diagonal placeholder activities as 'exactly singular'."
            ),
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        package = self._load_package()
        product_catalog = ProductCatalog.load(self.settings.paths.registry_product_catalog)
        biosphere = pd.read_parquet(self.settings.paths.registry_biosphere_catalog)

        method = MethodAliases.resolve(args.method)
        decomposer = ScoreDecomposer(
            package=package,
            product_catalog=product_catalog,
            biosphere_catalog=biosphere,
            use_pardiso=(args.solver == "pardiso"),
        )
        result = decomposer.decompose((args.database, args.code), method, top_n=args.top_n)

        self._print_human_summary(result, args.method)

        if args.inventory:
            inv = decomposer.inventory((args.database, args.code), method=method, top_n=args.top_n)
            print()
            header = "All" if args.top_n is None else f"Top {args.top_n}"
            print(f"{header} INVENTORY flows (by |mass|):")
            self._print_full(inv)

        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            slug = f"{args.code}__{args.method}"
            payload = {
                "product_key": list(result.product_key),
                "method": list(result.method),
                "score": result.score,
                "global_score": result.global_score,
                "correction_score": result.correction_score,
                "flow_contributions": result.flow_contributions.to_dict(orient="records"),
            }
            (args.out / f"decomp_{slug}.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )

    # ------------------------------------------------------------------

    def _load_package(self):
        report_path = self.settings.paths.dashboard_run_report
        if not report_path.exists():
            raise FileNotFoundError(
                f"run_report.json not found at {report_path}. Run dds-link-all first."
            )
        report = json.loads(report_path.read_text())
        content_hash = report["stages"]["scoring_package"]["content_hash"]
        return ScoringPackageStore(root=self.settings.paths.scoring_packages_root).read(
            content_hash
        )

    @classmethod
    def _print_human_summary(cls, result, method_alias: str) -> None:
        print(f"== {result.product_key[1]} | {method_alias} | score = {result.score:.6g}")
        if result.correction_score != 0.0:
            print(
                f"   global term (Q @ inv) = {result.global_score:.6g}; "
                f"regional correction (Δ @ supply) = {result.correction_score:+.6g}"
            )
        print()
        print(f"FLOW contributions ({len(result.flow_contributions)} rows):")
        cls._print_full(result.flow_contributions)

    @staticmethod
    def _print_full(df: pd.DataFrame) -> None:
        """Render a DataFrame with every column and every row visible."""
        with pd.option_context(
            "display.max_columns",
            None,
            "display.max_rows",
            None,
            "display.width",
            None,
            "display.max_colwidth",
            80,
        ):
            print(df.to_string(index=False))


def main() -> int:
    return DecomposeScoreCli().run()


if __name__ == "__main__":
    sys.exit(main())
