"""``dds-set-parameter`` — override a SimaPro input parameter and rescore.

Writes the override to ``source/parameter_overrides.csv`` (the single
source of override truth; gitignored, cleared by ``dds-reset``), prints
the directly changed exchange amounts, then reruns link + backtest so the
dashboard reflects the what-if. ``--no-rescore`` writes the file only.

Names are the SimaPro-facing ones (e.g. ``Packaging_Weight``), matched
case-insensitively. ``--product <code>`` targets one process;
``--all-products`` every process defining the parameter.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import ClassVar

from cli._base import BaseCli
from pipelines import BacktestOptions, BacktestPipeline, FastRescorePipeline, LinkAllPipeline
from transforms import (
    GLOBAL_SCOPE,
    ParameterOverride,
    ParameterOverridesStore,
    ParameterReevaluator,
    SimaProImporter,
)
from transforms.linked_cache import LinkedSpCache


class SetParameterCli(BaseCli):
    PROG: ClassVar[str] = "cli.set_parameter"
    DESCRIPTION: ClassVar[str] = (
        "Override a SimaPro input parameter (persisted to "
        "source/parameter_overrides.csv) and rescore."
    )

    MAX_PREVIEW_ROWS: ClassVar[int] = 20

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument("name", help="Parameter name as known in SimaPro, e.g. Packaging_Weight.")
        p.add_argument("value", type=float, help="New numeric value.")
        scope = p.add_mutually_exclusive_group(required=True)
        scope.add_argument("--product", metavar="CODE", help="Apply to this process code only.")
        scope.add_argument(
            "--all-products",
            action="store_true",
            help="Apply to every process that defines the parameter.",
        )
        p.add_argument("--comment", default="", help="Free-text note stored with the override.")
        p.add_argument(
            "--no-rescore",
            action="store_true",
            help="Write the override file only; skip the link + backtest rerun.",
        )
        p.add_argument(
            "--fast",
            action="store_true",
            help=(
                "Rescore via the linked-cache fast path (skips parse/transforms/"
                "matching; needs one prior full dds-link-all to seed the cache)."
            ),
        )
        p.add_argument("--solver", choices=("scipy", "pardiso"), default="pardiso")
        return p

    def execute(self, args: argparse.Namespace) -> None:
        if not math.isfinite(args.value):
            raise ValueError(
                f"Value must be a finite number, got {args.value!r} — "
                f"NaN/inf would corrupt the scoring matrices."
            )
        # Fail BEFORE persisting the override: a missing linked cache
        # would otherwise leave the store and the dashboard out of sync.
        if (
            args.fast
            and not args.no_rescore
            and not LinkedSpCache(settings=self.settings).path.exists()
        ):
            raise FileNotFoundError(
                "cache/linked_cache.pkl not found — run dds-link-all once "
                "to seed the fast path, or drop --fast. No override was saved."
            )
        scope = args.product if args.product else GLOBAL_SCOPE
        override = ParameterOverride(
            parameter_name=args.name,
            scope=scope,
            value=args.value,
            comment=args.comment,
        )

        sp = SimaProImporter(self.settings).load()
        reevaluator = ParameterReevaluator(parameters=sp.parameters)
        # Validates name/kind/scope (raises with suggestions) and previews the
        # direct effect of THIS override before anything is persisted.
        # Ratio mode matches what the pipelines apply (see
        # ``ParameterOverridesApplier``).
        result = reevaluator.ratio_patch(sp.data, [override])

        store = ParameterOverridesStore(settings=self.settings)
        stamped = store.upsert(override)

        print(
            f"Override saved: {stamped.parameter_name} = {stamped.value} "
            f"(scope: {stamped.scope}) -> {store.path}"
        )
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        print(f"Directly changed exchange amounts: {len(result.changes)}")
        for change in result.changes[: self.MAX_PREVIEW_ROWS]:
            print(
                f"  {change.process_code}  {change.exchange_name[:60]:60s} "
                f"{change.old_amount:.6g} -> {change.new_amount:.6g}"
            )
        if len(result.changes) > self.MAX_PREVIEW_ROWS:
            print(f"  ... and {len(result.changes) - self.MAX_PREVIEW_ROWS} more")
        if result.changes:
            print(
                "  (preview shows parse-level amounts; final matrix amounts "
                "scale identically through the transforms)"
            )

        if args.no_rescore:
            print("Rescore skipped (--no-rescore). Run dds-link-all + dds-backtest to apply.")
            return

        if args.fast:
            print("Rescoring (fast path): linked-cache replay + dds-backtest...")
            FastRescorePipeline(settings=self.settings).run()
        else:
            print("Rescoring: dds-link-all + dds-backtest (this takes a while)...")
            LinkAllPipeline(self.settings).run()
        BacktestPipeline(self.settings, options=BacktestOptions(solver=args.solver)).run()
        print("Rescore complete — dashboard data refreshed.")


def main() -> int:
    return SetParameterCli().run()


if __name__ == "__main__":
    sys.exit(main())
