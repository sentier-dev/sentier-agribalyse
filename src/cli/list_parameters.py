"""``dds-list-parameters`` — browse SimaPro parameters and active overrides.

Aggregates the process-local parameter definitions by SimaPro-facing name
(577 distinct in AGB 3.2). Reads ``registry/parameters.parquet`` when
``dds-build-parameters`` has materialized it, otherwise falls back to the
importer cache (which re-parses the CSV when absent).
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

import pandas as pd

from cli._base import BaseCli
from transforms import ParameterOverridesStore, SimaProImporter


class ListParametersCli(BaseCli):
    PROG: ClassVar[str] = "cli.list_parameters"
    DESCRIPTION: ClassVar[str] = "List SimaPro parameter names, definition counts, and overrides."

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--name-like",
            default=None,
            help="Case-insensitive substring filter on the parameter name.",
        )
        p.add_argument(
            "--product",
            metavar="CODE",
            default=None,
            help="Show only parameters defined by this process, with their values.",
        )
        return p

    def _load_frame(self) -> pd.DataFrame:
        parquet = self.settings.paths.registry_parameters
        if parquet.exists():
            return pd.read_parquet(parquet)
        sp = SimaProImporter(self.settings).load()
        return pd.DataFrame(sp.parameters)

    def execute(self, args: argparse.Namespace) -> None:
        df = self._load_frame()
        if df.empty:
            print("No parameters found — is the CSV parsed? (run dds-link-all once)")
            return
        overrides = ParameterOverridesStore(settings=self.settings).load()
        # Keyed by (name, scope); lookup mirrors ParameterReevaluator.plan()
        # precedence: process-specific beats "*".
        overridden = {(ov.parameter_name.lower(), ov.scope): ov for ov in overrides}

        def override_for(name: str, product: str | None) -> object | None:
            if product is not None:
                specific = overridden.get((name.lower(), product))
                if specific is not None:
                    return specific
            return overridden.get((name.lower(), "*"))

        if args.name_like:
            df = df[df["original_name"].str.contains(args.name_like, case=False, na=False)]
        if args.product:
            df = df[df["process_code"] == args.product]
            for row in df.itertuples():
                ov = override_for(str(row.original_name), args.product)
                mark = f"  [override: {ov.value} @ {ov.scope}]" if ov else ""
                formula = f"  = {row.formula}" if row.kind == "calculated" else ""
                print(f"{row.original_name:45s} {row.kind:10s} {row.amount!r}{formula}{mark}")
            print(f"\n{len(df)} parameters on process {args.product}")
            return

        grouped = (
            df.groupby(["original_name", "kind"])
            .agg(
                definitions=("process_code", "nunique"),
                amount_min=("amount", "min"),
                amount_max=("amount", "max"),
            )
            .reset_index()
            .sort_values(["kind", "original_name"])
        )
        print(f"{'name':45s} {'kind':10s} {'defs':>6s} {'min':>12s} {'max':>12s}  override")
        for row in grouped.itertuples():
            # Aggregate view: show every active override for the name
            # (there can be one "*" row plus process-scoped rows).
            mark = "; ".join(
                f"{o.value} @ {o.scope}"
                for o in overrides
                if o.parameter_name.lower() == str(row.original_name).lower()
            )
            print(
                f"{row.original_name:45s} {row.kind:10s} {row.definitions:>6d} "
                f"{row.amount_min!r:>12s} {row.amount_max!r:>12s}  {mark}"
            )
        print(
            f"\n{grouped['original_name'].nunique()} distinct names; {len(overrides)} override(s) active"
        )


def main() -> int:
    return ListParametersCli().run()


if __name__ == "__main__":
    sys.exit(main())
