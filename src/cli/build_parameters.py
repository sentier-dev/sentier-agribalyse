"""``dds-build-parameters`` — materialize the parameter/formula parquets.

Writes ``registry/parameters.parquet`` (per-process parameter definitions)
and ``registry/exchange_formulas.parquet`` (per-exchange formula table)
from the parsed CSV. These are the artifacts later ported packages
(bundle, public twin) consume, and the fast source for
``dds-list-parameters``. Rebuild after replacing the SimaPro CSV.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import ClassVar

import pandas as pd

from cli._base import BaseCli
from core.parquet_io import ParquetAtomicWriter
from transforms import SimaProImporter


class BuildParametersCli(BaseCli):
    PROG: ClassVar[str] = "cli.build_parameters"
    DESCRIPTION: ClassVar[str] = (
        "Write registry/parameters.parquet and registry/exchange_formulas.parquet."
    )

    def execute(self, args: argparse.Namespace) -> None:
        sp = SimaProImporter(self.settings).load()

        params = pd.DataFrame(sp.parameters)
        ParquetAtomicWriter.write(params, self.settings.paths.registry_parameters)
        print(f"{self.settings.paths.registry_parameters}: {len(params)} rows")

        # Reference detection derives from the ACTUAL extracted names per
        # process (word-boundary match, the same predicate the reevaluator
        # uses) — never from a hardcoded naming-convention regex, which
        # would silently under-report if bw_simapro_csv's normalization
        # ever changes.
        names_by_process: dict[str, list[str]] = {}
        for row in sp.parameters:
            names_by_process.setdefault(row["process_code"], []).append(row["name"])

        rows: list[dict] = []
        for ds in sp.data:
            code = ds.get("code")
            known = names_by_process.get(code, [])
            for idx, exc in enumerate(ds.get("exchanges", []) or []):
                formula = exc.get("formula")
                if not formula:
                    continue
                rows.append(
                    {
                        "process_code": code,
                        "exchange_index": idx,
                        "exchange_name": exc.get("name", ""),
                        "exchange_type": exc.get("type", ""),
                        "formula": formula,
                        "baked_amount": exc.get("amount"),
                        "referenced_parameters": sorted(
                            n for n in set(known) if re.search(rf"\b{re.escape(n)}\b", formula)
                        ),
                    }
                )
        formulas = pd.DataFrame(rows)
        ParquetAtomicWriter.write(formulas, self.settings.paths.registry_exchange_formulas)
        print(f"{self.settings.paths.registry_exchange_formulas}: {len(formulas)} rows")


def main() -> int:
    return BuildParametersCli().run()


if __name__ == "__main__":
    sys.exit(main())
