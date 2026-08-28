"""``dds-clear-parameters`` — remove parameter overrides.

No arguments clears every override (the file is deleted, restoring the
pristine baseline). ``--name``/``--product`` narrow the removal. Rerun
``dds-link-all`` + ``dds-backtest`` afterwards to rescore the baseline.
"""

from __future__ import annotations

import argparse
import sys
from typing import ClassVar

from cli._base import BaseCli
from transforms import ParameterOverridesStore


class ClearParametersCli(BaseCli):
    PROG: ClassVar[str] = "cli.clear_parameters"
    DESCRIPTION: ClassVar[str] = "Remove parameter overrides (all, by name, or by process)."

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument("--name", default=None, help="Only overrides of this parameter name.")
        p.add_argument(
            "--product",
            metavar="CODE",
            default=None,
            help="Only overrides scoped to this process code.",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        store = ParameterOverridesStore(settings=self.settings)
        removed = store.clear(name=args.name, scope=args.product)
        print(f"Removed {removed} override(s).")
        if removed:
            print("Run dds-link-all + dds-backtest to rescore without them.")


def main() -> int:
    return ClearParametersCli().run()


if __name__ == "__main__":
    sys.exit(main())
