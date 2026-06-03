"""``dds-reset`` — clear the scoring-package cache.

Deletes ``cache/scoring_packages/`` so the next ``dds-link-all`` run
regenerates all matrices from the SimaPro CSV. The biosphere catalog,
EF flows, and method-CF registries are NOT cleared — they are source-
derived parquets that only change when you re-run ``dds-build-*``.

Use this when:
- You want to force a full re-link without changing the mappings.
- You suspect the scoring package is stale (e.g. after a code refactor).
"""

from __future__ import annotations

import argparse
import shutil
from typing import ClassVar

from cli._base import BaseCli
from core.logging import Logging


class ResetCli(BaseCli):
    PROG: ClassVar[str] = "cli.reset"
    DESCRIPTION: ClassVar[str] = (
        "Clear cache/scoring_packages/ so the next dds-link-all regenerates."
    )

    def execute(self, args: argparse.Namespace) -> None:
        log = Logging.get(self.PROG)
        target = self.settings.paths.scoring_packages_root
        if target.exists():
            shutil.rmtree(target)
            log.info("reset.done", path=str(target))
        else:
            log.info("reset.noop", path=str(target), reason="dir absent")


def main() -> int:
    return ResetCli().run()


if __name__ == "__main__":
    raise SystemExit(main())
