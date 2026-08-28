"""``dds-reset`` — restore the pristine baseline state.

Deletes ``cache/scoring_packages/``, ``cache/importer_cache.pkl``,
``cache/linked_cache.pkl``, and ``source/parameter_overrides.csv`` so the
next ``dds-link-all`` run re-parses the SimaPro CSV and regenerates all
matrices from it, with no parameter what-ifs applied. The biosphere catalog, EF flows, and
method-CF registries are NOT cleared — they are source-derived parquets
that only change when you re-run ``dds-build-*``.

Use this when:
- You modified ``source/AGB32_final.CSV`` — the importer pickle carries
  no hash of the CSV, so without a reset the old parse is silently reused.
- You want to force a full re-link without changing the mappings.
- You suspect the scoring package is stale (e.g. after a code refactor).
- You want to drop all parameter overrides (``--keep-overrides`` opts out;
  ``dds-clear-parameters`` removes them without touching the caches).
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
        "Clear cache/scoring_packages/, cache/importer_cache.pkl, "
        "cache/linked_cache.pkl, and source/parameter_overrides.csv so the "
        "next dds-link-all rebuilds the pristine baseline."
    )

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--keep-overrides",
            action="store_true",
            help="Preserve source/parameter_overrides.csv (default: deleted).",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        log = Logging.get(self.PROG)
        target = self.settings.paths.scoring_packages_root
        if target.exists():
            shutil.rmtree(target)
            log.info("reset.done", path=str(target))
        else:
            log.info("reset.noop", path=str(target), reason="dir absent")

        for pkl in (
            self.settings.paths.importer_cache_pkl,
            self.settings.paths.linked_cache_pkl,
        ):
            if pkl.exists():
                pkl.unlink()
                log.info("reset.done", path=str(pkl))
            else:
                log.info("reset.noop", path=str(pkl), reason="file absent")

        overrides = self.settings.paths.parameter_overrides_csv
        if args.keep_overrides:
            log.info("reset.noop", path=str(overrides), reason="--keep-overrides")
        elif overrides.exists():
            overrides.unlink()
            log.info("reset.done", path=str(overrides))
        else:
            log.info("reset.noop", path=str(overrides), reason="file absent")


def main() -> int:
    return ResetCli().run()


if __name__ == "__main__":
    raise SystemExit(main())
