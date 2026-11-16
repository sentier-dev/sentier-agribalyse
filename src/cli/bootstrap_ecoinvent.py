"""``dds-bootstrap-ecoinvent`` — everything BOOTSTRAP.md asks for, in one command.

Downloads ecoinvent with the credentials in ``.env``, imports it into a
private Brightway project under ``cache/``, and writes the four
ecoinvent-derived files under ``source/``. Needs the ``bootstrap`` extra.
Running it again reuses the imported project and only rewrites the files;
``--fresh`` starts from an empty project.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from typing import ClassVar

from bootstrap import EcoinventBootstrapper
from cli._base import BaseCli


class BootstrapEcoinventCli(BaseCli):
    PROG: ClassVar[str] = "cli.bootstrap_ecoinvent"
    DESCRIPTION: ClassVar[str] = (
        "Download ecoinvent with your licence credentials and write the four "
        "ecoinvent-derived snapshots under source/. One-time, needs the bootstrap extra."
    )
    BRIGHTWAY_DIR_ENV: ClassVar[str] = "BRIGHTWAY2_DIR"

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = super().parser()
        p.add_argument(
            "--fresh",
            action="store_true",
            help=(
                "Delete the private Brightway project before starting and download "
                "ecoinvent again. Use it after an interrupted import."
            ),
        )
        p.add_argument(
            "--discard-project",
            action="store_true",
            help=(
                "Delete the private Brightway project (about 1.5 GB under cache/) once the "
                "snapshots are written. The next run downloads ecoinvent again."
            ),
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        paths = self.settings.paths
        brightway_dir = paths.bootstrap_brightway_dir
        if args.fresh:
            shutil.rmtree(brightway_dir, ignore_errors=True)
        brightway_dir.mkdir(parents=True, exist_ok=True)
        # Must be set before bw2data is imported: it picks its data directory
        # at import time. A private directory keeps the user's projects apart.
        os.environ[self.BRIGHTWAY_DIR_ENV] = str(brightway_dir)
        bd, bi = self._brightway()
        report = EcoinventBootstrapper(settings=self.settings, bd=bd, bi=bi).run(
            os.environ.get("ECOINVENT_USERNAME"), os.environ.get("ECOINVENT_PASSWORD")
        )
        state = "downloaded and imported" if report["imported"] else "already imported, reused"
        print(f"ecoinvent {self.settings.resolved_ecoinvent_version}: {state}")
        print(
            f"Wrote {report['exchanges']} exchanges, {report['activities']} activities, "
            f"{report['biosphere_flows']} elementary flows, {report['methods']} EF methods "
            f"({report['method_cfs']} factors) under {paths.source}"
        )
        print("These files are licensed ecoinvent data: gitignored, never commit them.")
        if args.discard_project:
            shutil.rmtree(brightway_dir, ignore_errors=True)
            print(f"Removed {brightway_dir}")
        print("Next: uv run dds-fetch-harmonised-flows, then uv run dds-build-registry")

    @staticmethod
    def _brightway() -> tuple[object, object]:
        try:
            return importlib.import_module("bw2data"), importlib.import_module("bw2io")
        except ImportError as exc:
            raise RuntimeError(
                "dds-bootstrap-ecoinvent needs the bootstrap extra: "
                "run `uv sync --extra pardiso --extra bootstrap` first."
            ) from exc


def main() -> int:
    return BootstrapEcoinventCli().run()


if __name__ == "__main__":
    sys.exit(main())
