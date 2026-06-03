"""``dds-build-skeleton`` — produce the AGB-only skeleton for the bundle.

Wraps :class:`SkeletonExtractor`. Reads an existing scoring package
from ``cache/scoring_packages/<hash>/``, writes the skeleton (with
ecoinvent columns zeroed and ``ecoinvent_slot_index.parquet`` emitted)
to ``<bundle-root>/skeleton/<hash>/``.

Run AFTER ``dds-link-all`` produced a verified scoring package.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from cli._base import BaseCli
from config import Settings
from core.logging import Logging
from scoring.skeleton_extractor import SkeletonExtractor


@dataclass
class BuildSkeletonCli(BaseCli):
    """CLI that extracts a skeleton + slot-index from a cached scoring
    package and writes it under the customer-bundle's ``skeleton/`` root."""

    PROG: ClassVar[str] = "dds-build-skeleton"
    DESCRIPTION: ClassVar[str] = (
        "Strip ecoinvent IP out of a built scoring package and emit the "
        "AGB-only skeleton consumed by sentier_agribalyse-bundle."
    )

    settings: Settings = field(default_factory=Settings)

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog=cls.PROG, description=cls.DESCRIPTION)
        p.add_argument(
            "--content-hash",
            required=True,
            help="Content hash of the source scoring package under cache/scoring_packages/.",
        )
        p.add_argument(
            "--bundle-root",
            type=Path,
            required=True,
            help="Root of the sentier_agribalyse-bundle checkout (skeleton/ will be written there).",
        )
        return p

    def execute(self, args: argparse.Namespace) -> None:
        log = Logging.get(self.PROG)
        paths = self.settings.paths
        source = paths.scoring_packages_root / args.content_hash
        if not source.exists():
            raise FileNotFoundError(
                f"Source scoring package not found at {source}. "
                f"Run dds-link-all first to materialise it."
            )
        bundle_root = args.bundle_root.resolve()
        if not bundle_root.exists():
            raise FileNotFoundError(f"Bundle root does not exist: {bundle_root}")
        output = bundle_root / "skeleton" / args.content_hash
        output.parent.mkdir(parents=True, exist_ok=True)

        extractor = SkeletonExtractor(ecoinvent_catalog_path=paths.registry_ecoinvent_catalog)
        extractor.extract(source_dir=source, output_dir=output)
        log.info("skeleton.cli.done", source=str(source), output=str(output))


def main() -> int:
    return BuildSkeletonCli().run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
