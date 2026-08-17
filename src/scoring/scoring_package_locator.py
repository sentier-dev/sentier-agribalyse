"""``ScoringPackageLocator`` — run_report.json → cached ``ScoringPackage``.

The link pipeline records the content hash of the ScoringPackage it built
under ``stages.scoring_package.content_hash`` in ``dashboard/run_report.json``.
Every consumer that wants "the package of the last link run" (backtest,
decompose, the Brightway export) resolves it the same way; this class is that
one shared way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from config import Settings
from scoring.scoring_package import ScoringPackage, ScoringPackageStore


@dataclass(frozen=True)
class ScoringPackageLocator:
    """Resolve and load the ScoringPackage referenced by the last link run."""

    settings: Settings = field(default_factory=Settings)

    def content_hash(self) -> str:
        report_path = self.settings.paths.dashboard_run_report
        if not report_path.exists():
            raise FileNotFoundError(
                f"run_report.json not found at {report_path}. Run `dds-link-all` first."
            )
        report = json.loads(report_path.read_text())
        try:
            return report["stages"]["scoring_package"]["content_hash"]
        except KeyError as exc:
            raise KeyError(
                f"run_report.json at {report_path} has no stages.scoring_package.content_hash. "
                f"Re-run `dds-link-all` to regenerate it."
            ) from exc

    def load(self) -> ScoringPackage:
        return ScoringPackageStore(root=self.settings.paths.scoring_packages_root).read(
            self.content_hash()
        )
