"""``FastRescorePipeline`` — rebuild the scoring package without relinking.

The full ``dds-link-all`` run has two cost centres: (a) parse + transforms
+ matching/linking, and (b) the scoring-package emit (frame → allocator →
matrices → store). Parameter overrides change exchange *amounts* only, so
(a) is identity-invariant under any override — its output is the pristine
linked-graph snapshot ``LinkedSpCache`` writes (always BEFORE overrides
apply, so the cache is never override-tainted). This pipeline loads that
snapshot, applies the overrides store with the SAME
``ParameterOverridesApplier`` (ratio mode) the full path uses, and replays
the SAME emit stage (``LinkAllPipeline.emit_scoring_package``).

Both paths therefore share every override-relevant code path; a
zero-override fast run reproduces the baseline content hash (verified
live 2026-08-17: hash ``b54a0b1f…`` reproduced in 81s vs ~17min full),
and ratio-mode semantics are unit-pinned in
``tests/unit/test_transforms_parameter_reevaluator.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from core.logging import Logging, StepTimer
from pipelines.link_all import LinkAllPipeline
from reporting import RunReport
from transforms.linked_cache import LinkedSpCache
from transforms.parameter_overrides import ParameterOverridesApplier


@dataclass
class FastRescorePipeline:
    settings: Settings = field(default_factory=Settings)

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def run(self) -> RunReport:
        s = self.settings
        log = self._log
        # Reuse the existing run_report (same pattern as EndToEndPipeline)
        # so the fast path doesn't clobber settings_summary, coverage, and
        # link-stage metadata from the last full run.
        report_path = s.paths.dashboard_run_report
        report = RunReport.load(report_path) if report_path.exists() else RunReport()
        # The reused report keeps baseline link stages; stamp THIS run so the
        # dashboard doesn't show the old run's time next to a new hash.
        report.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report.add_stage("fast_rescore", {"fast_rescore": True})

        with StepTimer(log, "fast_rescore.linked_cache.load"):
            sp = LinkedSpCache(settings=s).load()

        with StepTimer(log, "fast_rescore.parameter_overrides"):
            override_stats = ParameterOverridesApplier(settings=s).apply(sp)
        report.add_stage("parameter_overrides", override_stats)

        with StepTimer(log, "fast_rescore.scoring_package.write"):
            payload = LinkAllPipeline.emit_scoring_package(sp, s, report)
        report.add_stage("scoring_package", payload)
        # Persist like link_all does — dds-backtest / dds-decompose-score
        # resolve the active package via the content_hash recorded here.
        report.write(report_path)
        log.info("fast_rescore.done", content_hash=payload.get("content_hash"))
        return report
