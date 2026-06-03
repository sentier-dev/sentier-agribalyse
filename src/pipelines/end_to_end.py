"""``EndToEndPipeline`` — link → register methods → score a sample of products."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from core import StepTimer
from core.logging import Logging
from pipelines.link_all import LinkAllOptions, LinkAllPipeline
from reporting.run_report import RunReport
from scoring.native_scorer import NativeLciaScorer
from scoring.product_catalog import ProductCatalog
from scoring.scoring_package import ScoringPackageStore


@dataclass(frozen=True)
class EndToEndOptions:
    skip_linking: bool = False
    skip_ecoinvent: bool = False
    solver: str = "pardiso"
    n_sample_products: int = 5


@dataclass
class EndToEndPipeline:
    settings: Settings
    options: EndToEndOptions = field(default_factory=EndToEndOptions)

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def run(self) -> dict[str, Any]:
        s = self.settings
        s.paths.ensure_runtime_dirs()

        if not self.options.skip_linking:
            link_opts = LinkAllOptions(skip_ecoinvent=self.options.skip_ecoinvent)
            report = LinkAllPipeline(s, options=link_opts).run()
        else:
            # Reuse the existing run_report so we don't clobber the
            # scoring_package metadata (content_hash, matrix_shape, …)
            # that downstream tools like ``dds-backtest`` read back.
            report = RunReport.load(s.paths.dashboard_run_report)

        with StepTimer(self._log, "endtoend.score"):
            scores = self._score_sample()
        report.add_stage("sample_scores", scores)
        report.write(s.paths.dashboard_run_report)
        return {"report": report.as_dict(), "scores": scores}

    def _score_sample(self) -> dict[str, Any]:
        """Score a small sample of products from the product catalog.

        Uses NativeLciaScorer against the current ScoringPackage.
        Reads the content_hash from dashboard/run_report.json if present;
        returns a warning dict if no package is available yet.
        """
        s = self.settings
        report_path = s.paths.dashboard_run_report
        if not report_path.exists():
            return {"warning": "no run_report.json — run dds-link-all first"}
        try:
            report_data = json.loads(report_path.read_text())
            content_hash = report_data["stages"]["scoring_package"]["content_hash"]
        except (KeyError, json.JSONDecodeError):
            return {"warning": "run_report.json has no stages.scoring_package.content_hash"}

        pkg = ScoringPackageStore(root=s.paths.scoring_packages_root).read(content_hash)
        ef_keys = sorted(
            m for m in pkg.methods if isinstance(m, tuple) and len(m) == 4 and m[1] == "EF v3.1"
        )
        if not ef_keys:
            return {"warning": "no EF v3.1 methods in scoring package"}

        catalog_df = ProductCatalog.load(s.paths.registry_product_catalog)._df
        # Skip ``[Dummy]`` placeholder activities — they have no linked
        # exchanges (``drop_unlinked`` strips all of them), so scoring
        # them returns trivially-zero results that look like a bug.
        # The smoke test should exercise the supply-chain solve.
        scoreable = catalog_df[
            catalog_df["type"].isin({"process", "multifunctional"})
            & ~catalog_df["name"].str.startswith("[Dummy]")
        ]
        products_df = scoreable.head(self.options.n_sample_products)

        products: list[tuple[tuple[str, str], int]] = [
            ((str(row.database), str(row.code)), int(row.product_id))
            for row in products_df.itertuples(index=False)
        ]
        if not products:
            return {"warning": "product catalog is empty"}

        use_pardiso = self.options.solver == "pardiso"
        scorer = NativeLciaScorer(package=pkg, use_pardiso=use_pardiso)
        results = scorer.score(products, ef_keys)
        return {
            "sampled": len(products),
            "methods": len(ef_keys),
            "scores": [
                {
                    "key": list(key),
                    "skip_reason": r.skip_reason,
                    "elapsed_s": round(r.elapsed_s, 2),
                    "scores": (
                        {k[2]: v for k, v in r.scores.items()} if r.scores is not None else None
                    ),
                }
                for key, r in results.items()
            ],
        }
