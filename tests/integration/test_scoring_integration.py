"""Integration test: native scoring slice on a tiny synthetic system.

Wires up NativeLciaScorer → ScoringPackageBuilder with mocked-at-the-matrix
boundary. The test verifies the assembled scoring path produces coherent
ScoringResult dicts.

Legacy bw2calc-based LciaScorer tests have been removed (REFACTOR_LINKING
phase L4). See tests/unit/test_scoring_native.py for detailed unit coverage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reporting import RunReport
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.native_scorer import NativeLciaScorer
from scoring.scoring_package import ScoringPackageBuilder


def _activity(database, code, name, exchanges, **extra):
    return {"database": database, "code": code, "name": name, "exchanges": exchanges, **extra}


def _build_package(method_key: tuple):
    sp_data = [
        _activity(
            "agb",
            "wheat",
            "wheat",
            exchanges=[
                {"type": "production", "input": ("agb", "wheat-product"), "amount": 1.0},
                {"type": "biosphere", "input": ("bio3", "co2"), "amount": 1.0},
            ],
        ),
        _activity(
            "agb",
            "tomato",
            "tomato",
            exchanges=[
                {"type": "production", "input": ("agb", "tomato-product"), "amount": 1.0},
                {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
            ],
        ),
    ]
    frame = ExchangeFrameBuilder().from_sp_data(sp_data)
    co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
    cfs = {method_key: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
    return ScoringPackageBuilder().build(frame, cfs)


@pytest.mark.integration
class TestNativeScoringSliceEndToEnd:
    def test_scores_multiple_products_against_method(self):
        method_key = ("climate",)
        package = _build_package(method_key)
        scorer = NativeLciaScorer(package=package, use_pardiso=False)

        products = [
            (("agb", "wheat"), ExchangeFrameBuilder.flow_id_for(("agb", "wheat-product"))),
            (("agb", "tomato"), ExchangeFrameBuilder.flow_id_for(("agb", "tomato-product"))),
        ]
        results = scorer.score(products, [method_key])

        assert set(results.keys()) == {("agb", "wheat"), ("agb", "tomato")}
        for _key, res in results.items():
            assert res.scores is not None
            assert method_key in res.scores
            assert isinstance(res.scores[method_key], float)

    def test_skip_reason_set_for_unknown_product(self):
        method_key = ("climate",)
        package = _build_package(method_key)
        scorer = NativeLciaScorer(package=package, use_pardiso=False)
        results = scorer.score([(("agb", "ghost"), 9_999_999_999)], [method_key])
        assert results[("agb", "ghost")].skip_reason == "product id not in technosphere"
        assert results[("agb", "ghost")].scores is None

    def test_results_can_be_serialised_into_run_report(self):
        method_key = ("climate",)
        package = _build_package(method_key)
        scorer = NativeLciaScorer(package=package, use_pardiso=False)
        product_id = ExchangeFrameBuilder.flow_id_for(("agb", "wheat-product"))
        results = scorer.score([(("agb", "wheat"), product_id)], [method_key])

        report = RunReport()
        report.add_stage(
            "scoring",
            {
                "n_products": len(results),
                "n_skipped": sum(1 for r in results.values() if r.scores is None),
                "scores_sample": {
                    "/".join(map(str, k)): {"/".join(m): v for m, v in (r.scores or {}).items()}
                    for k, r in results.items()
                },
            },
        )
        d = report.as_dict()
        assert d["stages"]["scoring"]["n_products"] == 1
        assert d["stages"]["scoring"]["n_skipped"] == 0
        assert "agb/wheat" in d["stages"]["scoring"]["scores_sample"]
