"""End-to-end smoke for ``dds-build-flow-decomp``.

Builds a tiny on-disk project (same shape as ``test_cli_decompose_score``)
and verifies (1) parser accepts the documented flags with the right
defaults, (2) ``execute`` writes one JSON per mapped product under the
``--out`` directory, (3) the JSON has the expected top-level keys.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cli.build_flow_decomp import BuildFlowDecompCli
from cli.decompose_score import MethodAliases
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.scoring_package import ScoringPackageBuilder, ScoringPackageStore
from tests.fixtures.builders import make_settings


def _setup_project(tmp_path):
    cache = tmp_path / "cache" / "scoring_packages"
    cache.mkdir(parents=True)
    registry = tmp_path / "registry"
    registry.mkdir(parents=True)
    dashboard_backtest = tmp_path / "dashboard" / "backtest"
    dashboard_backtest.mkdir(parents=True)

    sp_data = [
        {
            "database": "agribalyse-3.2",
            "code": "1",
            "name": "wheat",
            "exchanges": [
                {"type": "production", "input": ("agribalyse-3.2", "1-prod"), "amount": 1.0},
                {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
            ],
        },
    ]
    frame = ExchangeFrameBuilder().from_sp_data(sp_data)
    co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
    climate = MethodAliases.resolve("climate")
    cfs = {climate: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
    pkg = ScoringPackageBuilder().build(frame, cfs)
    ScoringPackageStore(root=cache).write(pkg)

    (tmp_path / "dashboard" / "run_report.json").write_text(
        json.dumps({"stages": {"scoring_package": {"content_hash": pkg.content_hash}}})
    )

    pc_df = pd.DataFrame(
        [
            {
                "database": "agribalyse-3.2",
                "code": "1",
                "name": "wheat",
                "type": "process",
                "unit": "kg",
                "product_id": ExchangeFrameBuilder.flow_id_for(("agribalyse-3.2", "1-prod")),
            }
        ]
    ).astype(
        {
            "database": "string",
            "code": "string",
            "name": "string",
            "type": "string",
            "unit": "string",
            "product_id": "int64",
        }
    )
    pc_df.to_parquet(registry / "product_catalog.parquet", index=False)

    bio_df = pd.DataFrame(
        [
            {
                "database": "bio3",
                "code": "co2",
                "name": "Carbon dioxide",
                "categories": ["air"],
                "unit": "kg",
                "cas": "",
            }
        ]
    )
    bio_df.to_parquet(registry / "biosphere_catalog.parquet", index=False)

    scores_df = pd.DataFrame(
        {
            "Code AGB": ["1"],
            "mapped": [True],
            "Nom du Produit": ["wheat"],
            "product_db": ["agribalyse-3.2"],
            "product_code": ["1"],
        }
    )
    scores_df.to_parquet(dashboard_backtest / "scores.parquet", index=False)


class TestBuildFlowDecompCliParser:
    def test_defaults(self):
        parser = BuildFlowDecompCli.parser()
        args = parser.parse_args([])
        assert args.top_n == 25
        assert args.solver == "pardiso"
        assert args.out is None  # resolved against settings in execute()

    def test_overrides(self):
        parser = BuildFlowDecompCli.parser()
        args = parser.parse_args(["--top-n", "5", "--solver", "scipy", "--out", "/tmp/x"])
        assert args.top_n == 5
        assert args.solver == "scipy"
        assert str(args.out) == "/tmp/x"


class TestBuildFlowDecompCliExecute:
    def test_writes_one_json_per_mapped_product(self, tmp_path):
        _setup_project(tmp_path)
        cli = BuildFlowDecompCli(settings=make_settings(tmp_path))

        rc = cli.run(["--solver", "scipy"])

        assert rc == 0
        out_file = tmp_path / "dashboard" / "decomp" / "1.json"
        assert out_file.exists()
        payload = json.loads(out_file.read_text())
        assert payload["code"] == "1"
        assert "climate" in payload["methods"]
        assert payload["methods"]["climate"]["score"] == pytest.approx(2.0)
