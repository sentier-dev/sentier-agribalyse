"""Smoke tests for ``DecomposeScoreCli``.

The CLI is a thin orchestration layer: load package → call decomposer →
print summary. We verify (1) the alias resolution from short → full
4-tuple, (2) the parser accepts the documented flags, and (3)
``execute`` runs end-to-end against a tiny synthetic package laid out
under a temporary settings directory.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cli.decompose_score import DecomposeScoreCli, MethodAliases
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.scoring_package import ScoringPackageBuilder, ScoringPackageStore


class TestMethodAliases:
    def test_known_short_resolves_to_full_tuple(self):
        full = MethodAliases.resolve("climate")
        assert full == (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )

    def test_unknown_alias_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown method alias"):
            MethodAliases.resolve("not-a-real-method")

    def test_comma_separated_passes_through_as_full_tuple(self):
        full = MethodAliases.resolve("a,b,c,d")
        assert full == ("a", "b", "c", "d")


class TestDecomposeScoreCliParser:
    def test_parser_accepts_required_flags(self):
        parser = DecomposeScoreCli.parser()
        args = parser.parse_args(
            [
                "--database",
                "agb",
                "--code",
                "p1",
                "--method",
                "climate",
            ]
        )
        assert args.database == "agb"
        assert args.code == "p1"
        assert args.method == "climate"
        assert args.top_n is None
        assert args.inventory is False

    def test_parser_accepts_optional_flags(self):
        parser = DecomposeScoreCli.parser()
        args = parser.parse_args(
            [
                "--database",
                "agb",
                "--code",
                "p1",
                "--method",
                "climate",
                "--top-n",
                "5",
                "--inventory",
            ]
        )
        assert args.top_n == 5
        assert args.inventory is True


class TestDecomposeScoreCliExecute:
    """Wire the CLI through a tiny on-disk scoring package + product catalog."""

    @staticmethod
    def _setup_project(tmp_path):
        from config import Settings

        settings = Settings()
        # We don't have a permissive constructor that takes paths; instead we
        # build the artefacts under tmp_path and monkeypatch the settings
        # paths via a custom CLI that consumes a pre-built Settings instance.
        # The simplest path is to monkey-patch via an injected cli object.
        cache = tmp_path / "cache" / "scoring_packages"
        cache.mkdir(parents=True)
        registry = tmp_path / "registry"
        registry.mkdir(parents=True)
        dashboard = tmp_path / "dashboard"
        dashboard.mkdir(parents=True)

        # 1) Build a tiny package: wheat → bio3.co2 with CF=1 on climate.
        sp_data = [
            {
                "database": "agb",
                "code": "wheat",
                "name": "wheat",
                "exchanges": [
                    {"type": "production", "input": ("agb", "wheat-prod"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
                ],
            },
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        method = MethodAliases.resolve("climate")
        cfs = {method: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        pkg = ScoringPackageBuilder().build(frame, cfs)
        ScoringPackageStore(root=cache).write(pkg)

        # 2) run_report.json pointing at the package.
        import json

        (dashboard / "run_report.json").write_text(
            json.dumps({"stages": {"scoring_package": {"content_hash": pkg.content_hash}}})
        )

        # 3) product_catalog.parquet.
        pc_df = pd.DataFrame(
            [
                {
                    "database": "agb",
                    "code": "wheat",
                    "name": "wheat",
                    "type": "process",
                    "unit": "kg",
                    "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "wheat-prod")),
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

        # 4) biosphere_catalog.parquet — minimal columns the labeller needs.
        bio_df = pd.DataFrame(
            [
                {
                    "database": "bio3",
                    "code": "co2",
                    "name": "Carbon dioxide",
                    "categories": ["air"],
                    "cas": "",
                }
            ]
        )
        bio_df.to_parquet(registry / "biosphere_catalog.parquet", index=False)

        return tmp_path, settings, pkg

    def test_execute_prints_score_and_contributions(self, tmp_path, monkeypatch, capsys):
        tmp_path, _settings, _pkg = self._setup_project(tmp_path)
        from tests.fixtures.builders import make_settings

        cli = DecomposeScoreCli(settings=make_settings(tmp_path))

        rc = cli.run(
            [
                "--database",
                "agb",
                "--code",
                "wheat",
                "--method",
                "climate",
                "--top-n",
                "3",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        # Score is 2.0 (CO2=2, CF=1).
        assert "score = 2" in captured.out
        # CO2 should appear in the flow contributions.
        assert "Carbon dioxide" in captured.out
        # The activity should also appear.
        assert "wheat" in captured.out

    def test_execute_with_inventory_dumps_uncharacterised_flows(
        self, tmp_path, monkeypatch, capsys
    ):
        tmp_path, _settings, _pkg = self._setup_project(tmp_path)
        from tests.fixtures.builders import make_settings

        cli = DecomposeScoreCli(settings=make_settings(tmp_path))
        rc = cli.run(
            [
                "--database",
                "agb",
                "--code",
                "wheat",
                "--method",
                "climate",
                "--inventory",
                "--top-n",
                "5",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "INVENTORY" in captured.out
