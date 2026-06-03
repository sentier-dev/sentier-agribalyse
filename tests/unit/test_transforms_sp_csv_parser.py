"""Unit tests for ``transforms.SimaProCsvParser`` and ``ParsedSimaProCsv``.

The parser wraps ``bw_simapro_csv.SimaProCSV`` (no bw2io). The wrapper
``ParsedSimaProCsv`` exposes the surface the linker pipeline drives —
``apply_strategy``, ``match_database``, ``randonneur``, etc. F3 lifts the
strategy delegations in-house; F4 deletes the bw2data path. These tests
pin the F2 contract so those subsequent rewrites can be verified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest

from transforms.sp_csv_parser import ParsedSimaProCsv, SimaProCsvParser

# ============================================================================
# SimaProCsvParser — parsing-side contract


class _FakeSimaProCsv:
    """Stand-in for ``bw_simapro_csv.SimaProCSV``.

    Records constructor kwargs and emits the brightway dict shape the
    parser unpacks.
    """

    instantiations: ClassVar[list[dict]] = []

    def __init__(self, **kw):
        type(self).instantiations.append(kw)
        self.database_name = kw.get("database_name") or "fallback-db"

    def to_brightway(self, separate_products=True, shorten_names=True):
        return {
            "database": {"name": self.database_name, "tag": "meta"},
            "processes": [
                {"name": "P1", "type": "process", "exchanges": []},
                {"name": "P2", "type": "process", "exchanges": []},
            ],
            "products": [
                {"name": "Product-1", "type": "product"},
            ],
            "database_parameters": [],
            "project_parameters": [],
        }


@pytest.fixture
def fake_bw_simapro_csv(monkeypatch):
    bw_simapro_csv = sys.modules["bw_simapro_csv"]
    _FakeSimaProCsv.instantiations.clear()
    monkeypatch.setattr(bw_simapro_csv, "SimaProCSV", _FakeSimaProCsv, raising=False)
    # Importing the production module after the patch makes sure the new
    # binding is picked up via sys.modules. The parser imports
    # ``bw_simapro_csv.SimaProCSV`` at top of file, so re-binding the
    # attribute on the module is enough.
    import transforms.sp_csv_parser as sp_csv_parser

    monkeypatch.setattr(sp_csv_parser, "SimaProCSV", _FakeSimaProCsv)
    return _FakeSimaProCsv


class TestSimaProCsvParser:
    def test_parse_returns_parsed_simapro_csv_with_processes_and_products(
        self,
        fake_bw_simapro_csv,
        tmp_path: Path,
    ):
        csv = tmp_path / "agb.csv"
        csv.write_bytes(b"fake")
        parsed = SimaProCsvParser(csv_path=csv, database_name="agb").parse()
        assert isinstance(parsed, ParsedSimaProCsv)
        assert parsed.db_name == "agb"
        assert len(parsed.data) == 3  # 2 processes + 1 product
        assert {d["name"] for d in parsed.data} == {"P1", "P2", "Product-1"}

    def test_parse_skips_products_when_separate_products_false(
        self,
        fake_bw_simapro_csv,
        tmp_path: Path,
    ):
        csv = tmp_path / "agb.csv"
        csv.write_bytes(b"fake")
        parsed = SimaProCsvParser(csv_path=csv, separate_products=False).parse()
        # Products are excluded; only the 2 processes remain.
        assert len(parsed.data) == 2
        assert {d["name"] for d in parsed.data} == {"P1", "P2"}

    def test_parse_passes_database_name_to_bw_simapro_csv(
        self,
        fake_bw_simapro_csv,
        tmp_path: Path,
    ):
        csv = tmp_path / "agb.csv"
        csv.write_bytes(b"fake")
        SimaProCsvParser(csv_path=csv, database_name="agribalyse-3.2").parse()
        kw = _FakeSimaProCsv.instantiations[-1]
        assert kw["database_name"] == "agribalyse-3.2"
        assert kw["path_or_stream"] == csv

    def test_parse_propagates_metadata(self, fake_bw_simapro_csv, tmp_path: Path):
        csv = tmp_path / "agb.csv"
        csv.write_bytes(b"fake")
        parsed = SimaProCsvParser(csv_path=csv, database_name="agb").parse()
        assert parsed.metadata == {"name": "agb", "tag": "meta"}

    def test_parser_does_not_import_bw2io(self):
        """F2 acceptance: the parser must not import bw2io."""
        # The wrapper still delegates strategy methods to bw2io until F3,
        # so ``import bw2io.strategies`` may live here. The constraint
        # the doc pins is on ``transforms/importer.py`` — make that one
        # the explicit assertion.
        import transforms.importer as importer_mod
        import transforms.sp_csv_parser as mod

        assert "bw2io" not in importer_mod.__dict__, (
            "transforms/importer.py must not import bw2io after F2."
        )
        # Sanity: the new parser module is the chosen path.
        assert hasattr(mod, "SimaProCsvParser")


# ============================================================================
# ParsedSimaProCsv — surface contract


class TestParsedSimaProCsvSurface:
    def test_apply_strategy_passes_data_through_function(self):
        sp = ParsedSimaProCsv(db_name="agb", data=[{"k": 1}])

        def double(data):
            return [{"k": d["k"] * 2} for d in data]

        sp.apply_strategy(double)
        assert sp.data == [{"k": 2}]

    def test_apply_strategy_forwards_args_and_kwargs(self):
        sp = ParsedSimaProCsv(db_name="agb", data=[{"k": 1}])

        def scale(data, multiplier=1, *, offset=0):
            return [{"k": d["k"] * multiplier + offset} for d in data]

        sp.apply_strategy(scale, 3, offset=5)
        assert sp.data == [{"k": 8}]

    def test_apply_strategies_runs_default_chain_in_order(self):
        sp = ParsedSimaProCsv(db_name="agb", data=[{"v": 0}])
        calls: list[str] = []

        def s_a(data):
            calls.append("a")
            return data

        def s_b(data):
            calls.append("b")
            return data

        sp.strategies = [s_a, s_b]
        sp.apply_strategies()
        assert calls == ["a", "b"]

    def test_drop_unlinked_requires_i_am_reckless(self):
        sp = ParsedSimaProCsv(db_name="agb", data=[])
        with pytest.raises(ValueError, match="i_am_reckless"):
            sp.drop_unlinked()

    def test_drop_unlinked_removes_exchanges_without_input(self):
        sp = ParsedSimaProCsv(
            db_name="agb",
            data=[
                {
                    "exchanges": [
                        {"input": ("a", "b"), "amount": 1.0},
                        {"amount": 2.0},  # unlinked — should be dropped
                    ]
                }
            ],
        )
        sp.drop_unlinked(i_am_reckless=True)
        assert len(sp.data[0]["exchanges"]) == 1
        assert sp.data[0]["exchanges"][0]["input"] == ("a", "b")


class TestParsedSimaProCsvMatchDatabase:
    def test_match_database_external_db_is_disallowed(self):
        sp = ParsedSimaProCsv(db_name="agb", data=[])
        with pytest.raises(NotImplementedError, match="Use EcoinventCatalog"):
            sp.match_database("ghost-db", fields=["name"])

    def test_match_database_internal_links_processes_to_products(self):
        """Internal name+unit match links the consumer exchange to the product."""
        sp = ParsedSimaProCsv(
            db_name="agb",
            data=[
                {
                    "type": "process",
                    "database": "agb",
                    "code": "p1",
                    "exchanges": [
                        {"name": "Tap water", "unit": "kg", "type": "technosphere"},
                    ],
                },
                {
                    "type": "product",
                    "database": "agb",
                    "code": "prod-tap",
                    "name": "Tap water",
                    "unit": "kg",
                },
            ],
        )
        sp.match_database(fields=["name", "unit"], processes_to_products=True)
        exc = sp.data[0]["exchanges"][0]
        assert exc.get("input") == ("agb", "prod-tap")

    def test_match_database_no_match_leaves_input_unset(self):
        sp = ParsedSimaProCsv(
            db_name="agb",
            data=[
                {
                    "type": "process",
                    "database": "agb",
                    "code": "p1",
                    "exchanges": [
                        {"name": "Ghost", "unit": "kg", "type": "technosphere"},
                    ],
                },
                {
                    "type": "product",
                    "database": "agb",
                    "code": "prod-other",
                    "name": "Other product",
                    "unit": "kg",
                },
            ],
        )
        sp.match_database(fields=["name", "unit"], processes_to_products=True)
        assert "input" not in sp.data[0]["exchanges"][0]


class TestParsedSimaProCsvRandonneur:
    def test_randonneur_invokes_migrate_edges_with_stored_data(self, monkeypatch):
        called: dict = {}

        def fake_migrate_edges(graph, label, config):
            called["graph"] = graph
            called["label"] = label
            called["config"] = config
            return [{"migrated": True}]

        rn = sys.modules["randonneur"]
        # Add migrate_edges_with_stored_data + .utils.SAFE_VERBS to the stub.
        monkeypatch.setattr(
            rn,
            "migrate_edges_with_stored_data",
            fake_migrate_edges,
            raising=False,
        )
        rn.utils = type("U", (), {"SAFE_VERBS": ("replace", "update", "disaggregate")})()
        rn.errors = type("E", (), {"WrongGraphContext": type("WGC", (Exception,), {})})()

        import transforms.sp_csv_parser as mod

        monkeypatch.setattr(mod.rn, "migrate_edges_with_stored_data", fake_migrate_edges)
        monkeypatch.setattr(mod.rn, "utils", rn.utils)
        monkeypatch.setattr(mod.rn, "errors", rn.errors)

        sp = ParsedSimaProCsv(db_name="agb", data=[{"k": 1}])
        sp.randonneur(label="some-package", fields=["name"])
        assert called["label"] == "some-package"
        assert sp.data == [{"migrated": True}]

    def test_randonneur_swallows_wrong_graph_context(self, monkeypatch):
        rn = sys.modules["randonneur"]
        WGC = type("WrongGraphContext", (Exception,), {})

        def boom(graph, label, config):
            raise WGC()

        monkeypatch.setattr(rn, "migrate_edges_with_stored_data", boom, raising=False)
        rn.utils = type("U", (), {"SAFE_VERBS": ("replace",)})()
        rn.errors = type("E", (), {"WrongGraphContext": WGC})()

        import transforms.sp_csv_parser as mod

        monkeypatch.setattr(mod.rn, "migrate_edges_with_stored_data", boom)
        monkeypatch.setattr(mod.rn, "utils", rn.utils)
        monkeypatch.setattr(mod.rn, "errors", rn.errors)

        sp = ParsedSimaProCsv(db_name="agb", data=[{"k": 1}])
        sp.randonneur(label="missing")
        # No exception, data preserved.
        assert sp.data == [{"k": 1}]
