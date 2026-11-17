"""Commands without ``--csv`` read the CSV the last ``dds-link-all`` asked for.

``dds-link-all --csv ecoinvent-free`` with ADEME's full export left in
``source/``, then ``dds-set-parameter ... --fast``: the override is checked
against the processes of the ecoinvent-free parse, nothing is parsed again,
and the fast path replays the link. Stand-in CSV files and a stand-in parser;
the link is ``LinkAllCli`` with a pipeline that does what the real one does
with the CSV (load through the importer, write the linked cache).
"""

from __future__ import annotations

import lzma
from typing import ClassVar

import pytest

from cli.link_all import LinkAllCli
from cli.reset import ResetCli
from cli.set_parameter import SetParameterCli
from transforms.agribalyse_csv_source import AgribalyseCsvSource
from transforms.importer import SimaProImporter
from transforms.linked_cache import LinkedSpCache
from transforms.sp_csv_parser import ParsedSimaProCsv


class _Parser:
    """Stands in for ``SimaProCsvParser``: one process per CSV, named after it."""

    parsed: ClassVar[list[str]] = []

    def __init__(self, csv_path, database_name):
        self.csv_path = csv_path

    def parse(self):
        _Parser.parsed.append(self.csv_path.name)
        code = "P_FREE" if "ecoinvent_free" in self.csv_path.name else "P_FULL"
        return ParsedSimaProCsv(
            db_name="agb",
            data=[
                {
                    "code": code,
                    "type": "process",
                    "exchanges": [
                        {
                            "name": "direct",
                            "amount": 1.0,
                            "formula": "(SP_RATIO * 2)",
                            "type": "technosphere",
                        }
                    ],
                }
            ],
            parameters=[
                {
                    "process_code": code,
                    "kind": "input",
                    "name": "SP_RATIO",
                    "original_name": "ratio",
                    "amount": 0.5,
                    "formula": None,
                    "comment": "",
                }
            ],
        )


class _LinkPipeline:
    """What ``LinkAllPipeline`` does with the CSV: load it, cache the link."""

    def __init__(self, settings, options=None):
        self.settings = settings

    def run(self):
        sp = SimaProImporter(self.settings).load()
        LinkedSpCache(settings=self.settings).write(sp)


@pytest.fixture
def scenario(settings, monkeypatch):
    _Parser.parsed = []
    monkeypatch.setattr("transforms.importer.SimaProCsvParser", _Parser)
    monkeypatch.setattr("cli.link_all.LinkAllPipeline", _LinkPipeline)
    replayed: list = []

    class _FastRescore:
        def __init__(self, settings):
            self.settings = settings

        def run(self):
            replayed.append(LinkedSpCache(settings=self.settings).load())

    class _Backtest:
        def __init__(self, settings, options=None):
            pass

        def run(self):
            return None

    monkeypatch.setattr("cli.set_parameter.FastRescorePipeline", _FastRescore)
    monkeypatch.setattr("cli.set_parameter.BacktestPipeline", _Backtest)
    settings.paths.agribalyse_csv.write_bytes(b"full export")
    settings.paths.agribalyse_csv_ecoinvent_free_xz.write_bytes(lzma.compress(b"free"))
    assert LinkAllCli(settings=settings).run(["--csv", "ecoinvent-free"]) == 0
    assert _Parser.parsed == ["AGB32_ecoinvent_free.CSV"]
    return replayed


def test_set_parameter_fast_follows_the_last_link(settings, scenario):
    rc = SetParameterCli(settings=settings).run(
        ["SP_RATIO", "0.6", "--product", "P_FREE", "--fast"]
    )

    assert rc == 0
    assert _Parser.parsed == ["AGB32_ecoinvent_free.CSV"]  # nothing parsed again
    assert [sp.data[0]["code"] for sp in scenario] == ["P_FREE"]  # the link was replayed


def test_set_parameter_full_path_relinks_the_same_csv(settings, scenario, monkeypatch):
    read: list[str | None] = []

    class _Relink:
        def __init__(self, settings, options=None):
            read.append(AgribalyseCsvSource(settings=settings).variant)

        def run(self):
            return None

    monkeypatch.setattr("cli.set_parameter.LinkAllPipeline", _Relink)

    rc = SetParameterCli(settings=settings).run(["SP_RATIO", "0.6", "--product", "P_FREE"])

    assert rc == 0
    assert read == ["ecoinvent-free"]


def test_a_command_without_csv_reads_the_last_links_parse(settings, scenario):
    """``dds-list-parameters``, ``dds-build-parameters`` and the randonneur
    export load the CSV exactly like this."""
    sp = SimaProImporter(settings).load()

    assert sp.data[0]["code"] == "P_FREE"
    assert _Parser.parsed == ["AGB32_ecoinvent_free.CSV"]


def test_a_plain_link_goes_back_to_the_full_export(settings, scenario):
    assert LinkAllCli(settings=settings).run([]) == 0

    assert SimaProImporter(settings).load().data[0]["code"] == "P_FULL"
    assert _Parser.parsed == ["AGB32_ecoinvent_free.CSV", "AGB32_final.CSV"]


def test_reset_forgets_the_request(settings, scenario):
    assert ResetCli(settings=settings).run([]) == 0

    assert not settings.paths.agribalyse_csv_request.exists()
    assert AgribalyseCsvSource(settings=settings).variant == "full"
