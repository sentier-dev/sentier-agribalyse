"""``SimaProImporter`` — the parse cache is reused only for the CSV it was parsed from."""

from __future__ import annotations

import lzma
import os
import pickle
from typing import ClassVar

import pytest

from transforms.importer import SimaProImporter
from transforms.sp_csv_parser import ParsedSimaProCsv


class _CountingParser:
    """Stands in for ``SimaProCsvParser``: records which file each parse read."""

    parsed: ClassVar[list[str]] = []

    def __init__(self, csv_path, database_name):
        self.csv_path = csv_path

    def parse(self):
        _CountingParser.parsed.append(self.csv_path.name)
        return ParsedSimaProCsv(db_name="agb", data=[{"name": self.csv_path.name}], parameters=[])


@pytest.fixture
def parsed(monkeypatch):
    _CountingParser.parsed = []
    monkeypatch.setattr("transforms.importer.SimaProCsvParser", _CountingParser)
    return _CountingParser.parsed


def _write_both(settings) -> None:
    settings.paths.agribalyse_csv.write_bytes(b"full export")
    settings.paths.agribalyse_csv_ecoinvent_free_xz.write_bytes(lzma.compress(b"free"))


def test_the_cached_parse_is_reused_for_the_same_csv(settings, parsed):
    _write_both(settings)

    SimaProImporter(settings=settings).load()
    sp = SimaProImporter(settings=settings).load()

    assert parsed == ["AGB32_final.CSV"]
    assert sp.data == [{"name": "AGB32_final.CSV"}]


def test_asking_for_the_ecoinvent_free_csv_reparses(settings, parsed):
    """Without the stamp, the cache would replay the full export's parse."""
    _write_both(settings)

    SimaProImporter(settings=settings).load()
    sp = SimaProImporter(settings=settings.with_agribalyse_csv("ecoinvent-free")).load()

    assert parsed == ["AGB32_final.CSV", "AGB32_ecoinvent_free.CSV"]
    assert sp.data == [{"name": "AGB32_ecoinvent_free.CSV"}]


def test_a_changed_csv_is_reparsed(settings, parsed):
    _write_both(settings)
    SimaProImporter(settings=settings).load()

    settings.paths.agribalyse_csv.write_bytes(b"a newer full export")
    SimaProImporter(settings=settings).load()

    assert parsed == ["AGB32_final.CSV", "AGB32_final.CSV"]


def test_a_touched_but_unchanged_csv_is_not_reparsed(settings, parsed):
    """A checkout or a copy moves the modification time of an unchanged file."""
    _write_both(settings)
    SimaProImporter(settings=settings).load()

    later = settings.paths.agribalyse_csv.stat().st_mtime + 60
    os.utime(settings.paths.agribalyse_csv, (later, later))
    SimaProImporter(settings=settings).load()

    assert parsed == ["AGB32_final.CSV"]


def test_a_cache_that_records_no_csv_is_reparsed(settings, parsed):
    _write_both(settings)
    cache = settings.paths.importer_cache_pkl
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(ParsedSimaProCsv(db_name="agb", data=[], parameters=[])))

    SimaProImporter(settings=settings).load()

    assert parsed == ["AGB32_final.CSV"]
