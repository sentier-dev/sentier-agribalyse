"""``AgribalyseCsvSource`` — full export when present, ecoinvent-free otherwise."""

from __future__ import annotations

import lzma
import os

import pytest

from transforms import AgribalyseCsvMissingError, AgribalyseCsvSource

PAYLOAD = b"{SimaPro 9.6.0.1}\r\nProcess\r\n\r\nEnd\r\n"


def _write_xz(settings, payload: bytes = PAYLOAD) -> None:
    settings.paths.agribalyse_csv_ecoinvent_free_xz.write_bytes(lzma.compress(payload))


class TestVariant:
    def test_full_csv_wins_when_present(self, settings):
        settings.paths.agribalyse_csv.write_bytes(b"full")
        _write_xz(settings)

        source = AgribalyseCsvSource(settings=settings)

        assert source.variant == "full"
        assert source.path() == settings.paths.agribalyse_csv

    def test_falls_back_to_the_ecoinvent_free_file(self, settings):
        _write_xz(settings)

        source = AgribalyseCsvSource(settings=settings)

        assert source.variant == "ecoinvent-free"
        assert source.path() == settings.paths.agribalyse_csv_ecoinvent_free_cache
        assert source.path().read_bytes() == PAYLOAD

    def test_neither_file_raises_with_both_names(self, settings):
        source = AgribalyseCsvSource(settings=settings)

        assert source.variant is None
        with pytest.raises(AgribalyseCsvMissingError, match=r"AGB32_ecoinvent_free\.CSV\.xz"):
            source.path()


class TestDecompression:
    def test_reuses_the_decompressed_copy(self, settings):
        _write_xz(settings)
        source = AgribalyseCsvSource(settings=settings)
        first = source.path().stat().st_mtime_ns

        assert source.path().stat().st_mtime_ns == first

    def test_refreshes_when_the_xz_is_newer(self, settings):
        _write_xz(settings)
        source = AgribalyseCsvSource(settings=settings)
        cached = source.path()
        _write_xz(settings, PAYLOAD + b"more\r\n")
        newer = cached.stat().st_mtime + 10
        os.utime(settings.paths.agribalyse_csv_ecoinvent_free_xz, (newer, newer))

        assert source.path().read_bytes() == PAYLOAD + b"more\r\n"

    def test_leaves_no_partial_file(self, settings):
        _write_xz(settings)

        AgribalyseCsvSource(settings=settings).path()

        assert [p.name for p in settings.paths.cache.iterdir()] == ["AGB32_ecoinvent_free.CSV"]


class TestDescribe:
    def test_describe_names_variant_and_file(self, settings):
        _write_xz(settings)

        assert AgribalyseCsvSource(settings=settings).describe() == {
            "variant": "ecoinvent-free",
            "file": "AGB32_ecoinvent_free.CSV.xz",
        }
