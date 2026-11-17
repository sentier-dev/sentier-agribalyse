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

        assert sorted(p.name for p in settings.paths.cache.iterdir()) == [
            "AGB32_ecoinvent_free.CSV",
            "AGB32_ecoinvent_free.CSV.sha256",
        ]

    def test_refreshes_when_the_archive_changes_although_the_copy_is_newer(self, settings):
        """A copy that keeps times (``cp -p``, ``rsync -a``) can put an older
        modification time on a newer archive."""
        _write_xz(settings)
        source = AgribalyseCsvSource(settings=settings)
        cached = source.path()
        _write_xz(settings, PAYLOAD + b"more\r\n")
        older = cached.stat().st_mtime - 10
        os.utime(settings.paths.agribalyse_csv_ecoinvent_free_xz, (older, older))

        assert source.path().read_bytes() == PAYLOAD + b"more\r\n"


class TestRequest:
    """``Settings.agribalyse_csv`` (``dds-link-all --csv``) picks the variant."""

    def test_ecoinvent_free_is_read_although_the_full_export_is_there(self, settings):
        settings.paths.agribalyse_csv.write_bytes(b"full")
        _write_xz(settings)

        source = AgribalyseCsvSource(settings=settings.with_agribalyse_csv("ecoinvent-free"))

        assert source.variant == "ecoinvent-free"
        assert source.path().read_bytes() == PAYLOAD

    def test_full_asked_for_but_absent_names_the_file(self, settings):
        _write_xz(settings)

        source = AgribalyseCsvSource(settings=settings.with_agribalyse_csv("full"))

        assert source.variant is None
        with pytest.raises(AgribalyseCsvMissingError, match=r"AGB32_final\.CSV"):
            source.path()

    def test_ecoinvent_free_asked_for_but_absent_names_the_file(self, settings):
        settings.paths.agribalyse_csv.write_bytes(b"full")

        source = AgribalyseCsvSource(settings=settings.with_agribalyse_csv("ecoinvent-free"))

        assert source.variant is None
        with pytest.raises(AgribalyseCsvMissingError, match=r"AGB32_ecoinvent_free\.CSV\.xz"):
            source.path()

    def test_an_unknown_request_is_refused(self, settings):
        with pytest.raises(ValueError, match="auto, full, ecoinvent-free"):
            settings.with_agribalyse_csv("free")


class TestLastLinkRequest:
    """A command that asks for no CSV (``Settings.agribalyse_csv is None``: every
    command but ``dds-link-all``) reads what the last link asked for."""

    @staticmethod
    def _both(settings) -> None:
        settings.paths.agribalyse_csv.write_bytes(b"full")
        _write_xz(settings)

    def test_without_a_record_the_full_export_wins(self, settings):
        self._both(settings)

        source = AgribalyseCsvSource(settings=settings)

        assert (source.requested, source.variant, source.follows_last_link) == (
            "auto",
            "full",
            False,
        )

    def test_the_last_links_request_is_followed(self, settings):
        self._both(settings)
        AgribalyseCsvSource(
            settings=settings.with_agribalyse_csv("ecoinvent-free")
        ).record_request()

        source = AgribalyseCsvSource(settings=settings)

        assert (source.requested, source.variant, source.follows_last_link) == (
            "ecoinvent-free",
            "ecoinvent-free",
            True,
        )
        assert source.describe()["follows_last_link"] is True

    def test_a_request_of_its_own_ignores_the_record(self, settings):
        """``dds-link-all`` always asks (``auto`` by default): without ``--csv``
        it reads the full export again, whatever the last link asked for."""
        self._both(settings)
        AgribalyseCsvSource(
            settings=settings.with_agribalyse_csv("ecoinvent-free")
        ).record_request()

        source = AgribalyseCsvSource(settings=settings.with_agribalyse_csv("auto"))

        assert source.variant == "full"
        source.record_request()
        assert AgribalyseCsvSource(settings=settings).variant == "full"

    def test_following_a_request_whose_file_is_gone_says_whose_request_it_was(self, settings):
        _write_xz(settings)
        AgribalyseCsvSource(settings=settings.with_agribalyse_csv("full")).record_request()

        with pytest.raises(AgribalyseCsvMissingError, match=r"last dds-link-all.*--csv full"):
            AgribalyseCsvSource(settings=settings).path()

    def test_a_record_that_names_no_csv_is_refused(self, settings):
        _write_xz(settings)
        settings.paths.agribalyse_csv_request.write_text("free\n")

        with pytest.raises(
            ValueError, match=r"agribalyse_csv_request\.txt names no Agribalyse CSV"
        ):
            AgribalyseCsvSource(settings=settings).path()


class TestStamp:
    """What the caches record: which file fed the run, and its state."""

    def test_the_stamp_names_the_file_the_run_reads(self, settings):
        settings.paths.agribalyse_csv.write_bytes(b"full")
        _write_xz(settings)

        auto = AgribalyseCsvSource(settings=settings).stamp()
        free = AgribalyseCsvSource(settings=settings.with_agribalyse_csv("ecoinvent-free")).stamp()

        assert auto[:2] == ("full", "AGB32_final.CSV")
        assert free[:2] == ("ecoinvent-free", "AGB32_ecoinvent_free.CSV.xz")

    def test_the_stamp_moves_when_the_file_changes(self, settings):
        settings.paths.agribalyse_csv.write_bytes(b"full")
        before = AgribalyseCsvSource(settings=settings).stamp()

        settings.paths.agribalyse_csv.write_bytes(b"a newer full export")

        assert AgribalyseCsvSource(settings=settings).stamp() != before

    def test_the_stamp_moves_when_the_content_changes_at_the_same_size_and_time(self, settings):
        csv = settings.paths.agribalyse_csv
        when = (1_700_000_000_000_000_000, 1_700_000_000_000_000_000)
        csv.write_bytes(b"full export, v1")
        os.utime(csv, ns=when)
        before = AgribalyseCsvSource(settings=settings).stamp()

        csv.write_bytes(b"full export, v2")
        os.utime(csv, ns=when)

        assert AgribalyseCsvSource(settings=settings).stamp() != before

    def test_the_stamp_stays_when_an_unchanged_file_is_touched(self, settings):
        """A checkout or a copy moves the time, not the content: no re-parse."""
        csv = settings.paths.agribalyse_csv
        csv.write_bytes(b"full export")
        before = AgribalyseCsvSource(settings=settings).stamp()

        later = csv.stat().st_mtime + 60
        os.utime(csv, (later, later))

        assert AgribalyseCsvSource(settings=settings).stamp() == before

    def test_no_csv_no_stamp(self, settings):
        assert AgribalyseCsvSource(settings=settings).stamp() is None


class TestDescribe:
    def test_describe_names_variant_and_file(self, settings):
        _write_xz(settings)

        assert AgribalyseCsvSource(settings=settings).describe() == {
            "requested": "auto",
            "follows_last_link": False,
            "variant": "ecoinvent-free",
            "file": "AGB32_ecoinvent_free.CSV.xz",
        }

    def test_describe_says_when_a_variant_was_asked_for(self, settings):
        settings.paths.agribalyse_csv.write_bytes(b"full")
        _write_xz(settings)

        source = AgribalyseCsvSource(settings=settings.with_agribalyse_csv("ecoinvent-free"))

        assert source.describe() == {
            "requested": "ecoinvent-free",
            "follows_last_link": False,
            "variant": "ecoinvent-free",
            "file": "AGB32_ecoinvent_free.CSV.xz",
        }
