"""``HarmonisedFlowsFetcher`` — pinned download of the Brightway Flows list."""

from __future__ import annotations

import hashlib
import io

import pytest

from registry import HarmonisedFlowsFetcher, HarmonisedFlowsMismatchError, HarmonisedFlowsRelease

PAYLOAD = b"flow list bytes"


class _Opener:
    def __init__(self, payload: bytes = PAYLOAD):
        self.payload = payload
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return io.BytesIO(self.payload)


def _release(payload: bytes = PAYLOAD) -> HarmonisedFlowsRelease:
    return HarmonisedFlowsRelease(
        version="9.9.9",
        doi="10.0000/test",
        url="https://example.org/flows.json.gz",
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _fetcher(settings, opener, release=None) -> HarmonisedFlowsFetcher:
    return HarmonisedFlowsFetcher(settings=settings, release=release or _release(), opener=opener)


class TestFetch:
    def test_downloads_when_missing(self, settings):
        opener = _Opener()

        outcome = _fetcher(settings, opener).fetch()

        assert outcome == "downloaded"
        assert settings.paths.harmonised_flows_gz.read_bytes() == PAYLOAD
        assert opener.requests[0].full_url == "https://example.org/flows.json.gz"
        assert opener.requests[0].get_header("User-agent") == "sentier-agribalyse"

    def test_is_a_no_op_when_the_pinned_file_is_present(self, settings):
        settings.paths.harmonised_flows_gz.write_bytes(PAYLOAD)
        opener = _Opener()

        assert _fetcher(settings, opener).fetch() == "present"
        assert opener.requests == []

    def test_refuses_to_replace_another_file_without_force(self, settings):
        settings.paths.harmonised_flows_gz.write_bytes(b"older snapshot")
        opener = _Opener()

        with pytest.raises(HarmonisedFlowsMismatchError, match="--force"):
            _fetcher(settings, opener).fetch()
        assert settings.paths.harmonised_flows_gz.read_bytes() == b"older snapshot"
        assert opener.requests == []

    def test_force_replaces_another_file(self, settings):
        settings.paths.harmonised_flows_gz.write_bytes(b"older snapshot")

        outcome = _fetcher(settings, _Opener()).fetch(force=True)

        assert outcome == "downloaded"
        assert settings.paths.harmonised_flows_gz.read_bytes() == PAYLOAD

    def test_corrupt_download_writes_nothing(self, settings):
        opener = _Opener(payload=b"tampered")

        with pytest.raises(HarmonisedFlowsMismatchError, match="Nothing was written"):
            _fetcher(settings, opener, release=_release()).fetch()
        assert list(settings.paths.source.glob("harmonised-flows*")) == []

    def test_corrupt_download_keeps_the_existing_file_under_force(self, settings):
        settings.paths.harmonised_flows_gz.write_bytes(b"older snapshot")

        with pytest.raises(HarmonisedFlowsMismatchError):
            _fetcher(settings, _Opener(payload=b"tampered")).fetch(force=True)
        assert settings.paths.harmonised_flows_gz.read_bytes() == b"older snapshot"


class TestIsCurrent:
    def test_false_when_missing_true_after_fetch(self, settings):
        fetcher = _fetcher(settings, _Opener())

        assert fetcher.is_current() is False
        fetcher.fetch()
        assert fetcher.is_current() is True


class TestPinnedRelease:
    def test_default_release_is_zenodo_1_1_0(self):
        release = HarmonisedFlowsRelease()

        assert release.version == "1.1.0"
        assert release.url.startswith("https://zenodo.org/api/records/23554066/")
        assert len(release.sha256) == 64
