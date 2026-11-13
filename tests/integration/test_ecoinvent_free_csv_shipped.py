"""Audit of the committed ecoinvent-free CSV (spec acceptance A1).

Runs on the real shipped archive: no block carries the ``Copied from
Ecoinvent`` marker, no process of the manifest or of the deletion list is
present, and the archive is the one the manifest describes.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import re

import pytest

from config import Paths
from transforms import EcoinventStripper

PATHS = Paths()

pytestmark = pytest.mark.skipif(
    not PATHS.agribalyse_csv_ecoinvent_free_xz.exists(), reason="stripped archive not present"
)


@pytest.fixture(scope="module")
def stripped() -> bytes:
    return lzma.decompress(PATHS.agribalyse_csv_ecoinvent_free_xz.read_bytes())


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(PATHS.stripped_processes_json.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def identifiers(stripped: bytes) -> set[str]:
    found = re.findall(rb"^Process identifier\r?\n([^\r\n]*)", stripped, flags=re.MULTILINE)
    return {i.decode("latin-1").strip() for i in found}


class TestShippedArchive:
    def test_matches_the_manifest(self, stripped, manifest):
        assert manifest["stripped_csv"] == {
            "bytes": len(stripped),
            "sha256": hashlib.sha256(stripped).hexdigest(),
        }
        assert manifest["processes"]["kept"] == len(
            re.findall(rb"^Process\r?\n", stripped, flags=re.MULTILINE)
        )

    def test_no_marked_block_is_left(self, stripped):
        assert EcoinventStripper().leftovers(stripped) == ()

    def test_no_product_line_carries_the_marker(self, stripped):
        """Independent of the stripper's own block logic: every line that
        directly follows a ``Products`` / ``Waste treatment`` header."""
        heads = re.findall(
            rb"^(?:Products|Waste treatment)\r?\n([^\r\n;]*)", stripped, flags=re.MULTILINE
        )
        assert heads
        assert [h for h in heads if b"copied from ecoinvent" in h.lower()] == []

    def test_no_removed_process_is_present(self, identifiers, manifest):
        removed = {row["identifier"] for row in manifest["removed"]}
        assert len(removed) == manifest["processes"]["removed"]
        assert removed & identifiers == set()

    def test_no_process_of_the_deletion_list_is_present(self, identifiers):
        payload = json.loads(PATHS.delete_aggregated_processes_json.read_text(encoding="utf-8"))
        listed = {row["source"]["identifier"] for row in payload["delete"]}
        assert listed
        assert listed & identifiers == set()

    def test_fits_the_file_size_guard(self):
        assert PATHS.agribalyse_csv_ecoinvent_free_xz.stat().st_size < 11_000_000
