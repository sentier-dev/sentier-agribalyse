"""``EcoinventStripper`` — text-level removal of embedded ecoinvent copies."""

from __future__ import annotations

import hashlib
import json
import lzma

import pytest

from transforms import EcoinventLeftoverError, EcoinventStripper

HEADER = "{SimaPro 9.6.0.1}\r\n{processes}\r\n{Project: mini}\r\n\r\n"


def _process(identifier: str, name: str, type_: str = "Unit process", section="Products") -> str:
    return (
        "Process\r\n\r\n"
        "Category type\r\nmaterial\r\n\r\n"
        f"Process identifier\r\n{identifier}\r\n\r\n"
        f"Type\r\n{type_}\r\n\r\n"
        "Process name\r\n\r\n\r\n"
        "Comment\r\nCaf\xe9 cr\xe8me\r\n\r\n"
        f"{section}\r\n{name};kg;1;100;not defined;Cat\\Sub;\r\n\r\n"
        "Emissions to air\r\nCarbon dioxide, fossil;;kg;1,5;Undefined;0;0;0;\r\n\r\n"
        "End\r\n\r\n"
    )


TRAILER = (
    "System description\r\n\r\nName\r\nmini\r\n\r\nEnd\r\n\r\n"
    "Units\r\nkg;Mass;1;kg\r\n\r\nEnd\r\n\r\n"
)

KEPT_A = _process("AGB0001", "Wheat flour {FR}| production | Cut-off, U")
COPY_S = _process(
    "AGB0002", "Steel {RER}| production | Cut-off, S - Copied from Ecoinvent U", type_="System"
)
ADAPTED = _process(
    "AGB0003", "Banana {CO}| banana production | Cut-off, U - Adapted from Ecoinvent U"
)
COPY_U = _process("AGB0004", "Lorry {GLO}| market | Cut-off, U - Copied from Ecoinvent U")
COPY_WT = _process(
    "AGB0005",
    "Waste {CH}| treatment | Cut-off, S - Copied from ecoinvent U",
    type_="System",
    section="Waste treatment",
)
KEPT_B = _process("AGB0006", "Bread {FR}| baking | Cut-off, U")


def _full() -> bytes:
    text = HEADER + KEPT_A + COPY_S + ADAPTED + COPY_U + COPY_WT + KEPT_B + TRAILER
    return text.encode("latin-1")


class TestStrip:
    def test_removes_only_marked_blocks_and_keeps_the_rest_byte_identical(self):
        result = EcoinventStripper().strip(_full())

        expected = (HEADER + KEPT_A + ADAPTED + KEPT_B + TRAILER).encode("latin-1")
        assert result.stripped == expected

    def test_removed_rows_carry_identifier_type_and_block_hash(self):
        result = EcoinventStripper().strip(_full())

        assert [r.identifier for r in result.removed] == ["AGB0002", "AGB0004", "AGB0005"]
        assert [r.process_type for r in result.removed] == ["System", "Unit process", "System"]
        block = COPY_S.encode("latin-1").rstrip(b"\r\n") + b"\r\n"
        assert result.removed[0].sha256 == hashlib.sha256(block).hexdigest()

    def test_counts(self):
        result = EcoinventStripper().strip(_full())

        assert result.processes_total == 6
        assert result.processes_kept == 3

    def test_marker_match_ignores_case(self):
        result = EcoinventStripper().strip(_full())

        assert "AGB0005" in {r.identifier for r in result.removed}

    def test_lf_only_input_is_supported(self):
        raw = _full().replace(b"\r\n", b"\n")

        result = EcoinventStripper().strip(raw)

        assert len(result.removed) == 3
        assert b"Copied from" not in result.stripped

    def test_nothing_to_strip_returns_input_unchanged(self):
        raw = (HEADER + KEPT_A + TRAILER).encode("latin-1")

        result = EcoinventStripper().strip(raw)

        assert result.stripped == raw
        assert result.removed == ()


class TestAudit:
    def test_clean_file_has_no_leftovers(self):
        stripper = EcoinventStripper()

        assert stripper.leftovers(stripper.strip(_full()).stripped) == ()

    def test_full_file_reports_marked_identifiers(self):
        assert EcoinventStripper().leftovers(_full()) == ("AGB0002", "AGB0004", "AGB0005")

    def test_required_identifiers_must_all_be_removed(self):
        stripper = EcoinventStripper(required_identifiers=frozenset({"AGB0002", "AGB0006"}))

        with pytest.raises(EcoinventLeftoverError, match="AGB0006"):
            stripper.strip(_full())


class TestWrite:
    def test_writes_xz_and_manifest(self, tmp_path):
        full = tmp_path / "full.CSV"
        full.write_bytes(_full())
        out = tmp_path / "out" / "free.CSV.xz"
        manifest_path = tmp_path / "out" / "manifest.json"

        result = EcoinventStripper().write(
            full_csv=full, stripped_xz=out, manifest_json=manifest_path
        )

        assert lzma.decompress(out.read_bytes()) == result.stripped
        manifest = json.loads(manifest_path.read_text())
        assert manifest["marker"] == "Copied from Ecoinvent"
        assert manifest["full_csv"]["sha256"] == hashlib.sha256(_full()).hexdigest()
        assert manifest["stripped_csv"]["sha256"] == hashlib.sha256(result.stripped).hexdigest()
        assert manifest["stripped_csv"]["bytes"] == len(result.stripped)
        assert manifest["processes"] == {"total": 6, "kept": 3, "removed": 3}
        assert manifest["removed"][0] == {
            "identifier": "AGB0002",
            "type": "System",
            "sha256": result.removed[0].sha256,
        }

    def test_manifest_carries_no_process_names(self, tmp_path):
        full = tmp_path / "full.CSV"
        full.write_bytes(_full())
        manifest_path = tmp_path / "manifest.json"

        EcoinventStripper().write(
            full_csv=full, stripped_xz=tmp_path / "free.CSV.xz", manifest_json=manifest_path
        )

        assert "Steel" not in manifest_path.read_text()

    def test_write_is_byte_stable(self, tmp_path):
        full = tmp_path / "full.CSV"
        full.write_bytes(_full())
        a, b = tmp_path / "a.xz", tmp_path / "b.xz"

        EcoinventStripper().write(full_csv=full, stripped_xz=a, manifest_json=tmp_path / "a.json")
        EcoinventStripper().write(full_csv=full, stripped_xz=b, manifest_json=tmp_path / "b.json")

        assert a.read_bytes() == b.read_bytes()
        assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()
