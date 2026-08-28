"""Unit tests for ``ProcessParameterExtractor``.

The load-bearing case is the position-zip: ~700 AGB 3.2 processes carry no
``Process identifier``, so ``lci_to_brightway`` synthesises uuid4 codes
that exist only on the output dicts. Parameters must be keyed by THOSE
codes or the processes become invisible to overrides (the 2026-08-18
dangling-reference finding: 16,745 formula rows referenced parameters the
extractor had skipped).
"""

from __future__ import annotations

import pytest

from transforms.parameter_extraction import ProcessParameterExtractor


class _Sub:
    def __init__(self, parsed):
        self.parsed = parsed


class Process:
    def __init__(self, identifier, params=None):
        self.parsed = {"metadata": {"Process identifier": identifier}}
        self.blocks = {}
        if params is not None:
            self.blocks["Input parameters"] = _Sub(params)


class _FakeSpcsv:
    def __init__(self, blocks):
        self.blocks = blocks


def _param(name="SP_X", original="x", amount=1.0):
    return {"name": name, "original_name": original, "amount": amount, "comment": ""}


class TestPositionZip:
    def test_uuid_coded_dataset_inherits_code_by_position(self):
        blocks = [
            Process("REAL1", [_param()]),
            Process(None, [_param(name="SP_Y", original="y")]),  # no identifier
            Process("REAL3", [_param(name="SP_Z", original="z")]),
        ]
        bw_processes = [
            {"code": "REAL1"},
            {"code": "deadbeefdeadbeefdeadbeefdeadbeef"},  # uuid4 synthesised
            {"code": "REAL3"},
        ]
        rows = ProcessParameterExtractor().extract(_FakeSpcsv(blocks), bw_processes=bw_processes)
        by_name = {r["name"]: r["process_code"] for r in rows}
        assert by_name["SP_Y"] == "deadbeefdeadbeefdeadbeefdeadbeef"
        assert by_name["SP_X"] == "REAL1"
        assert by_name["SP_Z"] == "REAL3"

    def test_order_drift_raises(self):
        blocks = [Process("REAL1", [_param()]), Process("REAL2", [_param()])]
        bw_processes = [{"code": "REAL2"}, {"code": "REAL1"}]  # swapped
        with pytest.raises(ValueError, match="order drift"):
            ProcessParameterExtractor().extract(_FakeSpcsv(blocks), bw_processes=bw_processes)

    def test_count_mismatch_raises(self):
        blocks = [Process("REAL1", [_param()])]
        with pytest.raises(ValueError, match="count mismatch"):
            ProcessParameterExtractor().extract(_FakeSpcsv(blocks), bw_processes=[])

    def test_identifierless_block_paired_with_real_code_raises(self):
        """An identifier-less block must pair with a uuid4-hex synthesised
        code; a real identifier there means alignment slipped inside an
        anchor gap (2026-08-18 adversarial F8)."""
        blocks = [Process(None, [_param()])]
        with pytest.raises(ValueError, match="order drift"):
            ProcessParameterExtractor().extract(
                _FakeSpcsv(blocks), bw_processes=[{"code": "AGRIBALU000000003100001"}]
            )

    def test_whitespace_identifier_matches_unstripped_dataset_code(self):
        blocks = [Process("REAL1 ", [_param()])]
        rows = ProcessParameterExtractor().extract(
            _FakeSpcsv(blocks), bw_processes=[{"code": "REAL1 "}]
        )
        # Keyed by the dataset's exact code — that's what re-evaluation joins on.
        assert rows[0]["process_code"] == "REAL1 "


class TestChildMirroring:
    def test_children_inherit_parent_parameters(self):
        blocks = [Process("PARENT1", [_param()])]
        bw_processes = [
            {"code": "PARENT1", "type": "multifunctional"},
            {
                "code": "child1hex",
                "type": "readonly_process",
                "mf_parent_key": ("agb", "PARENT1"),
            },
            {
                "code": "child2hex",
                "type": "readonly_process",
                "mf_parent_key": ("agb", "PARENT1"),
            },
        ]
        rows = ProcessParameterExtractor().extract(_FakeSpcsv(blocks), bw_processes=bw_processes)
        by_code = {r["process_code"] for r in rows}
        assert by_code == {"PARENT1", "child1hex", "child2hex"}
        assert len(rows) == 3  # one SP_X row mirrored onto each child

    def test_children_of_uuid_coded_parent_inherit_too(self):
        blocks = [Process(None, [_param()])]
        bw_processes = [
            {"code": "aaaa" * 8, "type": "multifunctional"},
            {
                "code": "bbbb" * 8,
                "type": "readonly_process",
                "mf_parent_key": ("agb", "aaaa" * 8),
            },
        ]
        rows = ProcessParameterExtractor().extract(_FakeSpcsv(blocks), bw_processes=bw_processes)
        assert {r["process_code"] for r in rows} == {"aaaa" * 8, "bbbb" * 8}


class TestLegacyFallback:
    def test_without_bw_processes_skips_identifier_less_blocks(self):
        blocks = [Process("REAL1", [_param()]), Process(None, [_param(name="SP_Y")])]
        rows = ProcessParameterExtractor().extract(_FakeSpcsv(blocks))
        assert {r["process_code"] for r in rows} == {"REAL1"}
