"""Unit tests for the deterministic parts of ``reporting.product_reasons``.

The LLM call itself is stubbed; these tests cover the scanner (CSV → outlier
products), the decomp-evidence extraction, the prompt assembly + JSON parsing,
and the generator's resume/skip behaviour. No network or CLI calls happen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reporting.product_reasons import (
    DecompEvidence,
    FlowLine,
    ProductOutlierScanner,
    ProductReasonGenerator,
    ProductReasonPrompt,
)

_CSV = (
    "code,name,mapped_to,type,resolution,climate,ht_c,water\n"
    "100,Apple,x,product,mapped,2.0,41.5,-97.0\n"  # ht_c + water outliers
    "200,Beef,x,product,mapped,5.0,10.0,15.0\n"  # no outliers
    "300,Carp,x,product,unmapped,99.0,99.0,99.0\n"  # filtered: not mapped
)


def test_scanner_groups_only_outlier_cells_of_mapped_rows() -> None:
    rows = ProductOutlierScanner.parse_csv(_CSV)
    products = ProductOutlierScanner(rows=rows, threshold_pct=30.0).scan()

    assert [p.code for p in products] == ["100"]  # Beef has none, Carp unmapped
    apple = products[0]
    assert {i.short for i in apple.impacts} == {"ht_c", "water"}
    water = next(i for i in apple.impacts if i.short == "water")
    assert water.direction == "under"
    assert next(i for i in apple.impacts if i.short == "ht_c").label == (
        "human toxicity: carcinogenic"
    )


def test_scanner_threshold_is_inclusive_and_signed() -> None:
    csv = "code,name,resolution,climate\n1,A,mapped,30.0\n2,B,mapped,-30.0\n3,C,mapped,29.9\n"
    products = ProductOutlierScanner(
        rows=ProductOutlierScanner.parse_csv(csv), threshold_pct=30.0
    ).scan()
    assert sorted(p.code for p in products) == ["1", "2"]  # 29.9 excluded


def test_flowline_cf_delta_and_render() -> None:
    line = FlowLine(
        flow_name="Nitrate",
        compartment="water",
        sub_compartment="ground-",
        cf=2.0,
        sp_cf=1.0,
        share_pct=62.0,
        provenance="cas",
        contribution_sign="positive",
    )
    assert line.cf_delta_pct == pytest.approx(100.0)
    rendered = line.render()
    assert "Nitrate" in rendered and "62% of" in rendered and "EF CF +100% vs SimaPro" in rendered

    no_sp = FlowLine("X", "air", "", cf=1.0, sp_cf=None, share_pct=5.0, provenance="—",
                     contribution_sign="negative")
    assert no_sp.cf_delta_pct is None
    assert "no SimaPro CF" in no_sp.render()


def test_decomp_evidence_filters_by_share_and_top_n(tmp_path: Path) -> None:
    decomp = {
        "code": "100",
        "methods": {
            "ht_c": {
                "flows": [
                    {"flow_name": "A", "compartment": "air", "share": 0.5, "cf": 2, "sp_cf": 1,
                     "contribution": 3.0, "sp_match_provenance": "exact_name"},
                    {"flow_name": "B", "compartment": "air", "share": 0.005, "cf": 1, "sp_cf": 1,
                     "contribution": 0.1},  # below 1% share → dropped
                ]
            }
        },
    }
    (tmp_path / "100.json").write_text(json.dumps(decomp))
    ev = DecompEvidence(decomp_dir=tmp_path, top_n=6, min_share_pct=1.0)
    loaded = ev.for_product("100")
    flows = ev.flows(loaded, "ht_c")
    assert [f.flow_name for f in flows] == ["A"]
    assert ev.for_product("999") is None  # missing file


def test_prompt_includes_evidence_and_parses_json(tmp_path: Path) -> None:
    rows = ProductOutlierScanner.parse_csv(_CSV)
    product = ProductOutlierScanner(rows=rows, threshold_pct=30.0).scan()[0]
    (tmp_path / "100.json").write_text(
        json.dumps(
            {
                "methods": {
                    "ht_c": {"flows": [
                        {"flow_name": "Chromium VI", "compartment": "water", "share": 0.8,
                         "cf": 5, "sp_cf": 4, "contribution": 9.0,
                         "sp_match_provenance": "synonym"}
                    ]},
                    "water": {"flows": []},
                }
            }
        )
    )
    prompt = ProductReasonPrompt(
        impact_notes={"ht_c": {"short": "EF v3.1 CF bias."}},
        evidence=DecompEvidence(decomp_dir=tmp_path),
    )
    decomp = prompt.evidence.for_product("100")
    user = prompt.user(product, decomp)
    assert "Chromium VI" in user
    assert "+42% vs ADEME (over-estimates)" in user  # 41.5 rounds half-to-even
    assert "EF v3.1 CF bias." in user

    reply = 'noise before {"ht_c": "Driven by chromium VI.", "water": "  ", "bogus": "x"} after'
    parsed = prompt.parse(reply, wanted=["ht_c", "water"])
    assert parsed == {"ht_c": "Driven by chromium VI."}  # blank + unwanted dropped


class _StubClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def ask(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def test_generator_writes_keyed_json_and_resumes(tmp_path: Path) -> None:
    rows = ProductOutlierScanner.parse_csv(_CSV)
    products = ProductOutlierScanner(rows=rows, threshold_pct=30.0).scan()
    out = tmp_path / "product_reasons.json"
    prompt = ProductReasonPrompt(impact_notes={}, evidence=DecompEvidence(decomp_dir=tmp_path))

    client = _StubClient('{"ht_c": "note A", "water": "note B"}')
    gen = ProductReasonGenerator(client=client, prompt=prompt, out_path=out, max_workers=1)
    result = gen.run(products)

    assert result["100"] == {"ht_c": "note A", "water": "note B"}
    on_disk = json.loads(out.read_text())
    assert on_disk["100"]["ht_c"] == "note A"
    assert "_about" in on_disk

    # Resume: a second run with the same output skips the already-covered product.
    client2 = _StubClient('{"ht_c": "SHOULD NOT APPEAR"}')
    gen2 = ProductReasonGenerator(client=client2, prompt=prompt, out_path=out, max_workers=1)
    gen2.run(products)
    assert client2.calls == 0
    assert json.loads(out.read_text())["100"]["ht_c"] == "note A"
