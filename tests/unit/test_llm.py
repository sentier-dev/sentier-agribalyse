"""Unit tests for the LLM-assisted mapping module.

No real CLI invocations or network calls — every test injects a stub client
with canned text responses, or exercises pure-Python helpers (candidate
ranking, JSON extraction, xlsx merging).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from unittest.mock import patch

import pandas as pd
import pytest

from domain import Bucket
from llm import (
    AgbUnlinkedFlow,
    AnthropicApiClient,
    CandidatePool,
    ClaudeCliClient,
    LlmMappingSuggester,
    LlmReviewedXlsxAppender,
    Suggestion,
    UnlinkedReader,
)
from matching.bio_catalog import BioFlowRef, BiosphereCatalog
from tests.fixtures.builders import make_settings

# ---------------------------------------------------------------------------
# Stubs.


@dataclass
class _StubClient:
    """LlmClient that returns canned strings and records every call."""

    canned: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.canned:
            raise AssertionError("StubClient ran out of canned responses")
        return self.canned.pop(0)


def _propose_json(idx: int, conf: str = "high", note: str = "ok") -> str:
    return json.dumps(
        {"decision": "propose", "candidate_index": idx, "confidence": conf, "rationale": note}
    )


def _reject_json(conf: str = "low", note: str = "no match") -> str:
    return json.dumps({"decision": "reject", "confidence": conf, "rationale": note})


def _make_catalog(flows: list[BioFlowRef]) -> BiosphereCatalog:
    return BiosphereCatalog(
        db_names=tuple(sorted({f.db for f in flows})),
        flows=tuple(flows),
    )


# ---------------------------------------------------------------------------
# CandidatePool.


class TestCandidatePool:
    def test_filters_by_bucket(self):
        catalog = _make_catalog(
            [
                BioFlowRef(
                    db="biosphere3", code="A", name="Acetylene", unit="kg", bucket=Bucket.AIR
                ),
                BioFlowRef(
                    db="biosphere3", code="B", name="Acetylene", unit="kg", bucket=Bucket.WATER
                ),
                BioFlowRef(db="biosphere3", code="C", name="Methane", unit="kg", bucket=Bucket.AIR),
            ]
        )
        pool = CandidatePool(catalog=catalog, max_candidates=10)
        assert [c.code for c in pool.candidates_for("Acetylene", Bucket.AIR)] == ["A"]
        assert [c.code for c in pool.candidates_for("Acetylene", Bucket.WATER)] == ["B"]

    def test_ranks_by_token_overlap(self):
        catalog = _make_catalog(
            [
                BioFlowRef(
                    db="bio3",
                    code="best",
                    name="carbon dioxide non fossil",
                    unit="kg",
                    bucket=Bucket.AIR,
                ),
                BioFlowRef(
                    db="bio3", code="med", name="carbon dioxide", unit="kg", bucket=Bucket.AIR
                ),
                BioFlowRef(db="bio3", code="worst", name="dioxide", unit="kg", bucket=Bucket.AIR),
                BioFlowRef(db="bio3", code="zero", name="methane", unit="kg", bucket=Bucket.AIR),
            ]
        )
        ranked = CandidatePool(catalog=catalog).candidates_for(
            "carbon dioxide, non-fossil", Bucket.AIR
        )
        assert [c.code for c in ranked] == ["best", "med", "worst"]

    def test_returns_empty_when_no_overlap(self):
        catalog = _make_catalog(
            [BioFlowRef(db="bio3", code="X", name="ammonia", unit="kg", bucket=Bucket.AIR)]
        )
        assert CandidatePool(catalog=catalog).candidates_for("foo bar", Bucket.AIR) == []

    def test_max_candidates_caps_results(self):
        flows = [
            BioFlowRef(
                db="bio3", code=f"c{i}", name=f"water variant {i}", unit="kg", bucket=Bucket.WATER
            )
            for i in range(50)
        ]
        out = CandidatePool(catalog=_make_catalog(flows), max_candidates=5).candidates_for(
            "water", Bucket.WATER
        )
        assert len(out) == 5

    def test_deterministic_tiebreak_on_equal_overlap(self):
        catalog = _make_catalog(
            [
                BioFlowRef(
                    db="bio3",
                    code="long",
                    name="zinc compound oxide variant",
                    unit="kg",
                    bucket=Bucket.WATER,
                ),
                BioFlowRef(db="bio3", code="short", name="zinc", unit="kg", bucket=Bucket.WATER),
                BioFlowRef(
                    db="bio3", code="med", name="zinc oxide", unit="kg", bucket=Bucket.WATER
                ),
            ]
        )
        out = CandidatePool(catalog=catalog).candidates_for("zinc", Bucket.WATER)
        assert [c.code for c in out] == ["short", "med", "long"]


# ---------------------------------------------------------------------------
# UnlinkedReader.


class TestUnlinkedReader:
    @pytest.fixture
    def settings(self, tmp_path):
        return make_settings(tmp_path)

    def test_skips_unmatchable_and_already_reviewed_but_keeps_unfired_mappings(
        self, settings, tmp_path
    ):
        path = settings.paths.unlinked / "biosphere_unlinked.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "source_name": "Forever chem",
                    "source_unit": "kg",
                    "source_top_cat": "air",
                    "source_sub_cat": "",
                },
                {
                    "source_name": "Already reviewed",
                    "source_unit": "kg",
                    "source_top_cat": "air",
                    "source_sub_cat": "",
                },
                {
                    "source_name": "Has dead mapping",
                    "source_unit": "kg",
                    "source_top_cat": "soil",
                    "source_sub_cat": "",
                },
                {
                    "source_name": "Truly missing",
                    "source_unit": "kg",
                    "source_top_cat": "soil",
                    "source_sub_cat": "agricultural",
                },
            ]
        ).to_excel(path, index=False)

        # Existing LLM-reviewed xlsx has "Already reviewed" — must be skipped.
        reviewed = tmp_path / "reviewed.xlsx"
        pd.DataFrame([{"source_name": "Already reviewed"}]).to_excel(reviewed, index=False)

        out = UnlinkedReader(settings=settings).read(
            registry_unmatchable=pd.DataFrame([{"source_name": "Forever chem"}]),
            already_reviewed_xlsx=reviewed,
        )
        names = [f.name for f in out]
        # "Has dead mapping" stays — its registry mapping didn't fire (e.g. wrong
        # compartment). The LLM should help find a target that does exist.
        assert names == ["Has dead mapping", "Truly missing"]

    def test_returns_empty_when_xlsx_missing(self, settings):
        out = UnlinkedReader(settings=settings).read(
            registry_unmatchable=pd.DataFrame(),
        )
        assert out == []


# ---------------------------------------------------------------------------
# LlmMappingSuggester — driven by the stub client.


class TestLlmMappingSuggester:
    SOURCE_FLOW = AgbUnlinkedFlow(
        name="Carbon dioxide, non-fossil", unit="kg", top_cat="air", sub_cat=""
    )

    @pytest.fixture
    def settings(self, tmp_path):
        return make_settings(tmp_path)

    @pytest.fixture
    def pool_with_targets(self):
        flows = [
            BioFlowRef(
                db="biosphere3",
                code="co2-nf",
                name="Carbon dioxide non-fossil",
                unit="kg",
                bucket=Bucket.AIR,
            ),
            BioFlowRef(
                db="biosphere3", code="co2", name="Carbon dioxide", unit="kg", bucket=Bucket.AIR
            ),
        ]
        return CandidatePool(catalog=_make_catalog(flows))

    def test_proposes_when_llm_picks_a_candidate(self, settings, pool_with_targets):
        client = _StubClient(canned=[_propose_json(0, "high", "exact non-fossil match")])
        out = LlmMappingSuggester(
            settings=settings, candidate_pool=pool_with_targets, client=client
        ).suggest([self.SOURCE_FLOW])
        assert out[0].decision == "propose"
        assert out[0].target_db == "biosphere3"
        assert out[0].target_code == "co2-nf"
        assert out[0].confidence == "high"
        assert "non-fossil" in out[0].rationale

    def test_rejects_when_llm_says_so(self, settings, pool_with_targets):
        client = _StubClient(canned=[_reject_json("low", "no defensible match")])
        out = LlmMappingSuggester(
            settings=settings, candidate_pool=pool_with_targets, client=client
        ).suggest([self.SOURCE_FLOW])
        assert out[0].decision == "reject"
        assert out[0].target_db == ""

    def test_no_candidates_skips_the_llm_entirely(self, settings):
        client = _StubClient()
        out = LlmMappingSuggester(
            settings=settings,
            candidate_pool=CandidatePool(catalog=_make_catalog([])),
            client=client,
        ).suggest([self.SOURCE_FLOW])
        assert out[0].decision == "no_candidates"
        assert client.calls == []

    def test_invalid_candidate_index_falls_back_to_reject(self, settings, pool_with_targets):
        client = _StubClient(canned=[_propose_json(99, "high", "oops")])
        out = LlmMappingSuggester(
            settings=settings, candidate_pool=pool_with_targets, client=client
        ).suggest([self.SOURCE_FLOW])
        assert out[0].decision == "reject"
        assert "invalid candidate_index" in out[0].rationale

    def test_unparseable_response_falls_back_to_reject(self, settings, pool_with_targets):
        client = _StubClient(canned=["I cannot answer that question."])
        out = LlmMappingSuggester(
            settings=settings, candidate_pool=pool_with_targets, client=client
        ).suggest([self.SOURCE_FLOW])
        assert out[0].decision == "reject"
        assert "JSON" in out[0].rationale

    def test_extracts_json_embedded_in_prose(self, settings, pool_with_targets):
        # Some CLI responses wrap the JSON in chatty prose. The parser must
        # find the first JSON object regardless.
        client = _StubClient(
            canned=[
                "Sure, here is my answer:\n```\n" + _propose_json(1, "medium", "fallback") + "\n```"
            ]
        )
        out = LlmMappingSuggester(
            settings=settings, candidate_pool=pool_with_targets, client=client
        ).suggest([self.SOURCE_FLOW])
        assert out[0].decision == "propose"
        assert out[0].target_code == "co2"

    def test_client_exception_is_caught_as_reject(self, settings, pool_with_targets):
        @dataclass
        class _Boom:
            def ask(self, system, user):
                raise RuntimeError("network down")

        out = LlmMappingSuggester(
            settings=settings, candidate_pool=pool_with_targets, client=_Boom()
        ).suggest([self.SOURCE_FLOW])
        assert out[0].decision == "reject"
        assert "network down" in out[0].rationale

    def test_user_message_contains_source_and_candidates(self, settings, pool_with_targets):
        client = _StubClient(canned=[_reject_json("medium", "n/a")])
        LlmMappingSuggester(
            settings=settings, candidate_pool=pool_with_targets, client=client
        ).suggest([self.SOURCE_FLOW])
        _, user = client.calls[0]
        assert "Carbon dioxide, non-fossil" in user
        assert "co2-nf" in user
        assert "Pick ONE candidate index" in user


# ---------------------------------------------------------------------------
# ClaudeCliClient — subprocess.run is patched so no real CLI runs.


class TestClaudeCliClient:
    def test_passes_combined_prompt_to_stdin_and_returns_stdout(self):
        with (
            patch("llm.client.subprocess.run") as mock_run,
            patch("llm.client.shutil.which", return_value="/usr/local/bin/claude"),
        ):
            mock_run.return_value = subprocess.CompletedProcess(
                args=["claude"], returncode=0, stdout="response text", stderr=""
            )
            out = ClaudeCliClient().ask("SYS", "USER")

            assert out == "response text"
            kwargs = mock_run.call_args.kwargs
            assert kwargs["input"] == "SYS\n\nUSER"
            assert kwargs["text"] is True
            assert kwargs["capture_output"] is True
            assert mock_run.call_args.args[0][0] == "/usr/local/bin/claude"
            assert "--print" in mock_run.call_args.args[0]

    def test_nonzero_exit_raises_runtime_error(self):
        with (
            patch("llm.client.subprocess.run") as mock_run,
            patch("llm.client.shutil.which", return_value="/usr/local/bin/claude"),
        ):
            mock_run.return_value = subprocess.CompletedProcess(
                args=["claude"], returncode=2, stdout="", stderr="auth missing"
            )
            with pytest.raises(RuntimeError, match="exit 2"):
                ClaudeCliClient().ask("S", "U")


# ---------------------------------------------------------------------------
# AnthropicApiClient — Anthropic SDK is mocked.


class TestAnthropicApiClient:
    def test_calls_messages_create_and_concatenates_text(self):
        @dataclass
        class _Block:
            text: str
            type: str = "text"

        @dataclass
        class _Resp:
            content: list

        @dataclass
        class _Messages:
            calls: list = field(default_factory=list)

            def create(self, **kw):
                self.calls.append(kw)
                return _Resp(content=[_Block(text="hello "), _Block(text="world")])

        @dataclass
        class _Sdk:
            messages: _Messages = field(default_factory=_Messages)

        sdk = _Sdk()
        out = AnthropicApiClient(client=sdk, model="claude-sonnet-4-6").ask("S", "U")
        assert out == "hello world"
        assert sdk.messages.calls[0]["model"] == "claude-sonnet-4-6"
        assert sdk.messages.calls[0]["system"] == "S"
        assert sdk.messages.calls[0]["messages"] == [{"role": "user", "content": "U"}]


# ---------------------------------------------------------------------------
# LlmReviewedXlsxAppender — auto-accept, no sidecar file.


class TestLlmReviewedXlsxAppender:
    def _suggestions(self):
        return [
            Suggestion(
                source=AgbUnlinkedFlow(name="Acetylene", unit="kg", top_cat="air", sub_cat=""),
                decision="propose",
                target_db="biosphere3",
                target_code="ethyne",
                target_name="Ethyne",
                target_categories=("air",),
                target_unit="kg",
                confidence="high",
                rationale="synonym",
            ),
            Suggestion(
                source=AgbUnlinkedFlow(name="Mystery", unit="kg", top_cat="soil", sub_cat=""),
                decision="reject",
                confidence="low",
                rationale="no defensible match",
            ),
            Suggestion(
                source=AgbUnlinkedFlow(name="Exotic", unit="kg", top_cat="air", sub_cat=""),
                decision="no_candidates",
            ),
        ]

    def test_creates_file_with_only_proposed_rows_marked_accept(self, tmp_path):
        out = tmp_path / "agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx"
        stats = LlmReviewedXlsxAppender(output_path=out).append(self._suggestions())

        df = pd.read_excel(out)
        # Only the propose row should be written; reject + no_candidates are dropped.
        assert len(df) == 1
        assert df.iloc[0]["source_name"] == "Acetylene"
        assert df.iloc[0]["decision"] == "accept"
        assert df.iloc[0]["target_code"] == "ethyne"
        assert "auto-accepted" in df.iloc[0]["reviewer_note"]
        assert stats == {"appended": 1, "kept_existing": 0, "rejected_or_skipped": 2}

    def test_preserves_existing_rows_and_skips_duplicate_names(self, tmp_path):
        out = tmp_path / "existing.xlsx"
        # An existing row a human had REJECTED — must not be silently overwritten.
        pd.DataFrame(
            [
                {
                    "source_name": "Acetylene",
                    "source_unit": "kg",
                    "target_name": "Ethyne",
                    "target_db": "biosphere3",
                    "decision": "reject",
                    "reviewer_note": "human said no",
                }
            ]
        ).to_excel(out, index=False)

        LlmReviewedXlsxAppender(output_path=out).append(self._suggestions())
        df = pd.read_excel(out)
        # Acetylene row preserved as-is (decision still reject), no duplicate appended.
        rows = df[df["source_name"] == "Acetylene"]
        assert len(rows) == 1
        assert rows.iloc[0]["decision"] == "reject"
        assert "human said no" in rows.iloc[0]["reviewer_note"]

    def test_does_not_create_file_when_nothing_to_write(self, tmp_path):
        out = tmp_path / "noop.xlsx"
        stats = LlmReviewedXlsxAppender(output_path=out).append([])
        assert not out.exists()
        assert stats["appended"] == 0
