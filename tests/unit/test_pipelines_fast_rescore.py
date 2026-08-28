"""Unit tests for the fast rescore path (phase 2).

Three contracts:

1. ``ParameterReevaluator.ratio_patch`` — on post-transform data whose
   amounts were rescaled by an unknown multiplicative factor, patching by
   ``new_eval / baseline_eval`` must land exactly where the absolute
   re-evaluation followed by the same rescale would (the equivalence gate,
   at unit scale).
2. ``LinkedSpCache`` — pickle round-trip + actionable error when missing.
3. ``FastRescorePipeline`` — loads cache, patches, replays
   ``LinkAllPipeline.emit_scoring_package`` (monkeypatched), reports stages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from transforms.linked_cache import LinkedCacheMissingError, LinkedSpCache
from transforms.parameter_overrides import ParameterOverride, ParameterOverridesStore
from transforms.parameter_reevaluator import ParameterReevaluator
from transforms.sp_csv_parser import ParsedSimaProCsv

_PARAMS = [
    {
        "process_code": "PROC1",
        "kind": "input",
        "name": "SP_X",
        "original_name": "x",
        "amount": 2.0,
        "formula": None,
        "comment": "",
    },
    {
        "process_code": "PROC1",
        "kind": "calculated",
        "name": "SP_Y",
        "original_name": "y",
        "amount": 4.0,
        "formula": "(SP_X * SP_X)",
        "comment": "",
    },
]

_RESCALE = 3.6  # the unknown multiplicative factor transforms applied


def _linked_data() -> list[dict]:
    """Post-transform shape: amounts = parse-level values × _RESCALE."""
    return [
        {
            "code": "PROC1",
            "type": "process",
            "exchanges": [
                # parse amount was SP_X*2 = 4.0 → linked 14.4
                {"name": "direct", "amount": 4.0 * _RESCALE, "formula": "(SP_X * 2)"},
                # parse amount was SP_Y = 4.0 → linked 14.4
                {"name": "derived", "amount": 4.0 * _RESCALE, "formula": "SP_Y"},
                {"name": "constant", "amount": 7.0},
            ],
        },
    ]


class TestRatioPatch:
    def test_matches_absolute_reevaluation_times_rescale(self):
        reev = ParameterReevaluator(parameters=_PARAMS)
        overrides = [ParameterOverride("x", "*", 3.0)]
        result = reev.ratio_patch(_linked_data(), overrides)
        by_name = {c.exchange_name: c for c in result.changes}
        # absolute: direct = 3*2 = 6 → linked-equivalent 6*RESCALE
        assert by_name["direct"].new_amount == pytest.approx(6.0 * _RESCALE)
        # absolute: derived = 3² = 9 → linked-equivalent 9*RESCALE
        assert by_name["derived"].new_amount == pytest.approx(9.0 * _RESCALE)

    def test_no_overrides_is_noop(self):
        reev = ParameterReevaluator(parameters=_PARAMS)
        assert reev.ratio_patch(_linked_data(), []).changes == ()

    def test_original_value_is_noop(self):
        reev = ParameterReevaluator(parameters=_PARAMS)
        result = reev.ratio_patch(_linked_data(), [ParameterOverride("x", "*", 2.0)])
        assert result.changes == ()

    def test_zero_baseline_is_unpatchable_and_warns(self):
        params = [dict(_PARAMS[0], amount=0.0)]  # baseline SP_X = 0
        data = [
            {
                "code": "PROC1",
                "exchanges": [{"name": "e", "amount": 0.0, "formula": "(SP_X * 2)"}],
            }
        ]
        result = ParameterReevaluator(parameters=params).ratio_patch(
            data, [ParameterOverride("x", "*", 5.0)]
        )
        assert result.changes == ()
        assert any("cannot derive the transform scale" in w for w in result.warnings)
        assert result.data[0]["exchanges"][0]["amount"] == 0.0


class TestLinkedSpCache:
    @pytest.fixture
    def cache(self, tmp_path: Path, monkeypatch) -> LinkedSpCache:
        settings = Settings()
        monkeypatch.setattr(
            type(settings.paths),
            "linked_cache_pkl",
            property(lambda self: tmp_path / "linked_cache.pkl"),
        )
        csv = tmp_path / "AGB.csv"
        csv.write_text("fake agb csv")
        monkeypatch.setattr(
            type(settings.paths),
            "agribalyse_csv",
            property(lambda self: tmp_path / "AGB.csv"),
        )
        return LinkedSpCache(settings=settings)

    def test_round_trip(self, cache):
        sp = ParsedSimaProCsv(db_name="agb", data=_linked_data(), parameters=_PARAMS)
        cache.write(sp)
        loaded = cache.load()
        assert loaded.db_name == "agb"
        assert loaded.data == sp.data
        assert loaded.parameters == _PARAMS

    def test_missing_cache_raises_actionable_error(self, cache):
        with pytest.raises(LinkedCacheMissingError, match="dds-link-all"):
            cache.load()

    def test_stale_when_source_csv_changed(self, cache):
        """A snapshot built from a different CSV must refuse to replay
        (2026-08-18 adversarial F2)."""
        import os
        import time

        from transforms.linked_cache import LinkedCacheStaleError

        cache.write(ParsedSimaProCsv(db_name="agb", data=[], parameters=[]))
        csv = cache.settings.paths.agribalyse_csv
        csv.write_text("replaced with a newer export!")
        past = time.time() - 3600
        os.utime(csv, (past, past))
        with pytest.raises(LinkedCacheStaleError, match="different source CSV"):
            cache.load()

    def test_old_format_payload_is_stale(self, cache):
        """Pre-stamp pickles (raw object, no schema wrapper) fail loudly."""
        import pickle

        from transforms.linked_cache import LinkedCacheStaleError

        cache.path.write_bytes(
            pickle.dumps(ParsedSimaProCsv(db_name="agb", data=[], parameters=[]))
        )
        with pytest.raises(LinkedCacheStaleError, match="old cache format"):
            cache.load()

    def test_truncated_pickle_raises_corrupt_error(self, cache):
        from transforms.linked_cache import LinkedCacheCorruptError

        cache.write(ParsedSimaProCsv(db_name="agb", data=[], parameters=[]))
        blob = cache.path.read_bytes()
        cache.path.write_bytes(blob[: len(blob) // 2])
        with pytest.raises(LinkedCacheCorruptError, match="rerun dds-link-all"):
            cache.load()

    def test_write_failure_leaves_no_partial(self, cache, monkeypatch):
        import pickle as pickle_mod

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(pickle_mod, "dump", boom)
        with pytest.raises(OSError):
            cache.write(ParsedSimaProCsv(db_name="agb", data=[], parameters=[]))
        assert list(cache.path.parent.glob("*.partial")) == []


class TestFastRescorePipeline:
    def test_loads_patches_and_replays_emit(self, tmp_path: Path, monkeypatch):
        from pipelines import FastRescorePipeline
        from pipelines.link_all import LinkAllPipeline

        settings = Settings()
        monkeypatch.setattr(
            type(settings.paths),
            "linked_cache_pkl",
            property(lambda self: tmp_path / "linked_cache.pkl"),
        )
        monkeypatch.setattr(
            type(settings.paths),
            "parameter_overrides_csv",
            property(lambda self: tmp_path / "parameter_overrides.csv"),
        )
        monkeypatch.setattr(
            type(settings.paths),
            "dashboard_run_report",
            property(lambda self: tmp_path / "run_report.json"),
        )
        (tmp_path / "AGB.csv").write_text("fake agb csv")
        monkeypatch.setattr(
            type(settings.paths),
            "agribalyse_csv",
            property(lambda self: tmp_path / "AGB.csv"),
        )
        LinkedSpCache(settings=settings).write(
            ParsedSimaProCsv(db_name="agb", data=_linked_data(), parameters=_PARAMS)
        )
        ParameterOverridesStore(settings=settings).upsert(ParameterOverride("x", "*", 3.0))

        seen: dict = {}

        def fake_emit(cls_sp, cls_settings, cls_report):
            seen["sp"] = cls_sp
            return {"content_hash": "deadbeef"}

        monkeypatch.setattr(
            LinkAllPipeline,
            "emit_scoring_package",
            classmethod(lambda cls, sp, s, r: fake_emit(sp, s, r)),
        )

        FastRescorePipeline(settings=settings).run()

        # The emit stage received the PATCHED graph, parameters intact.
        patched = seen["sp"].data[0]["exchanges"][0]["amount"]
        assert patched == pytest.approx(6.0 * _RESCALE)
        assert seen["sp"].parameters == _PARAMS
