"""Unit tests for ``ExchangeFrameBuilder`` (sp.data → ExchangeFrame),
``NativeLciaScorer`` (ScoringPackage → scores), ``NativeWorkerPayload``,
and ``NativeScoreWorker``."""

from __future__ import annotations

import pandas as pd
import pytest

from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.native_scorer import NativeLciaScorer, NativeScoreWorker, NativeWorkerPayload
from scoring.scoring_package import ScoringPackageBuilder, ScoringPackageStore


def _activity(database, code, name, exchanges, **extra):
    return {
        "database": database,
        "code": code,
        "name": name,
        "exchanges": exchanges,
        **extra,
    }


class TestExchangeFrameBuilder:
    def test_extracts_production_and_consumption(self):
        sp_data = [
            _activity(
                "agb",
                "p1",
                "wheat",
                exchanges=[
                    {
                        "type": "production",
                        "input": ("agb", "wheat-product"),
                        "amount": 1.0,
                    },
                    {
                        "type": "technosphere",
                        "input": ("ecoinvent", "fertiliser"),
                        "amount": 0.5,
                    },
                    {
                        "type": "biosphere",
                        "input": ("biosphere3", "co2"),
                        "amount": 0.1,
                    },
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        assert frame.n_rows == 3
        assert len(frame.production) == 1
        assert len(frame.biosphere) == 1

    def test_unlinked_exchanges_dropped(self):
        # An exchange without ``input`` is unlinked — must be excluded
        # from the frame, not silently included with bogus ids.
        sp_data = [
            _activity(
                "agb",
                "p1",
                "wheat",
                exchanges=[
                    {
                        "type": "production",
                        "input": ("agb", "wheat-product"),
                        "amount": 1.0,
                    },
                    {"type": "technosphere", "amount": 0.3},  # no input
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        assert frame.n_rows == 1

    def test_id_assignment_is_deterministic(self):
        # Same (database, code) always hashes to the same int — caches
        # keyed by content hash stay stable across runs.
        a = ExchangeFrameBuilder.flow_id_for(("db", "code1"))
        b = ExchangeFrameBuilder.flow_id_for(("db", "code1"))
        assert a == b
        c = ExchangeFrameBuilder.flow_id_for(("db", "code2"))
        assert a != c

    def test_flow_id_for_is_deterministic_classmethod(self):
        # Public classmethod variant — used by ProductCatalogBuilder and
        # the L3 method-CF projection. Two calls with the same (db, code)
        # MUST collapse to the same int across module loads.
        a = ExchangeFrameBuilder.flow_id_for(("agb", "wheat"))
        b = ExchangeFrameBuilder.flow_id_for(("agb", "wheat"))
        assert a == b
        # Same code in a different database must yield a different id —
        # the database string is part of the hash input.
        c = ExchangeFrameBuilder.flow_id_for(("ecoinvent", "wheat"))
        assert a != c

    def test_flow_id_for_fits_in_int63(self):
        # Hash truncation must keep ids inside int64 (we mask to 63 bits
        # so pandas/numpy never silently downcast to int32 on Windows).
        for key in (("a", "b"), ("longer", "key-string"), ("", "")):
            value = ExchangeFrameBuilder.flow_id_for(key)
            assert 0 <= value < (1 << 63)

    def test_flow_id_for_matches_internal_assignment(self):
        # The id assigned to an activity inside ``from_sp_data`` is the
        # same one ``flow_id_for`` returns — no drift between the two
        # paths.
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {
                        "type": "production",
                        "input": ("agb", "wheat-product"),
                        "amount": 1.0,
                    },
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        expected_output = ExchangeFrameBuilder.flow_id_for(("agb", "wheat"))
        expected_input = ExchangeFrameBuilder.flow_id_for(("agb", "wheat-product"))
        assert frame.df["output_id"].iloc[0] == expected_output
        assert frame.df["input_id"].iloc[0] == expected_input

    def test_allocation_factor_carried_from_properties(self):
        sp_data = [
            _activity(
                "agb",
                "p1",
                "milk-cream",
                exchanges=[
                    {
                        "type": "production",
                        "input": ("agb", "milk"),
                        "amount": 1.0,
                        "properties": {"manual_allocation": 0.7},
                    },
                    {
                        "type": "production",
                        "input": ("agb", "cream"),
                        "amount": 1.0,
                        "properties": {"manual_allocation": 0.3},
                    },
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        prod = frame.production
        factors = sorted(prod["allocation_factor"].tolist())
        assert factors == [pytest.approx(0.3), pytest.approx(0.7)]


class TestNativeLciaScorerEndToEnd:
    """Wire ExchangeFrameBuilder → ScoringPackageBuilder → NativeLciaScorer
    through a tiny synthetic system. Verifies the SQLite-free scoring
    path produces the analytically correct score."""

    def test_one_product_one_method_score(self):
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {
                        "type": "production",
                        "input": ("agb", "wheat-product"),
                        "amount": 1.0,
                    },
                    {
                        "type": "biosphere",
                        "input": ("bio3", "co2"),
                        "amount": 2.0,  # 2 kg CO2 per kg wheat
                    },
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        cfs = {("ef", "climate"): pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        package = ScoringPackageBuilder().build(frame, cfs)

        scorer = NativeLciaScorer(package=package, use_pardiso=False)
        product_id = ExchangeFrameBuilder.flow_id_for(("agb", "wheat-product"))
        results = scorer.score([(("agb", "wheat"), product_id)], [("ef", "climate")])

        assert results[("agb", "wheat")].scores == {("ef", "climate"): pytest.approx(2.0)}
        assert results[("agb", "wheat")].skip_reason is None

    def test_unknown_method_raises_early(self):
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {
                        "type": "production",
                        "input": ("agb", "wheat-product"),
                        "amount": 1.0,
                    },
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        package = ScoringPackageBuilder().build(frame, {})
        scorer = NativeLciaScorer(package=package, use_pardiso=False)
        with pytest.raises(ValueError, match="does not carry"):
            scorer.score([], [("never-registered",)])

    def test_unknown_product_returns_skip_reason(self):
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {
                        "type": "production",
                        "input": ("agb", "wheat-product"),
                        "amount": 1.0,
                    },
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        cfs = {("m",): pd.DataFrame({"flow_id": [], "cf": []}, dtype="float64")}
        # Need at least one biosphere flow for the CF builder to make a non-trivial
        # CSR matrix; an empty cf_df returns a (1, 0) matrix and that's fine here.
        package = ScoringPackageBuilder().build(frame, cfs)
        scorer = NativeLciaScorer(package=package, use_pardiso=False)
        results = scorer.score([(("agb", "x"), 9_999_999_999)], [("m",)])
        assert results[("agb", "x")].skip_reason == "product id not in technosphere"
        assert results[("agb", "x")].scores is None


class TestNativeWorkerPayload:
    """NativeWorkerPayload is a frozen dataclass — verify construction and pickling."""

    def _make_tiny_package_store(self, tmp_path):
        sp_data = [
            _activity(
                "agb",
                "wheat",
                "wheat",
                exchanges=[
                    {"type": "production", "input": ("agb", "wheat-product"), "amount": 1.0},
                    {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
                ],
            )
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        method_key = ("ef", "climate")
        cfs = {method_key: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
        pkg = ScoringPackageBuilder().build(frame, cfs)
        store_root = tmp_path / "store"
        store_root.mkdir()
        ScoringPackageStore(root=store_root).write(pkg)
        return store_root, pkg

    def test_payload_is_immutable(self, tmp_path):
        store_root, pkg = self._make_tiny_package_store(tmp_path)
        payload = NativeWorkerPayload(
            store_root=store_root,
            content_hash=pkg.content_hash,
            products=((("agb", "wheat"), 1),),
            methods=(("ef", "climate"),),
            use_pardiso=False,
        )
        with pytest.raises(AttributeError):  # frozen dataclass
            payload.use_pardiso = True  # type: ignore[misc]

    def test_native_score_worker_returns_correct_scores(self, tmp_path):
        store_root, pkg = self._make_tiny_package_store(tmp_path)
        product_id = ExchangeFrameBuilder.flow_id_for(("agb", "wheat-product"))
        method_key = ("ef", "climate")
        payload = NativeWorkerPayload(
            store_root=store_root,
            content_hash=pkg.content_hash,
            products=((("agb", "wheat"), product_id),),
            methods=(method_key,),
            use_pardiso=False,
        )
        results = NativeScoreWorker()(payload)
        assert ("agb", "wheat") in results
        scores = results[("agb", "wheat")].scores
        assert scores is not None
        assert scores[method_key] == pytest.approx(2.0)

    def test_native_score_worker_unknown_product_returns_skip(self, tmp_path):
        store_root, pkg = self._make_tiny_package_store(tmp_path)
        payload = NativeWorkerPayload(
            store_root=store_root,
            content_hash=pkg.content_hash,
            products=((("agb", "ghost"), 9_999_999_999),),
            methods=(("ef", "climate"),),
            use_pardiso=False,
        )
        results = NativeScoreWorker()(payload)
        assert results[("agb", "ghost")].skip_reason == "product id not in technosphere"
        assert results[("agb", "ghost")].scores is None
