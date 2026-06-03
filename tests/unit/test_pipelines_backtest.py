"""Unit tests for ``BacktestPipeline`` — L4 native-scorer cut-over.

No bw2data, no SQLite. All scoring goes through NativeLciaScorer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scoring.product_catalog as pc_mod
from core.parquet_io import ParquetAtomicWriter
from pipelines.backtest import BacktestOptions, BacktestPipeline
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.native_scorer import NativeWorkerPayload
from scoring.product_catalog import ProductCatalog
from scoring.scorer import ScoringResult
from scoring.scoring_package import ScoringPackageBuilder, ScoringPackageStore

# ---------------------------------------------------------------------------
# Helpers


def _make_catalog(tmp_path: Path, entries: list[dict]) -> Path:
    """Write a product catalog parquet under tmp_path/registry/ and return its path."""
    reg = tmp_path / "registry"
    reg.mkdir(exist_ok=True)
    target = reg / "product_catalog.parquet"
    df = pd.DataFrame(
        entries,
        columns=["database", "code", "name", "type", "unit", "product_id"],
    ).astype(
        {
            "database": "string",
            "code": "string",
            "name": "string",
            "type": "string",
            "unit": "string",
            "product_id": "int64",
        }
    )
    ParquetAtomicWriter.write(df, target)
    return target


def _make_run_report(tmp_path: Path, content_hash: str) -> Path:
    """Write dashboard/run_report.json with a valid scoring_package entry."""
    dash = tmp_path / "dashboard"
    dash.mkdir(exist_ok=True)
    report = {"stages": {"scoring_package": {"content_hash": content_hash}}}
    path = dash / "run_report.json"
    path.write_text(json.dumps(report))
    return path


def _tiny_scoring_package(store_root: Path) -> tuple[object, str]:
    """Build and store a tiny ScoringPackage; return (pkg, content_hash)."""
    sp_data = [
        {
            "database": "agb",
            "code": "wheat",
            "name": "wheat [Ciqual code: 9999]",
            "type": "product",
            "unit": "kg",
            "exchanges": [
                {
                    "type": "production",
                    "input": ("agb", "wheat-product"),
                    "amount": 1.0,
                },
                {
                    "type": "biosphere",
                    "input": ("bio3", "co2"),
                    "amount": 2.0,
                },
            ],
        }
    ]
    frame = ExchangeFrameBuilder().from_sp_data(sp_data)
    co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
    method_key = (
        "ecoinvent-3.9.1",
        "EF v3.1",
        "climate change",
        "global warming potential (GWP100)",
    )
    cfs = {method_key: pd.DataFrame({"flow_id": [co2_id], "cf": [1.0]})}
    pkg = ScoringPackageBuilder().build(frame, cfs)
    store = ScoringPackageStore(root=store_root)
    store.write(pkg)
    return pkg, pkg.content_hash


# ---------------------------------------------------------------------------
# _load_scoring_package


class TestLoadScoringPackage:
    def test_raises_file_not_found_when_no_run_report(self, settings):
        pipeline = BacktestPipeline(settings)
        with pytest.raises(FileNotFoundError, match=r"run_report\.json"):
            pipeline._load_scoring_package()

    def test_raises_key_error_when_no_content_hash(self, settings):
        dash = settings.paths.dashboard
        dash.mkdir(parents=True, exist_ok=True)
        (dash / "run_report.json").write_text(json.dumps({"stages": {}}))
        pipeline = BacktestPipeline(settings)
        with pytest.raises(KeyError, match="content_hash"):
            pipeline._load_scoring_package()

    def test_loads_package_from_store(self, settings):
        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        _pkg, content_hash = _tiny_scoring_package(store_root)
        _make_run_report(settings.paths.package_root, content_hash)
        pipeline = BacktestPipeline(settings)
        loaded = pipeline._load_scoring_package()
        assert loaded.content_hash == content_hash


# ---------------------------------------------------------------------------
# _registered_methods


class TestRegisteredMethods:
    def test_returns_only_methods_in_package(self, settings):
        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        pkg, _ = _tiny_scoring_package(store_root)
        pipeline = BacktestPipeline(settings)
        methods = pipeline._registered_methods(pkg)
        # Only the one method we baked into the package should appear.
        climate_key = (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        assert climate_key in methods
        # All returned methods are a subset of METHOD_TO_ADEME.
        assert all(m in BacktestPipeline.METHOD_TO_ADEME for m in methods)

    def test_returns_empty_when_package_has_no_matching_methods(self, settings):
        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        sp_data = [
            {
                "database": "agb",
                "code": "x",
                "name": "x",
                "type": "product",
                "unit": "kg",
                "exchanges": [
                    {"type": "production", "input": ("agb", "x-product"), "amount": 1.0},
                ],
            }
        ]
        frame = ExchangeFrameBuilder().from_sp_data(sp_data)
        # Package carries a method that is NOT in METHOD_TO_ADEME.
        pkg = ScoringPackageBuilder().build(
            frame, {("unknown", "method"): pd.DataFrame({"flow_id": [], "cf": []})}
        )
        ScoringPackageStore(root=store_root).write(pkg)
        pipeline = BacktestPipeline(settings)
        assert pipeline._registered_methods(pkg) == []


# ---------------------------------------------------------------------------
# _map_products


class TestMapProducts:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a minimal ADEME-reference-style DataFrame."""
        cols = list(BacktestPipeline.INFO_COLS)
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols].reset_index(drop=True)

    def test_ciqual_match(self, settings):
        cat_path = _make_catalog(
            settings.paths.package_root,
            [
                {
                    "database": "agb",
                    "code": "w1",
                    "name": "wheat [Ciqual code: 1234]",
                    "type": "product",
                    "unit": "kg",
                    "product_id": 1,
                },
            ],
        )
        original_load = pc_mod.ProductCatalog.load
        pc_mod.ProductCatalog.load = classmethod(
            lambda cls, p: original_load.__func__(cls, cat_path)
        )

        try:
            catalog = ProductCatalog.load(cat_path)
            df = self._make_df([{"Code CIQUAL": "1234", "LCI Name": ""}])
            pipeline = BacktestPipeline(settings)
            mapping = pipeline._map_products(df, catalog)
            assert mapping[0] == ("agb", "w1")
        finally:
            pc_mod.ProductCatalog.load = original_load

    def test_ciqual_match_with_leading_zeros(self, settings):
        """ADEME stores CIQUAL '1' as a bare integer; AGB names embed it as
        '[Ciqual code: 0001]'. The matcher must collapse both to the same
        key so backtest pass-1 finds the consumer-stage product instead of
        falling through to the substring picker (which lands on a process
        activity at a different lifecycle stage)."""
        cat_path = _make_catalog(
            settings.paths.package_root,
            [
                {
                    "database": "agb",
                    "code": "p_consumer",
                    "name": "Beetroot juice ... at consumer {FR} [Ciqual code: 0001] U",
                    "type": "product",
                    "unit": "kg",
                    "product_id": 1,
                },
                {
                    "database": "agb",
                    "code": "p_processing",
                    "name": "Beetroot juice, pure juice, at processing {DE}",
                    "type": "process",
                    "unit": "kg",
                    "product_id": 2,
                },
            ],
        )
        original_load = pc_mod.ProductCatalog.load
        pc_mod.ProductCatalog.load = classmethod(
            lambda cls, p: original_load.__func__(cls, cat_path)
        )
        try:
            catalog = ProductCatalog.load(cat_path)
            df = self._make_df([{"Code CIQUAL": "1", "LCI Name": "Beetroot juice, pure juice"}])
            pipeline = BacktestPipeline(settings)
            mapping = pipeline._map_products(df, catalog)
            # MUST resolve to the [Ciqual code: 0001] consumer product, NOT
            # the substring-shortest at-processing process.
            assert mapping[0] == ("agb", "p_consumer")
        finally:
            pc_mod.ProductCatalog.load = original_load

    def test_code_agb_disambiguates_compound_codes(self, settings):
        """ADEME stores compound CIQUAL siblings (``9341_1`` and
        ``9341_2``) as separate rows that share the same ``Code CIQUAL``
        (``9341``) but differ in ``Code AGB``. Our catalog has both
        suffix variants as distinct products. Matching by Code CIQUAL
        alone collides them; matching by Code AGB first disambiguates."""
        cat_path = _make_catalog(
            settings.paths.package_root,
            [
                {
                    "database": "agb",
                    "code": "p_9341_1",
                    "name": "Quinoa, intl ... at consumer {FR} [Ciqual code: 9341_1] U",
                    "type": "product",
                    "unit": "kg",
                    "product_id": 1,
                },
                {
                    "database": "agb",
                    "code": "p_9341_2",
                    "name": "Quinoa, FR ... at consumer {FR} [Ciqual code: 9341_2] U",
                    "type": "product",
                    "unit": "kg",
                    "product_id": 2,
                },
            ],
        )
        original_load = pc_mod.ProductCatalog.load
        pc_mod.ProductCatalog.load = classmethod(
            lambda cls, p: original_load.__func__(cls, cat_path)
        )
        try:
            catalog = ProductCatalog.load(cat_path)
            df = self._make_df(
                [
                    {
                        "Code AGB": "9341_1",
                        "Code CIQUAL": "9341",
                        "LCI Name": "Quinoa, intl",
                    },
                    {
                        "Code AGB": "9341_2",
                        "Code CIQUAL": "9341",
                        "LCI Name": "Quinoa, FR",
                    },
                ]
            )
            pipeline = BacktestPipeline(settings)
            mapping = pipeline._map_products(df, catalog)
            assert mapping[0] == ("agb", "p_9341_1")
            assert mapping[1] == ("agb", "p_9341_2")
        finally:
            pc_mod.ProductCatalog.load = original_load

    def test_ciqual_match_compound_code(self, settings):
        """ADEME stores compound codes like '9341_1'; the regex used to
        extract from '[Ciqual code: 9341_1]' must capture the full code,
        not just the leading digits ('9341'). Otherwise pass-1 misses
        and the substring picker mis-routes the row to a different stage."""
        cat_path = _make_catalog(
            settings.paths.package_root,
            [
                {
                    "database": "agb",
                    "code": "p_consumer",
                    "name": "Quinoa cooked ... at consumer {FR} [Ciqual code: 9341_1] U",
                    "type": "product",
                    "unit": "kg",
                    "product_id": 1,
                },
                {
                    "database": "agb",
                    "code": "p_processing",
                    "name": "Quinoa, boiled/cooked in water, unsalted",
                    "type": "process",
                    "unit": "kg",
                    "product_id": 2,
                },
            ],
        )
        original_load = pc_mod.ProductCatalog.load
        pc_mod.ProductCatalog.load = classmethod(
            lambda cls, p: original_load.__func__(cls, cat_path)
        )
        try:
            catalog = ProductCatalog.load(cat_path)
            df = self._make_df(
                [{"Code CIQUAL": "9341_1", "LCI Name": "Quinoa, boiled/cooked in water, unsalted"}]
            )
            pipeline = BacktestPipeline(settings)
            mapping = pipeline._map_products(df, catalog)
            assert mapping[0] == ("agb", "p_consumer")
        finally:
            pc_mod.ProductCatalog.load = original_load

    def test_ciqual_storage_disambiguates_twin_product_variants(self, settings):
        """ADEME tags Confit de canard 8110 with storage 'Glacé' (preserved
        shelf-stable). AGB ships two products under that Ciqual code: a
        canned ``Ambient (long)`` variant (whose computed score matches
        ADEME's reference exactly) and a chilled ``Preserved duck`` Pack
        variant (which carries a different feed chain and scores ~5× higher
        on cc_luc). The pre-fix matcher fell through to LCI Name match in
        Pass 2 and picked the chilled twin because its English name
        ("Preserved duck") happens to equal ADEME's LCI Name column.

        Storage disambiguation: when Pass 1 returns multiple product
        candidates, parse the AGB English storage label (``Ambient (long)``
        / ``Chilled`` / ``Frozen`` …) from each candidate name and pick the
        one matching ADEME's Storage column via the documented French→
        English mapping (``Glacé`` → ``Ambient (long)``, ``Congelé`` →
        ``Frozen``, etc.).
        """
        cat_path = _make_catalog(
            settings.paths.package_root,
            [
                {
                    "database": "agb",
                    "code": "p_canned",
                    "name": (
                        "Duck confit, canned, processed in FR | Ambient (long) | "
                        "Pack | Oven and Pan frying | at consumer {FR} "
                        "[Ciqual code: 8110] U"
                    ),
                    "type": "product",
                    "unit": "kg",
                    "product_id": 1,
                },
                {
                    "database": "agb",
                    "code": "p_chilled",
                    "name": (
                        "Preserved duck, processed in FR | Chilled | Pack | "
                        "Oven | at consumer {FR} [Ciqual code: 8110] U"
                    ),
                    "type": "product",
                    "unit": "kg",
                    "product_id": 2,
                },
            ],
        )
        original_load = pc_mod.ProductCatalog.load
        pc_mod.ProductCatalog.load = classmethod(
            lambda cls, p: original_load.__func__(cls, cat_path)
        )
        try:
            catalog = ProductCatalog.load(cat_path)
            df = self._make_df(
                [
                    {
                        "Code AGB": "8110",
                        "Code CIQUAL": "8110",
                        "LCI Name": "Preserved duck",
                        "Storage": "Glacé",
                    }
                ]
            )
            pipeline = BacktestPipeline(settings)
            mapping = pipeline._map_products(df, catalog)
            assert mapping[0] == ("agb", "p_canned"), (
                "Pass 1.5 must disambiguate via Storage='Glacé' → "
                "'Ambient (long)' before LCI Name fallback picks the "
                "chilled English-name twin"
            )
        finally:
            pc_mod.ProductCatalog.load = original_load

    def test_exact_name_mode(self, settings):
        cat_path = _make_catalog(
            settings.paths.package_root,
            [
                {
                    "database": "agb",
                    "code": "w1",
                    "name": "Organic wheat, at farm",
                    "type": "process",
                    "unit": "kg",
                    "product_id": 2,
                },
            ],
        )
        original_load = pc_mod.ProductCatalog.load
        pc_mod.ProductCatalog.load = classmethod(
            lambda cls, p: original_load.__func__(cls, cat_path)
        )

        try:
            catalog = ProductCatalog.load(cat_path)
            df = self._make_df([{"LCI Name": "Organic wheat, at farm", "Code CIQUAL": ""}])
            pipeline = BacktestPipeline(settings, options=BacktestOptions(match_mode="exact_name"))
            mapping = pipeline._map_products(df, catalog)
            assert mapping[0] == ("agb", "w1")
        finally:
            pc_mod.ProductCatalog.load = original_load

    def test_unmatched_rows_return_none(self, settings):
        cat_path = _make_catalog(settings.paths.package_root, [])
        original_load = pc_mod.ProductCatalog.load
        pc_mod.ProductCatalog.load = classmethod(
            lambda cls, p: original_load.__func__(cls, cat_path)
        )

        try:
            catalog = ProductCatalog.load(cat_path)
            df = self._make_df([{"Code CIQUAL": "9999", "LCI Name": "no match"}])
            pipeline = BacktestPipeline(settings)
            mapping = pipeline._map_products(df, catalog)
            assert mapping[0] is None
        finally:
            pc_mod.ProductCatalog.load = original_load


# ---------------------------------------------------------------------------
# _compute_scores


class TestComputeScores:
    def test_serial_path_calls_native_scorer(self, settings, monkeypatch):
        """n_workers=1 → in-process NativeLciaScorer, no ProcessPoolExecutor."""
        called = {"pool": False}

        class _Boom:
            def __init__(self, *_, **__):
                called["pool"] = True
                raise AssertionError("pool must not be used for n_workers=1")

        monkeypatch.setattr("pipelines.backtest.ProcessPoolExecutor", _Boom)

        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        pkg, _ = _tiny_scoring_package(store_root)

        product_key = ("agb", "wheat")
        product_id = ExchangeFrameBuilder.flow_id_for(("agb", "wheat-product"))

        # Stub ProductCatalog.load so product_id_for returns our id.
        class _FakeCatalog:
            def product_id_for(self, db: str, code: str) -> int | None:
                if (db, code) == product_key:
                    return product_id
                return None

        monkeypatch.setattr(
            pc_mod.ProductCatalog, "load", classmethod(lambda cls, p: _FakeCatalog())
        )

        method_key = (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        pipeline = BacktestPipeline(settings, options=BacktestOptions(solver="scipy", n_workers=1))
        row_to_key = {0: product_key}
        catalog = _FakeCatalog()
        out = pipeline._compute_scores(row_to_key, [method_key], pkg, catalog)
        assert called["pool"] is False
        assert product_key in out
        assert method_key in out[product_key]

    def test_returns_empty_dict_when_no_keys(self, settings):
        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        pkg, _ = _tiny_scoring_package(store_root)

        cat_path = _make_catalog(settings.paths.package_root, [])
        catalog = ProductCatalog.load(cat_path)

        pipeline = BacktestPipeline(settings)
        out = pipeline._compute_scores({0: None}, [], pkg, catalog)
        assert out == {}

    def test_raises_if_all_results_empty(self, settings, monkeypatch):
        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        pkg, _ = _tiny_scoring_package(store_root)

        class _FakeCatalog:
            def product_id_for(self, db: str, code: str) -> int | None:
                return 999_999_999  # bogus id → skip_reason set, scores=None

        monkeypatch.setattr(
            pc_mod.ProductCatalog, "load", classmethod(lambda cls, p: _FakeCatalog())
        )

        method_key = (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )
        pipeline = BacktestPipeline(settings, options=BacktestOptions(solver="scipy", n_workers=1))
        row_to_key = {0: ("agb", "wheat")}
        catalog = _FakeCatalog()
        with pytest.raises(RuntimeError, match="empty"):
            pipeline._compute_scores(row_to_key, [method_key], pkg, catalog)


# ---------------------------------------------------------------------------
# _score_parallel_native


class TestScoreParallelNative:
    def test_chunks_and_merges(self, settings, monkeypatch):
        """Three products split across 2 workers; results merged."""
        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        pkg, _ = _tiny_scoring_package(store_root)

        class _SerialPool:
            def __init__(self, *_, **__):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def map(self, fn, payloads):
                return [fn(p) for p in payloads]

        monkeypatch.setattr("pipelines.backtest.ProcessPoolExecutor", _SerialPool)

        method_key = (
            "ecoinvent-3.9.1",
            "EF v3.1",
            "climate change",
            "global warming potential (GWP100)",
        )

        captured_payloads: list[NativeWorkerPayload] = []

        class _FakeWorker:
            def __call__(self, payload: NativeWorkerPayload) -> dict:
                captured_payloads.append(payload)
                return {
                    key: ScoringResult(process_key=key, scores={method_key: 1.0}, elapsed_s=0.0)
                    for key, _ in payload.products
                }

        monkeypatch.setattr("pipelines.backtest.NativeScoreWorker", _FakeWorker)

        pipeline = BacktestPipeline(settings, options=BacktestOptions(solver="scipy", n_workers=2))
        products = [
            (("agb", "a"), 1),
            (("agb", "b"), 2),
            (("agb", "c"), 3),
        ]
        merged = pipeline._score_parallel_native(products, [method_key], pkg, use_pardiso=False)

        assert len(captured_payloads) == 2
        all_keys = sorted(k for p in captured_payloads for k, _ in p.products)
        assert all_keys == [("agb", "a"), ("agb", "b"), ("agb", "c")]
        assert sorted(merged.keys()) == [("agb", "a"), ("agb", "b"), ("agb", "c")]
        assert all(p.content_hash == pkg.content_hash for p in captured_payloads)
        assert all(p.use_pardiso is False for p in captured_payloads)

    def test_returns_empty_dict_for_empty_products(self, settings):
        store_root = settings.paths.scoring_packages_root
        store_root.mkdir(parents=True, exist_ok=True)
        pkg, _ = _tiny_scoring_package(store_root)
        pipeline = BacktestPipeline(settings)
        result = pipeline._score_parallel_native([], [], pkg, use_pardiso=False)
        assert result == {}


# ---------------------------------------------------------------------------
# Pass 2 tie-break and substring fallback


def test_map_products_substring_fallback(tmp_path):
    """Pass 2 substring fallback picks shortest matching name."""
    from tests.fixtures.builders import make_settings

    settings = make_settings(tmp_path)
    for d in ("source", "cache", "registry", "dashboard", "to_review", "unlinked"):
        (tmp_path / d).mkdir(exist_ok=True)
    (tmp_path / "source" / "randonneur_packages").mkdir(exist_ok=True)

    cat_path = _make_catalog(
        tmp_path,
        [
            {
                "database": "agb",
                "code": "c1",
                "name": "beef burger, at supermarket",
                "type": "process",
                "unit": "kg",
                "product_id": 1,
            },
            {
                "database": "agb",
                "code": "c2",
                "name": "beef burger extra large, at supermarket",
                "type": "process",
                "unit": "kg",
                "product_id": 2,
            },
        ],
    )

    catalog = ProductCatalog.load(cat_path)
    pipeline = BacktestPipeline(settings=settings)

    ademe_df = pd.DataFrame(
        {
            "Code AGB": ["X"],
            "Code CIQUAL": [None],
            "Groupe d'aliment": [""],
            "Sous-groupe d'aliment": [""],
            "Nom du Produit": [""],
            "LCI Name": ["beef burger"],
        }
    )
    row_to_key = pipeline._map_products(ademe_df, catalog)
    # Should pick the shorter name "beef burger, at supermarket"
    assert row_to_key[0] == ("agb", "c1")


def _make_synthese_parquet(target: Path, data_rows: list[list[str]]) -> None:
    """Build a tiny ADEME synthese parquet at ``target``.

    The real file at ``source/AGRIBALYSE3.2_reference_synthese_raw.parquet``
    has 32 string-typed columns named ``"0".."31"``. Rows 0..2 are
    header / metadata; rows 3+ are data. Column 0 = Code AGB, 4 = Nom
    du Produit (French). Indices 13..31 carry the per-method numeric
    columns (passed through ``pd.to_numeric`` in ``_load_ademe_reference``).
    """
    cols = [str(i) for i in range(32)]
    headers = [["meta"] * 32, ["header2"] * 32, ["header3"] * 32]
    rows = headers + [r + [""] * (32 - len(r)) for r in data_rows]
    df = pd.DataFrame(rows, columns=cols).astype("string")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)


def test_load_ademe_reference_drops_emballage_corrige_twins(tmp_path):
    """ADEME's synthese ships two rows per product where one carries the
    legacy chain (``version "logiciel" - avec erreur sur l'emballage``)
    and a sibling re-runs scoring on a corrected packaging chain
    (``emballage corrigé``). Our matrix has only the original AGB
    activity. The corrected sibling produces double-rows on the same
    Code AGB and inflates the outlier list at +600 % ecotox / -47 %
    e_fw on Sardine 26034 and the Hareng family. Drop them at load.
    """
    from tests.fixtures.builders import make_settings

    settings = make_settings(tmp_path)
    for d in ("source", "cache", "registry", "dashboard", "to_review", "unlinked"):
        (tmp_path / d).mkdir(exist_ok=True)
    (tmp_path / "source" / "randonneur_packages").mkdir(exist_ok=True)

    _make_synthese_parquet(
        settings.paths.ademe_reference_synthese_raw,
        [
            [
                "26034",
                "26034",
                "viandes",
                "poissons",
                "Sardine, à l'huile, appertisée, égouttée "
                '(version "logiciel" - avec erreur sur l\'emballage)',
                "European pilchard or sardine, in oil, canned, drained",
            ],
            [
                "26034",
                "26034",
                "viandes",
                "poissons",
                "Sardine, à l'huile, appertisée, égouttée (emballage corrigé)",
                "European pilchard or sardine, in oil, canned, drained (packaging fixed)",
            ],
            [
                "9999",
                "9999",
                "céréaliers",
                "pâtes",
                "Beurre",
                "Butter",
            ],
        ],
    )

    pipeline = BacktestPipeline(settings)
    df = pipeline._load_ademe_reference()

    names = df["Nom du Produit"].tolist()
    # Both the legacy "logiciel" twin AND the unrelated row remain.
    assert any("Sardine" in n for n in names), names
    # The "emballage corrigé" twin is filtered out.
    assert all("emballage corrig" not in n.lower() for n in names), names
    # The non-twin row is preserved.
    assert any(n == "Beurre" for n in names), names


def test_map_products_pass2_product_only_tiebreak(tmp_path):
    """When exact LCI Name matches both process and product type, product wins."""
    from tests.fixtures.builders import make_settings

    settings = make_settings(tmp_path)
    for d in ("source", "cache", "registry", "dashboard", "to_review", "unlinked"):
        (tmp_path / d).mkdir(exist_ok=True)
    (tmp_path / "source" / "randonneur_packages").mkdir(exist_ok=True)

    lci_name = "beef, slaughtered"
    cat_path = _make_catalog(
        tmp_path,
        [
            {
                "database": "agb",
                "code": "proc1",
                "name": lci_name,
                "type": "process",
                "unit": "kg",
                "product_id": 1,
            },
            {
                "database": "agb",
                "code": "prod1",
                "name": lci_name,
                "type": "product",
                "unit": "kg",
                "product_id": 2,
            },
        ],
    )

    catalog = ProductCatalog.load(cat_path)
    pipeline = BacktestPipeline(settings=settings)

    ademe_df = pd.DataFrame(
        {
            "Code AGB": ["Y"],
            "Code CIQUAL": [None],
            "Groupe d'aliment": [""],
            "Sous-groupe d'aliment": [""],
            "Nom du Produit": [""],
            "LCI Name": [lci_name],
        }
    )
    row_to_key = pipeline._map_products(ademe_df, catalog)
    # product type wins over process when exact name is ambiguous
    assert row_to_key[0] == ("agb", "prod1")
