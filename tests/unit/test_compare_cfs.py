"""Unit tests for ``cli.compare_cfs`` — CF comparison CLI."""

from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import pytest

from cli.compare_cfs import (
    CfComparator,
    CfStatsComputer,
    CompareCfsCli,
    CompareConfig,
    ConsoleReporter,
    Ef31CfLoader,
    MethodAliasResolver,
    MethodCfsRegistryLoader,
    MethodNameNormalizer,
    MethodRow,
    SimaproCfLoader,
)


class TestMethodNameNormalizer:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Acidification", "acidification"),
            ("Climate change", "climate change"),
            ("Climate change - Biogenic", "climate change biogenic"),
            ("Climate change-Biogenic", "climate change biogenic"),
            (
                "Climate change - Land use and LU change",
                "climate change land use and land use change",
            ),
            (
                "Climate change-Land use and land use change",
                "climate change land use and land use change",
            ),
            ("Ecotoxicity, freshwater - inorganics", "ecotoxicity freshwater inorganics"),
            ("Ecotoxicity, freshwater_inorganics", "ecotoxicity freshwater inorganics"),
            ("Eutrophication, marine", "eutrophication marine"),
            ("Eutrophication marine", "eutrophication marine"),
            ("Ionising radiation", "ionising radiation"),
            ("Ionising radiation, human health", "ionising radiation"),
            ("Particulate matter", "particulate matter"),
            ("EF-particulate Matter", "particulate matter"),
            ("Photochemical ozone formation", "photochemical ozone formation"),
            ("Photochemical ozone formation - human health", "photochemical ozone formation"),
            ("  Acidification  ", "acidification"),
        ],
    )
    def test_normalize_pairs(self, raw: str, expected: str) -> None:
        assert MethodNameNormalizer.normalize(raw) == expected


class TestCfStatsComputer:
    def test_compute_known_values(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = CfStatsComputer.compute(series)
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert math.isclose(stats["std"], series.std(), rel_tol=1e-9)
        assert stats["sum"] == 15.0

    def test_compute_empty_series(self) -> None:
        series = pd.Series([], dtype="float64")
        stats = CfStatsComputer.compute(series)
        assert stats["count"] == 0
        # pandas returns NaN for min/max/mean/median/std of empty series;
        # sum of empty series is 0.0 by pandas convention.
        for key in ("min", "max", "mean", "median", "std"):
            assert math.isnan(stats[key])
        assert stats["sum"] == 0.0

    def test_stat_keys_order_is_stable(self) -> None:
        assert CfStatsComputer.STAT_KEYS == (
            "count",
            "min",
            "max",
            "mean",
            "median",
            "std",
            "sum",
        )


class TestSimaproCfLoader:
    def test_load_returns_method_raw_and_cf(self, tmp_path: Path) -> None:
        src = pd.DataFrame(
            {
                "simapro_method": ["Acidification", "Climate change"],
                "simapro_method_unit": ["mol H+ eq", "kg CO2 eq"],
                "compartment": ["Air", "Air"],
                "sub_compartment": ["(unspecified)", "(unspecified)"],
                "name": ["Ammonia", "CO2"],
                "cas": ["007664-41-7", "000124-38-9"],
                "cf": [3.02, 1.0],
                "flow_unit": ["kg", "kg"],
                "cf_unit": ["mol H+ eq / kg", "kg CO2 eq / kg"],
            }
        )
        path = tmp_path / "sp.parquet"
        src.to_parquet(path)

        out = SimaproCfLoader(path).load()
        assert list(out.columns) == ["method_raw", "cf"]
        assert len(out) == 2
        assert set(out["method_raw"]) == {"Acidification", "Climate change"}
        assert out.loc[out["method_raw"] == "Acidification", "cf"].iloc[0] == 3.02

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SimaproCfLoader(tmp_path / "does-not-exist.parquet").load()


class TestEf31CfLoader:
    def test_load_returns_method_raw_and_cf(self, tmp_path: Path) -> None:
        src = pd.DataFrame(
            {
                "FLOW_uuid": ["uuid-1", "uuid-2"],
                "FLOW_name": ["ammonia", "carbon dioxide"],
                "LCIAMethod_uuid EF3.1": ["m-1", "m-2"],
                "LCIAMethod_name": ["Acidification", "Climate change"],
                "CF EF3.1": [4.0, 1.0],
                "LCIAMethod_location": ["DE", ""],
                "FLOW_class0": ["Emissions", "Emissions"],
                "FLOW_class1": ["Emissions to air", "Emissions to air"],
                "FLOW_class2": ["x", "y"],
                "LCIAMethod_derivation": ["Calculated", "Calculated"],
                "LCIAMethod_direction": ["Output", "Output"],
            }
        )
        path = tmp_path / "ef.parquet"
        src.to_parquet(path)

        out = Ef31CfLoader(path).load()
        assert list(out.columns) == ["method_raw", "cf"]
        assert len(out) == 2
        assert set(out["method_raw"]) == {"Acidification", "Climate change"}
        assert out.loc[out["method_raw"] == "Acidification", "cf"].iloc[0] == 4.0

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Ef31CfLoader(tmp_path / "does-not-exist.parquet").load()


class _StubLoader:
    """Minimal duck-typed loader for tests."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def load(self) -> pd.DataFrame:
        return self._df


class TestCfComparator:
    def _make_sp(self, methods: list[str], cfs: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"method_raw": methods, "cf": cfs})

    def _make_ef(self, methods: list[str], cfs: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"method_raw": methods, "cf": cfs})

    def test_build_rows_method_in_both(self) -> None:
        sp = self._make_sp(["Acidification", "Acidification"], [1.0, 3.0])
        ef = self._make_ef(["Acidification", "Acidification"], [2.0, 4.0])
        comparator = CfComparator(
            simapro_loader=_StubLoader(sp),
            ef31_loader=_StubLoader(ef),
            method_filters=(),
        )
        rows = comparator.build_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row.norm_key == "acidification"
        assert row.display_name == "Acidification"
        assert row.simapro_stats is not None
        assert row.ef31_stats is not None
        assert row.simapro_stats["count"] == 2
        assert row.ef31_stats["mean"] == 3.0

    def test_build_rows_method_only_simapro(self) -> None:
        sp = self._make_sp(["Land use"], [10.0])
        ef = self._make_ef(["Acidification"], [1.0])
        comparator = CfComparator(
            simapro_loader=_StubLoader(sp),
            ef31_loader=_StubLoader(ef),
            method_filters=(),
        )
        rows = {row.norm_key: row for row in comparator.build_rows()}
        assert rows["land use"].ef31_stats is None
        assert rows["land use"].simapro_stats is not None
        assert rows["acidification"].simapro_stats is None
        assert rows["acidification"].ef31_stats is not None

    def test_build_rows_normalises_method_names(self) -> None:
        sp = self._make_sp(["Ionising radiation"], [1.0])
        ef = self._make_ef(["Ionising radiation, human health"], [2.0])
        comparator = CfComparator(
            simapro_loader=_StubLoader(sp),
            ef31_loader=_StubLoader(ef),
            method_filters=(),
        )
        rows = comparator.build_rows()
        assert len(rows) == 1
        # SimaPro name preferred for display.
        assert rows[0].display_name == "Ionising radiation"

    def test_build_rows_falls_back_to_ef31_display_name(self) -> None:
        sp = self._make_sp([], [])
        ef = self._make_ef(["Climate change"], [1.0])
        comparator = CfComparator(
            simapro_loader=_StubLoader(sp),
            ef31_loader=_StubLoader(ef),
            method_filters=(),
        )
        rows = comparator.build_rows()
        assert len(rows) == 1
        assert rows[0].display_name == "Climate change"

    def test_build_rows_sorted_by_norm_key(self) -> None:
        sp = self._make_sp(["Water use", "Acidification"], [1.0, 1.0])
        ef = self._make_ef(["Land use"], [1.0])
        comparator = CfComparator(
            simapro_loader=_StubLoader(sp),
            ef31_loader=_StubLoader(ef),
            method_filters=(),
        )
        rows = comparator.build_rows()
        assert [r.norm_key for r in rows] == [
            "acidification",
            "land use",
            "water use",
        ]

    def test_build_rows_applies_filter(self) -> None:
        sp = self._make_sp(["Acidification", "Climate change"], [1.0, 2.0])
        ef = self._make_ef(["Acidification", "Climate change"], [1.0, 2.0])
        comparator = CfComparator(
            simapro_loader=_StubLoader(sp),
            ef31_loader=_StubLoader(ef),
            method_filters=("climate",),
        )
        rows = comparator.build_rows()
        assert len(rows) == 1
        assert rows[0].display_name == "Climate change"

    def test_build_rows_empty_filter_match_returns_empty(self) -> None:
        sp = self._make_sp(["Acidification"], [1.0])
        ef = self._make_ef(["Acidification"], [1.0])
        comparator = CfComparator(
            simapro_loader=_StubLoader(sp),
            ef31_loader=_StubLoader(ef),
            method_filters=("nonsense",),
        )
        assert comparator.build_rows() == []


class TestConsoleReporter:
    def _row(self, name: str, sp_stats=None, ef_stats=None) -> MethodRow:
        return MethodRow(
            display_name=name,
            norm_key=MethodNameNormalizer.normalize(name),
            simapro_stats=sp_stats,
            ef31_stats=ef_stats,
        )

    def _stats(self, **overrides) -> dict[str, float]:
        base = {
            "count": 5,
            "min": 0.0,
            "max": 10.0,
            "mean": 5.0,
            "median": 5.0,
            "std": 3.16,
            "sum": 25.0,
        }
        base.update(overrides)
        return base

    def test_renders_block_per_method(self) -> None:
        rows = [
            self._row("Acidification", self._stats(), self._stats(mean=4.9)),
            self._row("Climate change", self._stats(count=10), self._stats(count=12)),
        ]
        buf = io.StringIO()
        ConsoleReporter(decimals=4).render(rows, stream=buf)
        out = buf.getvalue()
        assert "Acidification" in out
        assert "Climate change" in out
        assert "SimaPro" in out
        assert "EF31" in out
        for label in ("count", "min", "max", "mean", "median", "std", "sum"):
            assert label in out

    def test_renders_em_dash_for_missing_side(self) -> None:
        rows = [self._row("Land use", self._stats(), None)]
        buf = io.StringIO()
        ConsoleReporter(decimals=4).render(rows, stream=buf)
        out = buf.getvalue()
        # Each of the 7 stat rows should show "—" on the EF31 side.
        assert out.count("—") >= 7

    def test_decimals_controls_precision(self) -> None:
        rows = [self._row("Acidification", self._stats(mean=1.23456789), self._stats(mean=1.0))]
        buf = io.StringIO()
        ConsoleReporter(decimals=2).render(rows, stream=buf)
        assert "1.23" in buf.getvalue()
        assert "1.2346" not in buf.getvalue()

    def test_footer_counts(self) -> None:
        rows = [
            self._row("Acidification", self._stats(), self._stats()),  # both
            self._row("Land use", self._stats(), None),  # SP only
            self._row("Resource use", None, self._stats()),  # EF only
        ]
        buf = io.StringIO()
        ConsoleReporter(decimals=4).render(rows, stream=buf)
        out = buf.getvalue()
        assert "only in SimaPro: 1" in out
        assert "only in EF31: 1" in out
        assert "aligned: 1" in out


class TestCompareConfig:
    def test_defaults(self) -> None:
        cfg = CompareConfig()
        assert cfg.decimals == 4
        assert cfg.method_filters == ()
        assert cfg.source == CompareConfig.SOURCE_REGISTRY


class TestCompareCfsCli:
    """End-to-end smoke tests against the real ``cache/``/``source/``/``registry/`` artifacts."""

    @staticmethod
    def _skip_if_artifacts_missing() -> None:
        from config import Settings

        paths = Settings().paths
        if (
            not paths.ef_cf_parquet.exists()
            or not paths.simapro_ef31_cache.exists()
            or not paths.registry_method_cfs_index.exists()
        ):
            pytest.skip("Real CF artifacts not present; skipping smoke test")

    def test_default_uses_joined_registry(self, capsys: pytest.CaptureFixture) -> None:
        self._skip_if_artifacts_missing()

        # --no-emit so the smoke test prints stats without rewriting the real
        # registry/dashboard artifacts (the emit path is covered separately).
        exit_code = CompareCfsCli().run(["--no-emit"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Acidification" in out
        assert "Climate change" in out
        assert "SimaPro" in out
        assert "EF31" in out
        # Joined-registry default emits 19 methods (matches what the scoring
        # pipeline uses; SimaPro inorganic/organic sub-methods fold into
        # their parent inside the SP index).
        assert "aligned: 19" in out
        assert "only in SimaPro: 0" in out
        assert "only in EF31: 0" in out

    def test_emit_writes_three_artifacts_under_settings_paths(self, tmp_path: Path) -> None:
        # Drives the CLI's emit wiring with synthetic frames + a tmp-rooted
        # Settings, so no real inputs are needed and nothing real is written.
        import numpy as np

        from config import Settings
        from config.paths import Paths
        from ef.cf_flow_join import ContextNormaliser, JoinedFlowFrame

        acid_key = ("ecoinvent-3.9.1", "EF v3.1", "acidification", "accumulated exceedance (AE)")
        frame = JoinedFlowFrame(
            method_key=acid_key,
            df=pd.DataFrame(
                [
                    {
                        "code": "c1",
                        "database": "ecoinvent-3.9.1-biosphere",
                        "name": "ammonia",
                        "categories": np.array(["air", "(unspecified)"], dtype=object),
                        "cas": "7664-41-7",
                        "sp_cf": 3.02,
                        "ef_cf": 3.02,
                        "sp_match_provenance": "exact_name",
                        "sp_name": "Ammonia",
                        "sp_compartment": "Air",
                        "sp_sub_compartment": "(unspecified)",
                    }
                ],
                columns=list(JoinedFlowFrame.COLUMNS),
            ),
        )
        normaliser = ContextNormaliser(
            rules=pd.DataFrame({"source_context": [], "target_context": []})
        )
        settings = Settings(paths=Paths(package_root=tmp_path))
        # No run_report under tmp → used-flow filter unavailable → falls back to
        # keeping all flows, so the artifact still writes.
        CompareCfsCli(settings=settings)._emit_dashboard_artifacts(
            [frame], normaliser, used_only=True
        )

        assert settings.paths.dashboard_cf_comparison_csv.exists()
        assert settings.paths.registry_cf_comparison_join.exists()
        assert settings.paths.registry_cf_comparison_by_code.exists()
        df = pd.read_csv(settings.paths.dashboard_cf_comparison_csv)
        assert {"compartment_simapro", "compartment_registry", "match_provenance"} <= set(
            df.columns
        )
        assert df.iloc[0]["name_simapro"] == "Ammonia"
        assert df.iloc[0]["status"] == "both_agree"

    def test_source_raw_uses_jrc_parquet(self, capsys: pytest.CaptureFixture) -> None:
        self._skip_if_artifacts_missing()

        exit_code = CompareCfsCli().run(["--source", "raw", "--no-emit"])

        assert exit_code == 0
        out = capsys.readouterr().out
        # Raw EF31 ships all 25 SimaPro method names verbatim (no alias needed).
        assert "aligned: 25" in out
        assert "only in SimaPro: 0" in out
        assert "only in EF31: 0" in out

    def test_filter_no_match_returns_nonzero(self) -> None:
        self._skip_if_artifacts_missing()

        exit_code = CompareCfsCli().run(["--method", "definitely-not-a-method"])
        assert exit_code != 0


class TestMethodAliasResolver:
    @pytest.mark.parametrize(
        "raw, expected_root",
        [
            ("Human toxicity, cancer", "Human toxicity, cancer"),
            ("Human toxicity, cancer - inorganics", "Human toxicity, cancer"),
            ("Human toxicity, cancer - organics", "Human toxicity, cancer"),
            ("Ecotoxicity, freshwater - inorganics", "Ecotoxicity, freshwater"),
            ("Acidification", "Acidification"),
        ],
    )
    def test_strip_submethod(self, raw: str, expected_root: str) -> None:
        assert MethodAliasResolver().strip_submethod(raw) == expected_root

    @pytest.mark.parametrize(
        "raw, expected_registry",
        [
            ("Resource use, fossils", "energy resources: non-renewable"),
            ("Resource use, minerals and metals", "material resources: metals/minerals"),
            ("Human toxicity, cancer", "human toxicity: carcinogenic"),
            ("Human toxicity, cancer - inorganics", "human toxicity: carcinogenic"),
            ("Particulate matter", "particulate matter formation"),
            ("Photochemical ozone formation", "photochemical oxidant formation: human health"),
            ("Ionising radiation", "ionising radiation: human health"),
            ("Acidification", "acidification"),
        ],
    )
    def test_simapro_to_registry(self, raw: str, expected_registry: str) -> None:
        assert MethodAliasResolver().simapro_to_registry(raw) == expected_registry

    def test_simapro_to_registry_unknown_returns_none(self) -> None:
        assert MethodAliasResolver().simapro_to_registry("not a real method") is None


class TestMethodCfsRegistryLoader:
    def test_load_concatenates_per_method_parquets(self, tmp_path: Path) -> None:
        # Build a tiny fake registry: 2 methods with 3 CFs each.
        import json as _json

        method_a_df = pd.DataFrame(
            {
                "database": ["bio", "bio", "bio"],
                "code": ["a1", "a2", "a3"],
                "amount": [1.0, 2.0, 3.0],
            }
        )
        method_b_df = pd.DataFrame(
            {
                "database": ["bio", "bio", "bio"],
                "code": ["b1", "b2", "b3"],
                "amount": [10.0, 20.0, 30.0],
            }
        )
        method_a_df.to_parquet(tmp_path / "slug_a.parquet")
        method_b_df.to_parquet(tmp_path / "slug_b.parquet")
        index = {
            "version": 1,
            "methods": [
                {"key": ["bio", "EF v3.1", "method a", "indicator a"], "slug": "slug_a.parquet"},
                {"key": ["bio", "EF v3.1", "method b", "indicator b"], "slug": "slug_b.parquet"},
            ],
        }
        (tmp_path / "_index.json").write_text(_json.dumps(index))

        out = MethodCfsRegistryLoader(tmp_path).load()
        assert list(out.columns) == ["method_raw", "cf"]
        assert len(out) == 6
        assert set(out["method_raw"]) == {"method a", "method b"}
        assert out.loc[out["method_raw"] == "method a", "cf"].sum() == 6.0
        assert out.loc[out["method_raw"] == "method b", "cf"].sum() == 60.0

    def test_load_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MethodCfsRegistryLoader(tmp_path / "does-not-exist").load()

    def test_load_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MethodCfsRegistryLoader(tmp_path).load()

    def test_load_empty_index_returns_empty_frame(self, tmp_path: Path) -> None:
        import json as _json

        (tmp_path / "_index.json").write_text(_json.dumps({"version": 1, "methods": []}))
        out = MethodCfsRegistryLoader(tmp_path).load()
        assert list(out.columns) == ["method_raw", "cf"]
        assert len(out) == 0


class TestCfComparatorWithAliasResolver:
    """Exercises the resolver path: SimaPro names map to registry keys; sub-methods fold."""

    def _stub(self, df: pd.DataFrame) -> object:
        class _StubLoader:
            def load(self_inner) -> pd.DataFrame:
                return df

        return _StubLoader()

    def test_sub_methods_fold_into_root(self) -> None:
        sp = pd.DataFrame(
            {
                "method_raw": [
                    "Human toxicity, cancer",
                    "Human toxicity, cancer - inorganics",
                    "Human toxicity, cancer - organics",
                ],
                "cf": [1.0, 2.0, 3.0],
            }
        )
        registry = pd.DataFrame({"method_raw": ["human toxicity: carcinogenic"], "cf": [99.0]})
        comparator = CfComparator(
            simapro_loader=self._stub(sp),
            ef31_loader=self._stub(registry),
            method_filters=(),
            alias_resolver=MethodAliasResolver(),
        )
        rows = comparator.build_rows()
        assert len(rows) == 1
        row = rows[0]
        # Sub-methods are folded — SimaPro side has all 3 CFs aggregated.
        assert row.simapro_stats["count"] == 3
        assert row.simapro_stats["sum"] == 6.0
        # Display name is the root, not a sub-method.
        assert row.display_name == "Human toxicity, cancer"

    def test_unaliased_simapro_methods_dropped(self) -> None:
        sp = pd.DataFrame({"method_raw": ["Made-up method"], "cf": [1.0]})
        registry = pd.DataFrame({"method_raw": ["acidification"], "cf": [1.0]})
        comparator = CfComparator(
            simapro_loader=self._stub(sp),
            ef31_loader=self._stub(registry),
            method_filters=(),
            alias_resolver=MethodAliasResolver(),
        )
        rows = {row.norm_key: row for row in comparator.build_rows()}
        # The unaliased SimaPro method is dropped (no registry equivalent).
        assert "acidification" in rows
        assert rows["acidification"].simapro_stats is None
        assert all("made-up" not in k for k in rows)

    def test_registry_only_methods_render_with_lowercase_display(self) -> None:
        sp = pd.DataFrame({"method_raw": [], "cf": []})
        registry = pd.DataFrame({"method_raw": ["acidification"], "cf": [1.0]})
        comparator = CfComparator(
            simapro_loader=self._stub(sp),
            ef31_loader=self._stub(registry),
            method_filters=(),
            alias_resolver=MethodAliasResolver(),
        )
        rows = comparator.build_rows()
        assert len(rows) == 1
        assert rows[0].display_name == "acidification"
        assert rows[0].simapro_stats is None
        assert rows[0].ef31_stats is not None


class TestCompareConfig_Source:
    """Default and validity of the --source option."""

    def test_default_source_is_registry(self) -> None:
        assert CompareConfig().source == CompareConfig.SOURCE_REGISTRY

    def test_valid_sources_includes_both(self) -> None:
        assert set(CompareConfig.VALID_SOURCES) == {
            CompareConfig.SOURCE_REGISTRY,
            CompareConfig.SOURCE_RAW,
        }
