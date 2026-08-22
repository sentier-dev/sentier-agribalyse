"""Unit tests for ``reporting/`` — RunReport, CoverageSnapshot, UnlinkedExporter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reporting import (
    BacktestPass1Emitter,
    CoverageReporter,
    CoverageSnapshot,
    NearZeroFloor,
    RunReport,
    UnlinkedExporter,
)
from tests.fixtures.builders import make_dataset, make_exchange

# ============================================================================
# CoverageSnapshot + Reporter


class TestCoverageSnapshot:
    def test_rates_round_to_two_decimals(self):
        snap = CoverageSnapshot(label="x", bio_total=3, bio_linked=1, tech_total=4, tech_linked=1)
        assert snap.bio_rate == pytest.approx(33.33)
        assert snap.tech_rate == pytest.approx(25.0)

    def test_zero_total_does_not_divide_by_zero(self):
        snap = CoverageSnapshot(label="x", bio_total=0, bio_linked=0, tech_total=0, tech_linked=0)
        assert snap.bio_rate == 0.0
        assert snap.tech_rate == 0.0

    def test_as_dict_contains_label_and_rates(self):
        snap = CoverageSnapshot(
            label="pre", bio_total=10, bio_linked=8, tech_total=4, tech_linked=4
        )
        d = snap.as_dict()
        assert d["label"] == "pre"
        assert d["bio_rate"] == 80.0
        assert d["tech_rate"] == 100.0

    def test_from_sp_data_counts_linked_versus_total(self):
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(type="biosphere", name="a", input=("db", "x")),
                    make_exchange(type="biosphere", name="b"),
                    make_exchange(type="technosphere", name="t1", input=("db", "x")),
                ],
            )
        ]
        snap = CoverageReporter.from_sp_data("test", sp_data)
        assert snap.bio_total == 2
        assert snap.bio_linked == 1
        assert snap.tech_total == 1
        assert snap.tech_linked == 1


# ============================================================================
# RunReport


class TestRunReport:
    def test_add_stage_and_as_dict(self):
        rep = RunReport()
        rep.add_stage("link.biosphere", {"linked": 5})
        rep.add_stage("link.technosphere", {"linked": 3})
        d = rep.as_dict()
        assert d["stages"]["link.biosphere"] == {"linked": 5}
        assert d["stages"]["link.technosphere"] == {"linked": 3}

    def test_add_coverage_serialises_snapshot(self):
        rep = RunReport()
        snap = CoverageSnapshot(label="pre", bio_total=2, bio_linked=2, tech_total=1, tech_linked=0)
        rep.add_coverage(snap)
        assert rep.coverage[0]["label"] == "pre"

    def test_set_matrix_shape_records_deficit_and_square(self):
        rep = RunReport()
        rep.set_matrix_shape(rows=10, cols=8)
        assert rep.matrix_shape == {"rows": 10, "cols": 8, "deficit": 2, "square": False}

        rep.set_matrix_shape(rows=10, cols=10)
        assert rep.matrix_shape["square"] is True

    def test_write_emits_json_with_all_sections(self, tmp_path: Path):
        rep = RunReport()
        rep.add_stage("foo", {"a": 1})
        rep.set_drops({"x": 3})
        rep.set_suppressed({"strategy": 1})
        rep.set_matrix_shape(2, 2)

        path = rep.write(tmp_path / "deep" / "report.json")
        assert path.exists()
        data = json.loads(path.read_text())
        for section in (
            "timestamp",
            "stages",
            "coverage",
            "drops_by_strategy",
            "suppressed_strategies",
            "matrix_shape",
        ):
            assert section in data


# ============================================================================
# UnlinkedExporter


class TestUnlinkedExporter:
    def test_exports_unlinked_tech_json_and_bio_xlsx(self, settings):
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="UnknownBio",
                        unit="kg",
                        amount=2.0,
                        categories=("air",),
                    ),
                    make_exchange(
                        type="technosphere",
                        name="UnknownTech",
                        unit="kg",
                        amount=1.0,
                    ),
                    # Already-linked exchange must not appear in the unlinked export.
                    make_exchange(
                        type="biosphere",
                        name="LinkedBio",
                        unit="kg",
                        amount=1.0,
                        input=("biosphere3", "ok"),
                    ),
                ],
            )
        ]
        result = UnlinkedExporter(settings=settings).export(sp_data)
        # Counters returned by the exporter.
        assert result["bio_unique"] == 1
        assert result["tech_unique"] == 1
        # Files were written.
        unlinked_dir = settings.paths.unlinked
        assert (unlinked_dir / "technosphere_unlinked.json").exists()
        assert (unlinked_dir / "biosphere_unlinked.xlsx").exists()

    def test_unlink_reason_no_registry_returns_empty(self, settings):
        exp = UnlinkedExporter(settings=settings, registry=None)
        reason, prov = exp._unlink_reason("Methane", "air", "")
        assert reason == ""
        assert prov == ""

    def test_unlink_reason_flow_found_with_notes(self, settings):
        df = pd.DataFrame(
            [
                {
                    "source_name": "Methane",
                    "source_top_bucket": "air",
                    "notes": "No equivalent in EF",
                    "provenance": "curated_overrides",
                }
            ]
        )
        registry = SimpleNamespace(unmatchable_index=SimpleNamespace(df=df))
        exp = UnlinkedExporter(settings=settings, registry=registry)
        reason, prov = exp._unlink_reason("Methane", "air", "")
        assert reason == "No equivalent in EF"
        assert prov == "curated_overrides"

    def test_unlink_reason_placeholder_provenance_synthesised(self, settings):
        df = pd.DataFrame(
            [
                {
                    "source_name": "Bifenazate",
                    "source_top_bucket": "air",
                    "notes": "",
                    "provenance": "placeholder.neither",
                }
            ]
        )
        registry = SimpleNamespace(unmatchable_index=SimpleNamespace(df=df))
        exp = UnlinkedExporter(settings=settings, registry=registry)
        reason, _ = exp._unlink_reason("Bifenazate", "air", "")
        assert "ecoinvent" in reason and "EF" in reason

    def test_unlink_reason_flow_not_in_index(self, settings):
        registry = SimpleNamespace(
            unmatchable_index=SimpleNamespace(
                df=pd.DataFrame(columns=["source_name", "source_top_bucket", "notes", "provenance"])
            )
        )
        exp = UnlinkedExporter(settings=settings, registry=registry)
        reason, prov = exp._unlink_reason("Unknown", "air", "")
        assert "No matching" in reason
        assert prov == ""

    def test_export_with_registry_writes_unlink_reason_column(self, settings):
        df = pd.DataFrame(
            [
                {
                    "source_name": "UnknownBio",
                    "source_top_bucket": "air",
                    "notes": "no equivalent",
                    "provenance": "curated_overrides",
                }
            ]
        )
        registry = SimpleNamespace(unmatchable_index=SimpleNamespace(df=df))
        sp_data = [
            make_dataset(
                "P",
                exchanges=[
                    make_exchange(
                        type="biosphere",
                        name="UnknownBio",
                        unit="kg",
                        amount=1.0,
                        categories=("air",),
                    )
                ],
            )
        ]
        UnlinkedExporter(settings=settings, registry=registry).export(sp_data)
        out = pd.read_excel(settings.paths.unlinked / "biosphere_unlinked.xlsx")
        assert "unlink_reason" in out.columns
        assert out.iloc[0]["unlink_reason"] == "no equivalent"


# ============================================================================
# NearZeroFloor


class TestNearZeroFloor:
    @staticmethod
    def _scores(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_threshold_is_factor_times_median_abs_reference(self):
        df = self._scores(
            [
                {"computed_water": 1.0, "reference_water": 10.0},
                {"computed_water": 2.0, "reference_water": 100.0},
                {"computed_water": 3.0, "reference_water": 1000.0},
            ]
        )
        floor = NearZeroFloor.compute(df, ["water"], factor=0.01)
        # median of [10, 100, 1000] = 100, 1% = 1.0
        assert floor.thresholds["water"] == pytest.approx(1.0)
        assert floor.factor == 0.01

    def test_zero_when_both_below_threshold(self):
        df = self._scores(
            [
                # Reference median = 100 → threshold = 1.0.
                {"computed_water": 5.0, "reference_water": 10.0},
                {"computed_water": 50.0, "reference_water": 100.0},
                {"computed_water": 500.0, "reference_water": 1000.0},
                # Near-zero pair → must be floored.
                {"computed_water": 1e-9, "reference_water": -1e-9},
                # Computed above, reference below → must NOT be floored.
                {"computed_water": 50.0, "reference_water": 1e-9},
            ]
        )
        diff_abs = {"water": df["computed_water"] - df["reference_water"]}
        diff_pct = {
            "water": (df["computed_water"] - df["reference_water"]) / df["reference_water"] * 100
        }
        floor = NearZeroFloor.compute(df, ["water"], factor=0.01)
        counts = floor.apply(df, diff_abs, diff_pct)

        assert counts["water"] == 1
        # Row 3 zeroed in all four series.
        assert df.loc[3, "computed_water"] == 0.0
        assert df.loc[3, "reference_water"] == 0.0
        assert diff_abs["water"].iloc[3] == 0.0
        assert diff_pct["water"].iloc[3] == 0.0
        # Row 4 (asymmetric) untouched.
        assert df.loc[4, "computed_water"] == 50.0
        assert df.loc[4, "reference_water"] == 1e-9
        # Normal rows untouched.
        assert df.loc[0, "computed_water"] == 5.0

    def test_missing_method_columns_emit_zero_threshold_and_zero_count(self):
        df = self._scores([{"computed_water": 1.0, "reference_water": 1.0}])
        floor = NearZeroFloor.compute(df, ["missing"], factor=0.01)
        assert floor.thresholds["missing"] == 0.0
        diff_abs: dict[str, pd.Series] = {}
        diff_pct: dict[str, pd.Series] = {}
        counts = floor.apply(df, diff_abs, diff_pct)
        assert counts["missing"] == 0

    def test_all_zero_reference_yields_zero_threshold(self):
        df = self._scores(
            [
                {"computed_water": 1.0, "reference_water": 0.0},
                {"computed_water": 2.0, "reference_water": 0.0},
            ]
        )
        floor = NearZeroFloor.compute(df, ["water"], factor=0.01)
        assert floor.thresholds["water"] == 0.0
        diff_abs = {"water": pd.Series([1.0, 2.0])}
        diff_pct = {"water": pd.Series([float("inf"), float("inf")])}
        counts = floor.apply(df, diff_abs, diff_pct)
        # No flooring when threshold is 0 — preserves the data.
        assert counts["water"] == 0
        assert df.loc[0, "computed_water"] == 1.0


# ============================================================================
# BacktestPass1Emitter


class TestBacktestPass1Emitter:
    def test_writes_dashboard_csv_with_short_ids_and_mapped_filter(self, tmp_path: Path):
        scores_df = pd.DataFrame(
            [
                {
                    "Code AGB": "1",
                    "Nom du Produit": "Beetroot juice",
                    "LCI Name": "Beetroot juice, pure",
                    "mapped": True,
                },
                {
                    "Code AGB": "2",
                    "Nom du Produit": "Skipped",
                    "LCI Name": "n/a",
                    "mapped": False,
                },
            ]
        )
        diff_pct_df = pd.DataFrame(
            [
                {
                    "Code AGB": "1",
                    "Nom du Produit": "Beetroot juice",
                    "LCI Name": "Beetroot juice, pure",
                    "climate change": -19.5,
                    "climate change: biogenic": -98.0,
                    "ecotoxicity: freshwater": -33.0,
                },
                {
                    "Code AGB": "2",
                    "Nom du Produit": "Skipped",
                    "LCI Name": "n/a",
                    "climate change": float("nan"),
                    "climate change: biogenic": float("nan"),
                    "ecotoxicity: freshwater": float("nan"),
                },
            ]
        )
        out = tmp_path / "backtest_pass1.csv"
        BacktestPass1Emitter(out_path=out).write(scores_df, diff_pct_df)

        assert out.exists()
        result = pd.read_csv(out)
        # Header has 5 metadata cols + 19 method short IDs in fixed order.
        assert list(result.columns)[:5] == ["code", "name", "mapped_to", "type", "resolution"]
        assert list(result.columns)[5:] == list(BacktestPass1Emitter.SHORT_ORDER)
        # Only the mapped row is emitted.
        assert len(result) == 1
        row = result.iloc[0]
        assert row["code"] == 1
        assert row["name"] == "Beetroot juice"
        assert row["mapped_to"] == "Beetroot juice, pure"
        assert row["resolution"] == "mapped"
        assert row["climate"] == pytest.approx(-19.5)
        assert row["cc_bio"] == pytest.approx(-98.0)
        assert row["ecotox"] == pytest.approx(-33.0)

    def test_missing_method_columns_render_blank(self, tmp_path: Path):
        scores_df = pd.DataFrame(
            [{"Code AGB": "1", "Nom du Produit": "X", "LCI Name": "x", "mapped": True}]
        )
        # diff_pct_df has only the climate column out of the 19 expected.
        diff_pct_df = pd.DataFrame(
            [
                {
                    "Code AGB": "1",
                    "Nom du Produit": "X",
                    "LCI Name": "x",
                    "climate change": 1.0,
                }
            ]
        )
        out = tmp_path / "backtest_pass1.csv"
        BacktestPass1Emitter(out_path=out).write(scores_df, diff_pct_df)
        result = pd.read_csv(out)
        assert result.iloc[0]["climate"] == pytest.approx(1.0)
        # Other method columns present but empty.
        assert pd.isna(result.iloc[0]["cc_bio"])


# ============================================================================
# CF comparison: join builder, CSV emitter, by-code sidecar

import numpy as np

from ef.cf_flow_join import ContextNormaliser, JoinedFlowFrame
from reporting import (
    CfComparisonByCodeBuilder,
    CfComparisonCsvEmitter,
    CfComparisonJoinBuilder,
    CfComparisonJoinLoader,
)

_ACID_KEY = ("ecoinvent-3.9.1", "EF v3.1", "acidification", "accumulated exceedance (AE)")
_WATER_KEY = (
    "ecoinvent-3.9.1",
    "EF v3.1",
    "water use",
    "user deprivation potential (deprivation-weighted water consumption)",
)


def _empty_normaliser() -> ContextNormaliser:
    # No rules needed: the builder only uses the built-in compartment buckets
    # (ecoinvent top → Air/Water/Soil/Raw) and the JRC EF category table.
    return ContextNormaliser(rules=pd.DataFrame({"source_context": [], "target_context": []}))


def _joined_frame(method_key, rows: list[dict]) -> JoinedFlowFrame:
    return JoinedFlowFrame(
        method_key=method_key,
        df=pd.DataFrame(rows, columns=list(JoinedFlowFrame.COLUMNS)),
    )


def _matched_row(**kw) -> dict:
    base = {
        "code": "c1",
        "database": "ecoinvent-3.9.1-biosphere",
        "name": "nitrogen dioxide",
        "categories": np.array(["air", "urban air close to ground"], dtype=object),
        "sp_cf": 0.74,
        "ef_cf": 0.74,
        "sp_match_provenance": "exact_name",
        "sp_name": "Nitrogen dioxide",
        "sp_compartment": "Air",
        "sp_sub_compartment": "high. pop.",
    }
    base.update(kw)
    return base


def _registry_only_row(**kw) -> dict:
    base = {
        "code": "c9",
        "database": "ef",
        "name": "some ecoinvent organic",
        "categories": np.array(["water", "ground-"], dtype=object),
        "sp_cf": np.nan,
        "ef_cf": 5.0,
        "sp_match_provenance": "unmatched",
        "sp_name": "",
        "sp_compartment": "",
        "sp_sub_compartment": "",
    }
    base.update(kw)
    return base


class TestCfComparisonJoinBuilder:
    def test_matched_row_computes_status_compartments_and_metrics(self) -> None:
        frame = _joined_frame(_ACID_KEY, [_matched_row(sp_cf=0.80, ef_cf=0.74)])
        df = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build([frame])
        row = df.iloc[0]
        assert row["method"] == "acidification"
        assert row["compartment"] == "air"  # ecoinvent top "air" → bucket
        assert row["compartment_registry"] == "air / urban air close to ground"
        assert row["compartment_simapro"] == "Air / high. pop."
        assert row["name_simapro"] == "Nitrogen dioxide"
        assert row["status"] == "both_differ"
        assert row["match_provenance"] == "exact_name"
        assert row["sp_reg_ratio"] == pytest.approx(0.80 / 0.74)
        assert row["rel_diff"] == pytest.approx(abs(0.80 - 0.74) / 0.74)

    def test_equal_cfs_are_both_agree(self) -> None:
        frame = _joined_frame(_ACID_KEY, [_matched_row(sp_cf=0.74, ef_cf=0.74)])
        df = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build([frame])
        assert df.iloc[0]["status"] == "both_agree"

    def test_tiny_cfs_with_large_relative_gap_are_both_differ(self) -> None:
        # Both CFs are far below the 1e-9 absolute floor (routine for
        # toxicity methods), but SimaPro is ~1.7x the registry value. The
        # *relative* gap must drive the verdict — gating the reference on
        # AGREE_ABS used to let this 70% disagreement read as "agree".
        frame = _joined_frame(_ACID_KEY, [_matched_row(sp_cf=1.4735e-9, ef_cf=8.6876e-10)])
        row = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build([frame]).iloc[0]
        assert row["status"] == "both_differ"
        assert row["rel_diff"] == pytest.approx(abs(1.4735e-9 - 8.6876e-10) / 8.6876e-10)

    def test_equal_tiny_cfs_are_both_agree(self) -> None:
        # A genuinely tiny but matching CF stays "agree" (relative gap ~0).
        frame = _joined_frame(_ACID_KEY, [_matched_row(sp_cf=8.6876e-10, ef_cf=8.6876e-10)])
        row = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build([frame]).iloc[0]
        assert row["status"] == "both_agree"
        assert row["rel_diff"] == pytest.approx(0.0)

    def test_both_cfs_zero_are_both_agree(self) -> None:
        # Registry CF is exactly zero: no usable reference, but SimaPro is
        # also zero, so the pair agrees via the absolute floor.
        frame = _joined_frame(_ACID_KEY, [_matched_row(sp_cf=0.0, ef_cf=0.0)])
        row = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build([frame]).iloc[0]
        assert row["status"] == "both_agree"

    def test_unspecified_subcompartment_collapses(self) -> None:
        frame = _joined_frame(
            _ACID_KEY,
            [
                _matched_row(
                    categories=np.array(["air", "(unspecified)"], dtype=object),
                    sp_compartment="Air",
                    sp_sub_compartment="(unspecified)",
                )
            ],
        )
        df = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build([frame])
        assert df.iloc[0]["compartment_registry"] == "air"
        assert df.iloc[0]["compartment_simapro"] == "Air"

    def test_registry_only_row_has_blank_simapro_side(self) -> None:
        frame = _joined_frame(_WATER_KEY, [_registry_only_row()])
        df = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build([frame])
        row = df.iloc[0]
        assert row["status"] == "registry_only"
        assert pd.isna(row["cf_simapro"])
        assert row["name_simapro"] is None
        assert row["compartment_simapro"] == ""
        assert row["compartment"] == "water"
        assert row["cf_registry"] == 5.0


class TestUsedFlowFilter:
    def test_keeps_only_flows_whose_flow_id_is_in_the_package(self) -> None:
        from reporting import UsedFlowFilter
        from scoring.exchange_frame_builder import ExchangeFrameBuilder

        frames = [
            _joined_frame(
                _ACID_KEY,
                [
                    _matched_row(code="used1", database="ecoinvent-3.9.1-biosphere"),
                    _matched_row(code="unused1", database="ef"),
                ],
            )
        ]
        join = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build(frames)
        used_id = ExchangeFrameBuilder.flow_id_for(("ecoinvent-3.9.1-biosphere", "used1"))
        flt = UsedFlowFilter.from_biosphere_row_ids({str(used_id): 0})
        out = flt.filter(join)
        assert list(out["code"]) == ["used1"]

    def test_empty_package_keeps_nothing(self) -> None:
        from reporting import UsedFlowFilter

        frames = [_joined_frame(_ACID_KEY, [_matched_row()])]
        join = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build(frames)
        assert len(UsedFlowFilter.from_biosphere_row_ids({}).filter(join)) == 0


class TestCfComparisonCsvEmitter:
    @staticmethod
    def _join() -> pd.DataFrame:
        frames = [
            _joined_frame(_ACID_KEY, [_matched_row()]),
            _joined_frame(_WATER_KEY, [_registry_only_row()]),
        ]
        return CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build(frames)

    def test_writes_columns_in_comparison_first_order(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_comparison.csv"
        CfComparisonCsvEmitter(out_path=out).write(self._join())
        header = out.read_text().splitlines()[0].split(",")
        assert header == list(CfComparisonCsvEmitter.COLUMNS)
        # name/cf SimaPro fields sit immediately beside the registry ones.
        assert header.index("name_registry") == header.index("name_simapro") + 1
        assert header.index("cf_registry") == header.index("cf_simapro") + 1
        assert "compartment_simapro" in header and "compartment_registry" in header
        assert "match_provenance" in header
        # No candidate-era columns survive.
        assert "n_candidates" not in header
        assert "candidate_cfs" not in header

    def test_matched_only_by_default_drops_registry_only(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_comparison.csv"
        CfComparisonCsvEmitter(out_path=out).write(self._join())
        df = pd.read_csv(out)
        assert set(df["status"]) == {"both_agree"}
        row = df.iloc[0]
        assert row["name_simapro"] == "Nitrogen dioxide"
        assert row["compartment_simapro"] == "Air / high. pop."
        assert row["match_provenance"] == "exact_name"

    def test_full_mode_keeps_registry_only(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_comparison.csv"
        CfComparisonCsvEmitter(out_path=out, matched_only=False).write(self._join())
        df = pd.read_csv(out)
        assert "registry_only" in set(df["status"])

    def test_raises_on_missing_expected_column(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_comparison.csv"
        df = self._join().drop(columns=["cf_simapro"])
        with pytest.raises(ValueError, match="cf_simapro"):
            CfComparisonCsvEmitter(out_path=out).write(df)

    def test_loader_raises_when_parquet_absent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="CF comparison join"):
            CfComparisonJoinLoader(tmp_path / "nope.parquet").load()


class TestCfComparisonByCodeBuilder:
    def test_emits_matched_rows_only_in_lookup_schema(self) -> None:
        frames = [
            _joined_frame(_ACID_KEY, [_matched_row(code="c1"), _registry_only_row(code="c2")]),
        ]
        df = CfComparisonByCodeBuilder().build(frames)
        assert list(df.columns) == list(CfComparisonByCodeBuilder.COLUMNS)
        assert list(df["code"]) == ["c1"]
        row = df.iloc[0]
        assert row["method"] == "acidification"
        assert row["cf_simapro"] == 0.74
        assert row["match_basis"] == "exact_name"
        assert row["name_simapro"] == "Nitrogen dioxide"
        assert row["name_registry"] == "nitrogen dioxide"

    def test_dedups_on_method_and_code(self) -> None:
        frames = [
            _joined_frame(_ACID_KEY, [_matched_row(code="c1"), _matched_row(code="c1", sp_cf=9.9)]),
        ]
        df = CfComparisonByCodeBuilder().build(frames)
        assert len(df) == 1
        assert df.iloc[0]["cf_simapro"] == 0.74  # first wins

    def test_sidecar_is_consumable_by_simapro_cf_lookup(self) -> None:
        from reporting.simapro_cf_lookup import SimaProCfLookup

        frames = [_joined_frame(_ACID_KEY, [_matched_row(code="c1")])]
        sidecar = CfComparisonByCodeBuilder().build(frames)
        lookup = SimaProCfLookup.from_dataframe(sidecar)
        entry = lookup.get("c1", "acidification")
        assert entry is not None
        assert entry.sp_cf == 0.74
        assert entry.provenance == "exact_name"
        assert entry.sp_name == "Nitrogen dioxide"

    def test_loader_round_trips_parquet(self, tmp_path: Path) -> None:
        frames = [_joined_frame(_ACID_KEY, [_matched_row(), _registry_only_row()])]
        join = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build(frames)
        src = tmp_path / "cf_comparison_join.parquet"
        join.to_parquet(src, index=False)
        loaded = CfComparisonJoinLoader(src).load()
        assert set(CfComparisonCsvEmitter.COLUMNS).issubset(loaded.columns)
        assert len(loaded) == 2

    def test_emitter_creates_parent_dir(self, tmp_path: Path) -> None:
        frames = [_joined_frame(_ACID_KEY, [_matched_row()])]
        join = CfComparisonJoinBuilder(normaliser=_empty_normaliser()).build(frames)
        out = tmp_path / "nested" / "dir" / "cf_comparison.csv"
        CfComparisonCsvEmitter(out_path=out).write(join)
        assert out.exists()
