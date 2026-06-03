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
# CfStatsEmitter

from cli.compare_cfs import MethodRow
from reporting import CfStatsEmitter


class TestCfStatsEmitter:
    HEADER = (
        "method,alignment,"
        "sp_count,ef_count,"
        "sp_min,ef_min,sp_max,ef_max,"
        "sp_mean,ef_mean,sp_median,ef_median,"
        "sp_std,ef_std,sp_sum,ef_sum,"
        "diff_count,diff_min,diff_max,diff_mean,diff_median,diff_std,diff_sum"
    )

    @staticmethod
    def _stats(**overrides):
        base = {
            "count": 100,
            "min": 1.0,
            "max": 10.0,
            "mean": 5.0,
            "median": 5.0,
            "std": 2.0,
            "sum": 500.0,
        }
        base.update(overrides)
        return base

    def _row(self, name, sp=None, ef=None):
        return MethodRow(
            display_name=name,
            norm_key=name.lower(),
            simapro_stats=sp,
            ef31_stats=ef,
        )

    def test_writes_header_in_documented_order(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        CfStatsEmitter(out_path=out).write([])
        first_line = out.read_text().splitlines()[0]
        assert first_line == self.HEADER

    def test_both_sides_emits_symmetric_diff_fractions(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        sp = self._stats(mean=5.0, sum=500.0)
        ef = self._stats(mean=6.0, sum=550.0)
        CfStatsEmitter(out_path=out).write([self._row("Acidification", sp, ef)])
        df = pd.read_csv(out)
        row = df.iloc[0]
        assert row["alignment"] == "both"
        assert row["sp_mean"] == 5.0
        assert row["ef_mean"] == 6.0
        # diff = (ef - sp) / max(|sp|, |ef|), bounded in [-1, +1]
        assert row["diff_mean"] == pytest.approx((6.0 - 5.0) / 6.0)
        assert row["diff_sum"] == pytest.approx((550.0 - 500.0) / 550.0)

    def test_sp_only_leaves_ef_and_diff_blank(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        CfStatsEmitter(out_path=out).write([self._row("Land use", self._stats(), None)])
        df = pd.read_csv(out)
        row = df.iloc[0]
        assert row["alignment"] == "sp_only"
        assert row["sp_count"] == 100
        assert pd.isna(row["ef_count"])
        assert pd.isna(row["diff_count"])
        assert pd.isna(row["diff_mean"])

    def test_ef_only_leaves_sp_and_diff_blank(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        CfStatsEmitter(out_path=out).write([self._row("Resource use", None, self._stats())])
        df = pd.read_csv(out)
        row = df.iloc[0]
        assert row["alignment"] == "ef_only"
        assert pd.isna(row["sp_count"])
        assert row["ef_count"] == 100
        assert pd.isna(row["diff_count"])

    def test_sp_zero_with_nonzero_ef_saturates_to_plus_one(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        sp = self._stats(min=0.0, mean=5.0)
        ef = self._stats(min=0.1, mean=6.0)
        CfStatsEmitter(out_path=out).write([self._row("Acidification", sp, ef)])
        df = pd.read_csv(out)
        row = df.iloc[0]
        # Symmetric formula: sp=0, ef=0.1 -> +1.0 (saturated), not blank.
        assert row["diff_min"] == pytest.approx(1.0)
        assert row["diff_mean"] == pytest.approx((6.0 - 5.0) / 6.0)
        assert row["sp_min"] == 0.0
        assert row["ef_min"] == 0.1

    def test_both_zero_yields_zero_diff(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        sp = self._stats(min=0.0, mean=0.0)
        ef = self._stats(min=0.0, mean=0.0)
        CfStatsEmitter(out_path=out).write([self._row("Climate change - Biogenic", sp, ef)])
        df = pd.read_csv(out)
        row = df.iloc[0]
        assert row["diff_min"] == 0.0
        assert row["diff_mean"] == 0.0

    def test_diff_is_bounded_within_pm_unit_when_signs_match(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        # Historically (ef - sp) / |sp| produced +9900% on Ozone depletion
        # median (sp=0.005, ef=0.5). The symmetric form bounds to [-1, +1]
        # whenever sp and ef share a sign.
        sp = self._stats(median=0.005)
        ef = self._stats(median=0.5)
        CfStatsEmitter(out_path=out).write([self._row("Ozone depletion", sp, ef)])
        df = pd.read_csv(out)
        row = df.iloc[0]
        assert -1.0 <= row["diff_median"] <= 1.0
        assert row["diff_median"] == pytest.approx((0.5 - 0.005) / 0.5)

    def test_diff_sign_flip_can_exceed_unit(self, tmp_path: Path) -> None:
        out = tmp_path / "cf_stats.csv"
        # When sp and ef have opposite signs the symmetric formula can
        # produce |diff| > 1 (max |diff| = 2 at sp = -ef). Real example:
        # Water use joined median sp=-4.575, ef=+4.59 → +1.997.
        sp = self._stats(median=-4.575)
        ef = self._stats(median=4.59)
        CfStatsEmitter(out_path=out).write([self._row("Water use", sp, ef)])
        df = pd.read_csv(out)
        row = df.iloc[0]
        expected = (4.59 - (-4.575)) / max(abs(-4.575), abs(4.59))
        assert row["diff_median"] == pytest.approx(expected)
        assert row["diff_median"] > 1.0  # confirms sign-flip signal

    def test_per_flow_diff_stats_override_diff_columns(self, tmp_path: Path) -> None:
        # In joined-mode, the comparator computes per-flow diff stats
        # separately so diff_<stat> reads as "<stat> of per-flow diffs"
        # instead of "(ef_<stat> - sp_<stat>) / max(|sp|, |ef|)".
        # Water-use case: with 2 matched flows where positives and
        # negatives cancel, the "diff of means" formula reads -100%
        # while the per-flow mean diff is -6%. The emitter must trust
        # per_flow_diff_stats when present.
        out = tmp_path / "cf_stats.csv"
        sp = self._stats(mean=-0.0025, median=-0.0025)
        ef = self._stats(mean=-2.575, median=-2.575)
        per_flow = {
            "count": 2,
            "min": -0.12,
            "max": 0.0,
            "mean": -0.06,
            "median": -0.06,
            "std": 0.085,
            "sum": -0.12,
        }
        row = MethodRow(
            display_name="Water use",
            norm_key="water use",
            simapro_stats=sp,
            ef31_stats=ef,
            per_flow_diff_stats=per_flow,
        )
        CfStatsEmitter(out_path=out).write([row])
        df = pd.read_csv(out)
        r = df.iloc[0]
        # Per-flow values used for min/max/mean/median/std
        assert r["diff_mean"] == pytest.approx(-0.06)
        assert r["diff_median"] == pytest.approx(-0.06)
        assert r["diff_min"] == pytest.approx(-0.12)
        assert r["diff_max"] == pytest.approx(0.0)
        # count and sum keep the historical formula
        # (sp_sum=500, ef_sum=500 in the default _stats() → diff_sum = 0)
        assert r["diff_count"] == pytest.approx(0.0)
        assert r["diff_sum"] == pytest.approx(0.0)

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "cf_stats.csv"
        CfStatsEmitter(out_path=out).write([])
        assert out.exists()
