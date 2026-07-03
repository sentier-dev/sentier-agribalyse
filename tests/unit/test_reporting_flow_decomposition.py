"""Unit tests for ``FlowDecompositionEmitter``.

The emitter is a pure (decomposer + scores_df → JSON files) function
with no external side effects beyond the output directory. Tests build
a tiny synthetic ``ScoringPackage`` + ``ProductCatalog`` + biosphere
catalog (same pattern as ``test_cli_decompose_score.py``) and verify
each invariant the dashboard relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cli.decompose_score import MethodAliases
from reporting.flow_decomposition import FlowDecompositionEmitter
from reporting.simapro_cf_lookup import SimaProCfLookup
from scoring.decomposer import ScoreDecomposer
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.product_catalog import ProductCatalog
from scoring.scoring_package import ScoringPackageBuilder


def _simapro_lookup() -> SimaProCfLookup:
    """SimaPro CFs for the tiny world: CO2 matches (sp=1.1), CH4 has no
    SimaPro match under climate; ozone has no SimaPro coverage at all."""
    climate = MethodAliases.resolve("climate")
    return SimaProCfLookup.from_dataframe(
        pd.DataFrame(
            [
                {
                    "code": "co2",
                    "method": climate[2],
                    "cf_simapro": 1.1,
                    "match_basis": "code",
                },
                {
                    "code": "ch4",
                    "method": climate[2],
                    "cf_simapro": None,
                    "match_basis": None,
                },
            ]
        )
    )


def _build_package_and_catalog(tmp_path: Path):
    """Tiny world: ``wheat`` emits 2 kg CO2 and 0.5 kg CH4.

    Climate CF on CO2=1, CH4=30. Ozone CF only on CH4=0.001. Total
    climate score: 2*1 + 0.5*30 = 17. Total ozone score: 0.5*0.001 = 5e-4.
    """
    sp_data = [
        {
            "database": "agb",
            "code": "wheat",
            "name": "wheat",
            "exchanges": [
                {"type": "production", "input": ("agb", "wheat-prod"), "amount": 1.0},
                {"type": "biosphere", "input": ("bio3", "co2"), "amount": 2.0},
                {"type": "biosphere", "input": ("bio3", "ch4"), "amount": 0.5},
            ],
        }
    ]
    frame = ExchangeFrameBuilder().from_sp_data(sp_data)
    co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
    ch4_id = ExchangeFrameBuilder.flow_id_for(("bio3", "ch4"))
    climate = MethodAliases.resolve("climate")
    ozone = MethodAliases.resolve("ozone")
    cfs = {
        climate: pd.DataFrame({"flow_id": [co2_id, ch4_id], "cf": [1.0, 30.0]}),
        ozone: pd.DataFrame({"flow_id": [ch4_id], "cf": [0.001]}),
    }
    pkg = ScoringPackageBuilder().build(frame, cfs)

    pc_df = pd.DataFrame(
        [
            {
                "database": "agb",
                "code": "wheat",
                "name": "wheat",
                "type": "process",
                "unit": "kg",
                "product_id": ExchangeFrameBuilder.flow_id_for(("agb", "wheat-prod")),
            }
        ]
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
    pc_path = tmp_path / "product_catalog.parquet"
    pc_df.to_parquet(pc_path, index=False)

    bio_df = pd.DataFrame(
        [
            {
                "database": "bio3",
                "code": "co2",
                "name": "Carbon dioxide",
                "categories": ["air"],
                "unit": "kg",
                "cas": "",
            },
            {
                "database": "bio3",
                "code": "ch4",
                "name": "Methane",
                "categories": ["air"],
                "unit": "kg",
                "cas": "",
            },
        ]
    )
    return pkg, ProductCatalog.load(pc_path), bio_df


def _scores_df(
    codes: list[str],
    mapped: list[bool] | None = None,
    product_db: str = "agb",
    product_codes: list[str] | None = None,
) -> pd.DataFrame:
    mapped = mapped if mapped is not None else [True] * len(codes)
    # By default, the product_code equals the Code AGB (matches the existing
    # _build_package_and_catalog helper where the product is keyed as
    # ("agb", "wheat") with Code AGB = "wheat").
    product_codes = product_codes if product_codes is not None else list(codes)
    return pd.DataFrame(
        {
            "Code AGB": codes,
            "mapped": mapped,
            "Nom du Produit": codes,
            "product_db": [product_db if m else None for m in mapped],
            "product_code": [c if m else None for c, m in zip(product_codes, mapped, strict=True)],
        }
    )


class TestFlowDecompositionEmitterHappyPath:
    def test_writes_one_json_per_mapped_product(self, tmp_path):
        pkg, pc, bio = _build_package_and_catalog(tmp_path)
        decomposer = ScoreDecomposer(
            package=pkg, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        out_dir = tmp_path / "decomp"
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=bio,
            method_short_to_full={
                "climate": MethodAliases.resolve("climate"),
                "ozone": MethodAliases.resolve("ozone"),
            },
            out_dir=out_dir,
            top_n=10,
        )

        emitter.write(_scores_df(["wheat"]))

        out_file = out_dir / "wheat.json"
        assert out_file.exists()
        payload = json.loads(out_file.read_text())
        assert payload["code"] == "wheat"
        assert set(payload["methods"]) == {"climate", "ozone"}

        climate = payload["methods"]["climate"]
        assert climate["score"] == pytest.approx(17.0)
        # Two flows, sorted by |contribution| descending: CH4 (15) > CO2 (2).
        assert [f["flow_name"] for f in climate["flows"]] == ["Methane", "Carbon dioxide"]
        assert climate["flows"][0]["contribution"] == pytest.approx(15.0)
        assert climate["flows"][0]["inventory"] == pytest.approx(0.5)
        assert climate["flows"][0]["cf"] == pytest.approx(30.0)
        assert climate["flows"][0]["unit"] == "kg"
        # Shares against |Σcontribution| = 17, so CH4 share = 15/17.
        assert climate["flows"][0]["share"] == pytest.approx(15.0 / 17.0)
        assert climate["total_abs_contribution"] == pytest.approx(17.0)


class TestFlowDecompositionEmitterTruncation:
    def test_top_n_keeps_largest_abs_contributions(self, tmp_path):
        pkg, pc, bio = _build_package_and_catalog(tmp_path)
        decomposer = ScoreDecomposer(
            package=pkg, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=bio,
            method_short_to_full={"climate": MethodAliases.resolve("climate")},
            out_dir=tmp_path / "decomp",
            top_n=1,
        )
        emitter.write(_scores_df(["wheat"]))

        payload = json.loads((tmp_path / "decomp" / "wheat.json").read_text())
        flows = payload["methods"]["climate"]["flows"]
        assert len(flows) == 1
        # Largest |contribution| is CH4 (15) — CO2 (2) is dropped.
        assert flows[0]["flow_name"] == "Methane"
        # total_abs_contribution still reflects the UN-truncated set (17),
        # so the visible share bar can be less than 100%.
        assert payload["methods"]["climate"]["total_abs_contribution"] == pytest.approx(17.0)
        assert flows[0]["share"] == pytest.approx(15.0 / 17.0)
        # Truncation must leave the kept share strictly below 1.0 because
        # the dropped CO2 row still counts toward total_abs_contribution.
        assert flows[0]["share"] < 1.0


class TestFlowDecompositionEmitterMappedFilter:
    def test_skips_rows_where_mapped_is_false(self, tmp_path):
        pkg, pc, bio = _build_package_and_catalog(tmp_path)
        decomposer = ScoreDecomposer(
            package=pkg, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        out_dir = tmp_path / "decomp"
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=bio,
            method_short_to_full={"climate": MethodAliases.resolve("climate")},
            out_dir=out_dir,
            top_n=10,
        )
        emitter.write(_scores_df(["wheat", "unknown"], mapped=[True, False]))

        assert (out_dir / "wheat.json").exists()
        assert not (out_dir / "unknown.json").exists()


class TestFlowDecompositionEmitterUnknownProduct:
    def test_skips_mapped_row_whose_code_is_absent_from_product_catalog(self, tmp_path):
        pkg, pc, bio = _build_package_and_catalog(tmp_path)
        decomposer = ScoreDecomposer(
            package=pkg, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        out_dir = tmp_path / "decomp"
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=bio,
            method_short_to_full={"climate": MethodAliases.resolve("climate")},
            out_dir=out_dir,
            top_n=10,
        )
        emitter.write(_scores_df(["wheat", "ghost"]))

        assert (out_dir / "wheat.json").exists()
        assert not (out_dir / "ghost.json").exists()


class TestFlowDecompositionEmitterUnregisteredMethod:
    def test_method_not_in_scoring_package_is_dropped_from_payload(self, tmp_path):
        pkg, pc, bio = _build_package_and_catalog(tmp_path)
        decomposer = ScoreDecomposer(
            package=pkg, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        # 'acid' is a valid alias but no acid CFs exist in the test pkg
        # (it only has 'climate' and 'ozone'). The emitter logs a warning
        # and drops the method from the payload rather than crashing.
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=bio,
            method_short_to_full={
                "climate": MethodAliases.resolve("climate"),
                "acid": MethodAliases.resolve("acid"),
            },
            out_dir=tmp_path / "decomp",
            top_n=10,
        )
        emitter.write(_scores_df(["wheat"]))

        payload = json.loads((tmp_path / "decomp" / "wheat.json").read_text())
        assert "climate" in payload["methods"]
        assert "acid" not in payload["methods"]


class TestFlowDecompositionEmitterSimaProCf:
    def test_no_lookup_emits_null_simapro_fields(self, tmp_path):
        """Schema stability: the keys exist even without a lookup so the
        dashboard can always read them (rendering an em-dash)."""
        pkg, pc, bio = _build_package_and_catalog(tmp_path)
        decomposer = ScoreDecomposer(
            package=pkg, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=bio,
            method_short_to_full={"climate": MethodAliases.resolve("climate")},
            out_dir=tmp_path / "decomp",
            top_n=10,
        )
        emitter.write(_scores_df(["wheat"]))

        flows = json.loads((tmp_path / "decomp" / "wheat.json").read_text())["methods"]["climate"][
            "flows"
        ]
        for f in flows:
            assert f["sp_cf"] is None
            assert f["sp_match_provenance"] is None

    def test_lookup_attaches_simapro_cf_per_flow(self, tmp_path):
        pkg, pc, bio = _build_package_and_catalog(tmp_path)
        decomposer = ScoreDecomposer(
            package=pkg, product_catalog=pc, biosphere_catalog=bio, use_pardiso=False
        )
        emitter = FlowDecompositionEmitter(
            decomposer=decomposer,
            biosphere_catalog=bio,
            method_short_to_full={
                "climate": MethodAliases.resolve("climate"),
                "ozone": MethodAliases.resolve("ozone"),
            },
            out_dir=tmp_path / "decomp",
            top_n=10,
            simapro_cf=_simapro_lookup(),
        )
        emitter.write(_scores_df(["wheat"]))

        payload = json.loads((tmp_path / "decomp" / "wheat.json").read_text())
        by_name = {f["flow_name"]: f for f in payload["methods"]["climate"]["flows"]}
        # CO2 has a comparable SimaPro CF; CH4's row is present but sp_cf is null.
        assert by_name["Carbon dioxide"]["sp_cf"] == pytest.approx(1.1)
        assert by_name["Carbon dioxide"]["sp_match_provenance"] == "code"
        assert by_name["Carbon dioxide"]["cf"] == pytest.approx(1.0)
        assert by_name["Methane"]["sp_cf"] is None
        assert by_name["Methane"]["sp_match_provenance"] is None
        # The lookup carries no ozone coverage → all ozone flows stay null.
        for f in payload["methods"]["ozone"]["flows"]:
            assert f["sp_cf"] is None


class TestSimaProCfLookup:
    def test_skips_null_sp_cf_and_resolves_present(self):
        lookup = _simapro_lookup()
        climate = MethodAliases.resolve("climate")
        hit = lookup.get("co2", climate[2])
        assert hit is not None
        assert hit.sp_cf == pytest.approx(1.1)
        assert hit.provenance == "code"
        # Null cf_simapro rows are not stored, and unknown methods miss cleanly.
        assert lookup.get("ch4", climate[2]) is None
        assert lookup.get("co2", "ozone depletion") is None

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing columns"):
            SimaProCfLookup.from_dataframe(pd.DataFrame({"code": ["co2"]}))
