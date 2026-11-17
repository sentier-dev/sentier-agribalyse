"""Unit tests for ``LinkAllPipeline``.

Covers two contracts:

1. **EF parquet precondition** (Phase L2): the build-time CLIs must have
   produced ``registry/ef_flows.parquet`` and ``registry/method_cfs/_index.json``
   before the linker starts.
2. **Scoring-package emit tail** (Phase L3): the linker terminates by
   calling ``sp.drop_unlinked`` once, projecting ``sp.data`` through
   ``ExchangeFrameBuilder`` + ``Allocator`` + ``ScoringPackageBuilder``,
   writing the package directory under ``cache/scoring_packages/<hash>``,
   emitting ``registry/product_catalog.parquet``, and recording the
   content hash on the run report. ``sp.write_database`` and
   ``MatrixPurger`` MUST NOT be called from this path.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from pipelines.fast_rescore import FastRescorePipeline
from pipelines.link_all import LinkAllPipeline
from reporting import RunReport
from scoring.activity_location_overrides import ActivityLocationOverrides
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.method_slug import MethodSlug
from scoring.scoring_package_locator import ScoringPackageLocator
from tests.fixtures.builders import FakeSimaProImporter
from transforms.linked_cache import LinkedSpCache
from transforms.sp_csv_parser import ParsedSimaProCsv


class TestLinkAllPipelineEfPrecondition:
    def test_missing_flows_parquet_raises_clear_error(self, settings):
        with pytest.raises(FileNotFoundError, match="EF flows registry"):
            LinkAllPipeline._assert_ef_parquets(settings)

    def test_missing_index_raises_clear_error(self, settings):
        # Provide flows but no method-CFs index.
        flows_path = settings.paths.registry_ef_flows
        flows_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["code", "name", "categories", "unit", "cas"]).to_parquet(
            flows_path, index=False
        )
        with pytest.raises(FileNotFoundError, match="Method-CFs registry index"):
            LinkAllPipeline._assert_ef_parquets(settings)

    def test_returns_row_counts_when_both_present(self, settings):
        flows_path = settings.paths.registry_ef_flows
        flows_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "code": "u1",
                    "name": "n1",
                    "categories": ["air"],
                    "unit": "kg",
                    "cas": None,
                },
                {
                    "code": "u2",
                    "name": "n2",
                    "categories": ["water"],
                    "unit": "kg",
                    "cas": None,
                },
            ]
        ).to_parquet(flows_path, index=False)

        index_path = settings.paths.registry_method_cfs_index
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "methods": [
                        {"key": ["a", "b"], "slug": "a__b", "n_cfs": 3},
                    ],
                }
            )
        )

        stats = LinkAllPipeline._assert_ef_parquets(settings)
        assert stats == {"ef_flows": 2, "methods": 1}


# ============================================================================
# Phase L3: scoring-package emit tail.


def _seed_empty_ecoinvent_exchanges(settings):
    """Stub the source-side ecoinvent snapshot with an empty parquet.

    ``_emit_scoring_package`` always concatenates the AGB long-form
    rows with ecoinvent's; an empty parquet keeps the schema correct
    while letting the test focus on the AGB-only matrix shape."""
    path = settings.paths.ecoinvent_exchanges_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "output_database": pd.Series([], dtype="string"),
            "output_code": pd.Series([], dtype="string"),
            "input_database": pd.Series([], dtype="string"),
            "input_code": pd.Series([], dtype="string"),
            "amount": pd.Series([], dtype="float64"),
            "type": pd.Series([], dtype="string"),
        }
    ).to_parquet(path)


def _seed_biosphere_catalog(settings):
    """Catalog the synthetic emitted CO2 so climate coverage can be certified."""
    path = settings.paths.registry_biosphere_catalog
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "database": "bio3",
                "code": "co2",
                "name": "Carbon dioxide, fossil",
                "categories": ["air"],
            }
        ]
    ).to_parquet(path, index=False)


def _seed_empty_ef_cf_parquet(settings):
    """Empty EF v3.1 CF parquet with the columns ``RegionalWaterCfTable``
    + ``EfCfTable`` filter on. ``cf_by_location`` returns ``{}`` for an
    empty table, so the AWARE regional-CF lookup contributes no rows."""
    path = settings.paths.ef_cf_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "LCIAMethod_name": pd.Series([], dtype="string"),
            "LCIAMethod_location": pd.Series([], dtype="string"),
            "LCIAMethod_direction": pd.Series([], dtype="string"),
            "FLOW_uuid": pd.Series([], dtype="string"),
            "FLOW_name": pd.Series([], dtype="string"),
            "FLOW_class0": pd.Series([], dtype="string"),
            "FLOW_class1": pd.Series([], dtype="string"),
            "FLOW_class2": pd.Series([], dtype="string"),
            "CF EF3.1": pd.Series([], dtype="float64"),
        }
    ).to_parquet(path, index=False)


def _seed_method_cfs_registry(settings, method_key, rows):
    """Lay down a single-method registry: ``<dir>/<slug>/cfs.parquet`` + index."""
    out_dir = settings.paths.registry_method_cfs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = MethodSlug.encode(method_key)
    method_dir = out_dir / slug
    method_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["database", "code", "amount"]).astype(
        {"database": "string", "code": "string"}
    )
    df.to_parquet(method_dir / "cfs.parquet", index=False)
    index_path = settings.paths.registry_method_cfs_index
    index_path.write_text(
        json.dumps(
            {
                "version": 1,
                "methods": [
                    {"key": list(method_key), "slug": slug, "n_cfs": len(rows)},
                ],
            },
            sort_keys=True,
            indent=2,
        )
    )


def _trivial_sp_data():
    """Two activities, one shared product, one biosphere flow with a CF.

    Activity 'a' produces product (agb, p1) and emits 1 kg CO2.
    Activity 'b' produces product (agb, p2) and consumes (agb, p1).
    The technosphere is square (2 products × 2 activities) without
    needing the allocator to split anything.
    """
    return [
        {
            "database": "agb",
            "code": "a",
            "name": "wheat",
            "type": "process",
            "unit": "kg",
            "exchanges": [
                {"type": "production", "input": ("agb", "p1"), "amount": 1.0},
                {"type": "biosphere", "input": ("bio3", "co2"), "amount": 1.0},
            ],
        },
        {
            "database": "agb",
            "code": "b",
            "name": "bread",
            "type": "process",
            "unit": "kg",
            "exchanges": [
                {"type": "production", "input": ("agb", "p2"), "amount": 1.0},
                {"type": "technosphere", "input": ("agb", "p1"), "amount": 0.4},
                # An unlinked exchange — must be stripped by drop_unlinked
                # before reaching the matrix builders.
                {"type": "technosphere", "amount": 0.1},
            ],
        },
    ]


class TestEmitScoringPackage:
    """Tests on ``LinkAllPipeline._emit_scoring_package`` — exercises the
    whole tail without booting Brightway / SimaPro."""

    METHOD_KEY = ("ef", "v3.1", "climate change", "GWP100")
    CO2_FACTOR = ({"database": "bio3", "code": "co2", "amount": 1.0},)
    NO_CO2_FACTOR = ({"database": "bio3", "code": "unused", "amount": 1.0},)

    def _seed(self, settings, rows):
        _seed_method_cfs_registry(settings, self.METHOD_KEY, rows=list(rows))
        _seed_empty_ecoinvent_exchanges(settings)
        _seed_biosphere_catalog(settings)
        _seed_empty_ef_cf_parquet(settings)

    def test_failed_audit_writes_its_own_receipt_and_emits_no_package(self, settings):
        self._seed(settings, self.NO_CO2_FACTOR)
        sp = FakeSimaProImporter(_trivial_sp_data())
        with pytest.raises(ValueError, match="Characterization audit failed"):
            LinkAllPipeline._emit_scoring_package(sp, settings, RunReport())
        receipt = json.loads(settings.paths.dashboard_characterization_audit.read_text())
        assert receipt["status"] == "failed"
        assert receipt["methods"][0]["missing_core_gases"][0]["code"] == "co2"
        assert len(receipt["candidate_content_hash"]) == 64
        assert not settings.paths.dashboard_run_report.exists()
        assert not list(settings.paths.scoring_packages_root.glob("*/technosphere.csr.npz"))

    def test_failed_audit_keeps_the_pointer_to_the_last_emitted_package(self, settings):
        # A full link run passes and records its package, as LinkAllPipeline.run() does.
        self._seed(settings, self.CO2_FACTOR)
        report = RunReport()
        payload = LinkAllPipeline._emit_scoring_package(
            FakeSimaProImporter(_trivial_sp_data()), settings, report
        )
        report.add_stage("scoring_package", payload)
        report.write(settings.paths.dashboard_run_report)
        last_good = settings.paths.dashboard_run_report.read_bytes()
        assert report.stages["characterization_audit"]["status"] == "passed"
        receipt = json.loads(settings.paths.dashboard_characterization_audit.read_text())
        assert receipt["status"] == "passed"
        assert receipt["candidate_content_hash"] == payload["content_hash"]

        # The registry loses the CO2 factor; the next full run starts a fresh report.
        self._seed(settings, self.NO_CO2_FACTOR)
        with pytest.raises(ValueError, match="Characterization audit failed"):
            LinkAllPipeline._emit_scoring_package(
                FakeSimaProImporter(_trivial_sp_data()), settings, RunReport()
            )

        assert settings.paths.dashboard_run_report.read_bytes() == last_good
        locator = ScoringPackageLocator(settings=settings)
        assert locator.content_hash() == payload["content_hash"]
        assert locator.load().content_hash == payload["content_hash"]
        receipt = json.loads(settings.paths.dashboard_characterization_audit.read_text())
        assert receipt["status"] == "failed"
        assert receipt["candidate_content_hash"] != payload["content_hash"]

    def test_an_audit_that_raises_writes_no_receipt_and_keeps_the_last_one(self, settings):
        self._seed(settings, self.CO2_FACTOR)
        report = RunReport()
        payload = LinkAllPipeline._emit_scoring_package(
            FakeSimaProImporter(_trivial_sp_data()), settings, report
        )
        report.add_stage("scoring_package", payload)
        report.write(settings.paths.dashboard_run_report)
        last_receipt = settings.paths.dashboard_characterization_audit.read_bytes()
        last_report = settings.paths.dashboard_run_report.read_bytes()

        # Two factor rows for one flow: the audit cannot be made, it raises.
        self._seed(settings, self.CO2_FACTOR + self.CO2_FACTOR)
        with pytest.raises(ValueError, match="Duplicate characterization flow ids"):
            LinkAllPipeline._emit_scoring_package(
                FakeSimaProImporter(_trivial_sp_data()), settings, RunReport()
            )

        assert settings.paths.dashboard_characterization_audit.read_bytes() == last_receipt
        assert settings.paths.dashboard_run_report.read_bytes() == last_report
        assert not list(settings.paths.dashboard.glob("*.partial"))

    def test_registry_without_a_climate_method_emits_and_says_it_was_not_audited(self, settings):
        # Same emitted CO2 without a factor as in the failing tests, but no climate method:
        # the audit has nothing to certify, so it must not report a pass either.
        _seed_method_cfs_registry(
            settings, ("ef", "v3.1", "acidification", "AE"), rows=list(self.NO_CO2_FACTOR)
        )
        _seed_empty_ecoinvent_exchanges(settings)
        _seed_biosphere_catalog(settings)
        _seed_empty_ef_cf_parquet(settings)
        report = RunReport()
        payload = LinkAllPipeline._emit_scoring_package(
            FakeSimaProImporter(_trivial_sp_data()), settings, report
        )
        assert (settings.paths.scoring_packages_root / payload["content_hash"]).is_dir()
        assert report.stages["characterization_audit"]["status"] == "not_audited"
        receipt = json.loads(settings.paths.dashboard_characterization_audit.read_text())
        assert receipt["status"] == "not_audited"
        assert receipt["not_audited_reason"]

    def test_failed_audit_in_a_fast_rescore_keeps_the_last_run_report(self, settings):
        # The fast path loads the last report from disk; a failed audit must not put
        # ``status: failed`` next to the content hash of the package that passed.
        settings.paths.agribalyse_csv.parent.mkdir(parents=True, exist_ok=True)
        settings.paths.agribalyse_csv.write_text("stand-in for the SimaPro export")
        LinkedSpCache(settings=settings).write(
            ParsedSimaProCsv(db_name="agb", data=_trivial_sp_data(), parameters=[])
        )
        self._seed(settings, self.CO2_FACTOR)
        passed = FastRescorePipeline(settings=settings).run()
        last_good = settings.paths.dashboard_run_report.read_bytes()

        self._seed(settings, self.NO_CO2_FACTOR)
        with pytest.raises(ValueError, match="Characterization audit failed"):
            FastRescorePipeline(settings=settings).run()

        assert settings.paths.dashboard_run_report.read_bytes() == last_good
        assert (
            ScoringPackageLocator(settings=settings).content_hash()
            == passed.stages["scoring_package"]["content_hash"]
        )
        receipt = json.loads(settings.paths.dashboard_characterization_audit.read_text())
        assert receipt["status"] == "failed"

    def test_writes_scoring_package_and_records_content_hash(self, settings):
        method_key = ("ef", "v3.1", "climate change", "GWP100")
        co2_id = ExchangeFrameBuilder.flow_id_for(("bio3", "co2"))
        _seed_method_cfs_registry(
            settings,
            method_key,
            rows=[{"database": "bio3", "code": "co2", "amount": 1.0}],
        )

        _seed_empty_ecoinvent_exchanges(settings)
        _seed_biosphere_catalog(settings)
        _seed_empty_ef_cf_parquet(settings)
        sp = FakeSimaProImporter(_trivial_sp_data())
        report = RunReport()

        payload = LinkAllPipeline._emit_scoring_package(sp, settings, report)

        # drop_unlinked called exactly once with the reckless flag.
        assert sp.drop_unlinked_calls == 1
        # write_database NOT called — would have raised AssertionError.
        assert sp.write_database_calls == 0

        # Stage payload carries the package metadata.
        assert "content_hash" in payload and len(payload["content_hash"]) == 64
        assert payload["n_products"] == 2
        assert payload["n_activities"] == 2
        assert payload["n_methods"] == 1
        # exchanges_dropped reflects the unlinked technosphere row.
        assert payload["exchanges_dropped"] == 1

        # Run-report matrix shape matches the package.
        assert report.matrix_shape == {
            "rows": 2,
            "cols": 2,
            "deficit": 0,
            "square": True,
        }

        # Package directory exists and is keyed by the content hash.
        package_root = settings.paths.scoring_packages_root
        package_dir = package_root / payload["content_hash"]
        assert package_dir.is_dir()
        assert (package_dir / "technosphere.csr.npz").exists()
        assert (package_dir / "biosphere.csr.npz").exists()
        assert (package_dir / "ids.json").exists()

        # The CO2 CF was projected to (flow_id, cf) — flow_id matches
        # ``ExchangeFrameBuilder.flow_id_for`` for the same key.
        ids = json.loads((package_dir / "ids.json").read_text())
        assert str(co2_id) in ids["biosphere_row_id_to_idx"]

    def test_writes_product_catalog_parquet(self, settings):
        method_key = ("ef", "v3.1", "climate change", "GWP100")
        _seed_method_cfs_registry(
            settings,
            method_key,
            rows=[{"database": "bio3", "code": "co2", "amount": 1.0}],
        )

        _seed_empty_ecoinvent_exchanges(settings)
        _seed_biosphere_catalog(settings)
        _seed_empty_ef_cf_parquet(settings)
        sp = FakeSimaProImporter(_trivial_sp_data())
        LinkAllPipeline._emit_scoring_package(sp, settings, RunReport())

        catalog = pd.read_parquet(settings.paths.registry_product_catalog)
        # One row per activity in sp.data, sorted by (database, code).
        assert list(zip(catalog["database"], catalog["code"], strict=False)) == [
            ("agb", "a"),
            ("agb", "b"),
        ]
        # product_id resolves the activity's *production input* — the
        # actual product node, not the activity's own (db, code). For
        # ``_trivial_sp_data`` activity 'a' produces ('agb', 'p1') and
        # activity 'b' produces ('agb', 'p2').
        expected_products = {"a": ("agb", "p1"), "b": ("agb", "p2")}
        for _, row in catalog.iterrows():
            expected = ExchangeFrameBuilder.flow_id_for(expected_products[row["code"]])
            assert int(row["product_id"]) == expected
        # Schema columns present.
        assert set(catalog.columns) >= {
            "database",
            "code",
            "name",
            "type",
            "unit",
            "product_id",
        }

    def test_does_not_import_matrix_purger(self, settings, monkeypatch):
        # MatrixPurger has been deleted — but if any future refactor
        # accidentally re-introduces it, this guard fails fast.
        method_key = ("ef", "v3.1", "climate change", "GWP100")
        _seed_method_cfs_registry(
            settings,
            method_key,
            rows=[{"database": "bio3", "code": "co2", "amount": 1.0}],
        )

        _seed_empty_ecoinvent_exchanges(settings)
        _seed_biosphere_catalog(settings)
        _seed_empty_ef_cf_parquet(settings)
        sp = FakeSimaProImporter(_trivial_sp_data())
        # If anything tries to import ``reporting.matrix_purger`` the
        # import error surfaces here.
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("reporting.matrix_purger")

        payload = LinkAllPipeline._emit_scoring_package(sp, settings, RunReport())
        assert payload, "Scoring-package payload must be non-empty"
        assert "content_hash" in payload


class TestColIdToLocationOverrides:
    """``_col_id_to_location`` must honour the curated per-activity overrides.

    The default resolver leaves GLO/RoW/RER at None (global CF fallback),
    which over-counts evaporation on FR-greenhouse AGB stand-ins. The
    override table is the escape hatch — exhaustively tested here so a
    later refactor can't silently drop the override pass and let sweet
    pepper drift back to 4.8× ADEME.
    """

    @staticmethod
    def _sp_dataset(name: str, db: str = "agribalyse-3.2", code: str = "X"):
        return {"name": name, "database": db, "code": code, "location": None}

    def test_override_promotes_glo_to_iso_country(self, tmp_path):
        sp_data = [self._sp_dataset("bell pepper production, in heated greenhouse GLO")]
        key = ExchangeFrameBuilder.flow_id_for(("agribalyse-3.2", "X"))

        # Without override → GLO collapses to AGGREGATE_FALLBACK (None);
        # the entry is dropped from the map and the global CF is used.
        baseline = LinkAllPipeline._col_id_to_location(
            pd.DataFrame(columns=["database", "code", "location"]),
            sp_data,
            overrides=ActivityLocationOverrides.empty(),
        )
        assert key not in baseline

        # With override → the activity is mapped to FR despite the GLO
        # name suffix, so the regional correction will fire for it.
        overrides_path = self._write_overrides(
            tmp_path / "ov.json",
            [{"database": "agribalyse-3.2", "code": "X", "location": "FR"}],
        )
        with_override = LinkAllPipeline._col_id_to_location(
            pd.DataFrame(columns=["database", "code", "location"]),
            sp_data,
            overrides=ActivityLocationOverrides.load(overrides_path),
        )
        assert with_override[key] == "FR"

    def test_override_wins_over_ecoinvent_catalog_location(self, tmp_path):
        """Catalog says BR; override pins FR — override wins."""
        ei = pd.DataFrame(
            {"database": ["ecoinvent-3.9.1-cutoff"], "code": ["abc"], "location": ["BR"]}
        )
        overrides_path = tmp_path / "ov.json"
        self._write_overrides(
            overrides_path,
            [{"database": "ecoinvent-3.9.1-cutoff", "code": "abc", "location": "FR"}],
        )
        out = LinkAllPipeline._col_id_to_location(
            ei, sp_data=None, overrides=ActivityLocationOverrides.load(overrides_path)
        )
        key = ExchangeFrameBuilder.flow_id_for(("ecoinvent-3.9.1-cutoff", "abc"))
        assert out[key] == "FR"

    def test_no_override_kwarg_is_backwards_compatible(self):
        # Older call sites pre-fix don't pass ``overrides=``. The default
        # behavior must match the historic resolver: aggregate fallback
        # logic applies, no overrides forced.
        sp_data = [self._sp_dataset("activity name FR", code="Y")]
        out = LinkAllPipeline._col_id_to_location(
            pd.DataFrame(columns=["database", "code", "location"]), sp_data
        )
        key = ExchangeFrameBuilder.flow_id_for(("agribalyse-3.2", "Y"))
        assert out[key] == "FR"

    @staticmethod
    def _write_overrides(path, rows: list[dict]) -> pathlib.Path:  # noqa: F821 - typing-only string
        path.write_text(json.dumps({"version": 1, "overrides": rows}))
        return path


def test_project_method_cfs_empty():
    empty = pd.DataFrame(
        {
            "database": pd.Series([], dtype="string"),
            "code": pd.Series([], dtype="string"),
            "amount": pd.Series([], dtype="float64"),
        }
    )
    result = LinkAllPipeline._project_method_cfs(empty)
    assert list(result.columns) == ["flow_id", "cf"]
    assert len(result) == 0
    assert result["flow_id"].dtype == "int64"
    assert result["cf"].dtype == "float64"
