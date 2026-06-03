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

from pipelines.link_all import LinkAllPipeline
from reporting import RunReport
from scoring.activity_location_overrides import ActivityLocationOverrides
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.method_slug import MethodSlug
from tests.fixtures.builders import FakeSimaProImporter


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
    """Empty biosphere catalog with the schema AwareConsumptionCorrectionBuilder
    expects (``database``, ``code``, ``name``, ``categories``).

    The AWARE consumption correction reads this parquet to resolve the
    SimaPro EF v3.1 (adapted) water flow set; an empty file keeps the
    builder's gate from firing while letting the read succeed.
    """
    path = settings.paths.registry_biosphere_catalog
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "database": pd.Series([], dtype="string"),
            "code": pd.Series([], dtype="string"),
            "name": pd.Series([], dtype="string"),
            "categories": pd.Series([], dtype="object"),
        }
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
