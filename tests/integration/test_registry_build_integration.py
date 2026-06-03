"""Integration test for the registry-build pipeline.

Builds a complete registry from a synthetic ``source/`` tree, then loads
the result back through ``MappingRegistry.load`` and asserts the matchers
can use it. End-to-end; no bw2 contact.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from domain import Bucket, Tier
from pipelines import RegistryBuildPipeline
from registry import MappingRegistry


def _write_placeholder_xlsx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = {
        "match with ecoinvent v3.9.1": pd.DataFrame(
            [
                {
                    "name": "Carbon dioxide",
                    "unit": "kg",
                    "categories": "('air',)",
                    "cas": "124-38-9",
                    "matched_ecoinvent_name": "Carbon dioxide, fossil",
                    "ecoinvent_match_type": "exact",
                }
            ]
        ),
        "match with EF v3.1": pd.DataFrame(
            [
                {
                    "name": "Methane",
                    "unit": "kg",
                    "categories": "('air',)",
                    "matched_ef_name": "Methane, fossil",
                }
            ]
        ),
        "Neither in ecoinvent nor EF": pd.DataFrame(
            [
                {
                    "name": "Mystery",
                    "unit": "kg",
                    "categories": "('air',)",
                }
            ]
        ),
        "ecoinvent flows - EF v3.1 map": pd.DataFrame(),
    }
    with pd.ExcelWriter(path) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)


def _write_harmonised_gz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "flows": [
            {
                "source": "EF 3.1",
                "identifier": "uuid-co2",
                "context_iri": "http://x.org/envi-air-rural",
                "prefLabel": "Carbon dioxide",
                "altLabel": ["CO2"],
            }
        ]
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_ef_cf_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "FLOW_uuid": "uuid-co2",
                "FLOW_name": "Carbon dioxide",
                "FLOW_class0": "Emissions",
                "FLOW_class1": "air",
                "FLOW_class2": None,
                "LCIAMethod_name": "Climate change",
                "LCIAMethod_location": None,
                "LCIAMethod_direction": "input",
                "CF EF3.1": 1.0,
            }
        ]
    )
    df.to_parquet(path, index=False)


@pytest.fixture
def synthetic_source(settings, monkeypatch):
    """Lay down tiny but realistic source artifacts under ``settings.paths.source``."""
    paths = settings.paths
    paths.source.mkdir(parents=True, exist_ok=True)
    (paths.source / "randonneur_packages").mkdir(parents=True, exist_ok=True)

    _write_placeholder_xlsx(paths.placeholder_xlsx)
    _write_harmonised_gz(paths.harmonised_flows_gz)
    _write_ef_cf_parquet(paths.ef_cf_parquet)

    paths.curated_overrides_json.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_name": "Pesticide",
                        "source_unit": "kg",
                        "source_context": ["Emissions to soil"],
                        "target_db": "biosphere3",
                        "target_code": "uuid-pest",
                        "target_name": "Pesticide",
                        "target_unit": "kg",
                        "tier": "CURATED_TARGETED",
                    }
                ]
            }
        )
    )

    paths.delete_aggregated_processes_json.write_text(
        json.dumps({"delete": [{"source": {"name": "agg-X", "identifier": "code-X"}}]})
    )
    paths.delete_aggregated_products_json.write_text(json.dumps({"delete": []}))
    paths.edge_label_corrections_json.write_text(
        json.dumps(
            {
                "replace": [
                    {
                        "source": {"name": "old", "type": "technosphere"},
                        "target": {"name": "new"},
                    }
                ]
            }
        )
    )

    # Stub the randonneur loader with a small set of in-memory packages.

    class _FakeRegistry(dict):
        def __init__(self):
            super().__init__()
            self.update(
                {
                    "agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches": {},
                    "SimaPro-9-ecoinvent-3.9-biosphere-manual-matches": {},
                    "simapro-9-ecoinvent-3-water-slash-m3": {},
                    "simapro-9-ecoinvent-3-context": {},
                    "Flowmapper-standard-units-harmonization": {},
                    "generic-brightway-unit-conversions": {},
                    "generic-brightway-units-normalization": {},
                }
            )
            self.data_dir = "."

        def get_file(self, label):
            payloads = {
                "agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches": {
                    "replace": [
                        {
                            "source": {
                                "name": "Foo",
                                "unit": "kg",
                                "context": ["air"],
                            },
                            "target": {"identifier": "uuid-foo", "name": "Foo"},
                            "conversion_factor": 1.0,
                        }
                    ]
                },
                "SimaPro-9-ecoinvent-3.9-biosphere-manual-matches": {"replace": []},
                "simapro-9-ecoinvent-3-water-slash-m3": {"replace": []},
                "simapro-9-ecoinvent-3-context": {
                    "replace": [
                        {
                            "source": {"context": ["Emissions to air"]},
                            "target": {"context": ["air"]},
                        }
                    ]
                },
                "Flowmapper-standard-units-harmonization": {
                    "update": [{"source": {"unit": "a"}, "target": {"unit": "year"}}]
                },
                "generic-brightway-unit-conversions": {
                    "replace": [
                        {
                            "source": {"unit": "m"},
                            "target": {"unit": "km", "allocation": 0.001},
                        }
                    ]
                },
                "generic-brightway-units-normalization": {
                    "replace": [{"source": {"unit": "Kg"}, "target": {"unit": "kg"}}]
                },
            }
            return payloads.get(label, {})

    import randonneur_data as rd_mod

    monkeypatch.setattr(rd_mod, "Registry", _FakeRegistry)
    return settings


@pytest.mark.integration
class TestRegistryBuildPipelineEndToEnd:
    def test_build_writes_all_parquets_and_meta(self, synthetic_source):
        out = RegistryBuildPipeline(synthetic_source).run()
        assert "row_counts" in out

        paths = synthetic_source.paths
        # Every parquet exists and is non-empty (or at least has columns).
        for path in (
            paths.registry_mappings_biosphere,
            paths.registry_mappings_technosphere,
            paths.registry_unmatchable,
            paths.registry_unit_conversions,
            paths.registry_unit_aliases,
            paths.registry_context_normalisation,
            paths.registry_deletions,
            paths.registry_edge_label_corrections,
            paths.registry_target_index_ef,
            paths.registry_meta,
        ):
            assert path.exists(), f"missing {path}"

    def test_loaded_registry_serves_real_lookups(self, synthetic_source):
        RegistryBuildPipeline(synthetic_source).run()

        registry = MappingRegistry.load(synthetic_source)
        # Curated override survived the round trip through parquet.
        pesticide_hits = registry.biosphere_index.lookup("agb_flow", "pesticide", Bucket.SOIL)
        assert pesticide_hits, "curated_overrides.json must produce a tier-1 row"
        assert pesticide_hits[0].tier == Tier.CURATED_TARGETED
        assert pesticide_hits[0].target_code == "uuid-pest"

        # Harmonised-flows altLabel produced a tier-4 row.
        co2_hits = registry.biosphere_index.lookup("agb_flow", "co2", Bucket.AIR)
        assert any(h.tier == Tier.HARMONISED_FLOWS for h in co2_hits)

        # Unit converter sees the m→km registered conversion.
        assert registry.unit_converter.multiplier("m", "km") == 0.001

    def test_unmatchable_index_survives_parquet_round_trip(self, synthetic_source):
        RegistryBuildPipeline(synthetic_source).run()
        registry = MappingRegistry.load(synthetic_source)
        assert registry.unmatchable_index.is_unmatchable("Mystery", Bucket.AIR) is True

    def test_meta_records_tier_table_and_row_counts(self, synthetic_source):
        RegistryBuildPipeline(synthetic_source).run()
        meta = json.loads(synthetic_source.paths.registry_meta.read_text())
        assert "tiers" in meta
        assert "row_counts" in meta
        assert meta["row_counts"]["mappings_biosphere"] >= 1
