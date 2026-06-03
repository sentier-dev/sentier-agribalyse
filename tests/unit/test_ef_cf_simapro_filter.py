"""Tests for ``SimaProCfFilter`` — per-method ``keep`` predicate.

The filter intersects bw2io-inherited bio3 CFs with the SimaPro EF 3.1
(adapted) reference set. These tests pin the per-flow exclusion list
that overrides "SimaPro carries it" when JRC has zero rows for that
flow (eg. Barite at CF=322.16 in ecotox).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from ef.cf_simapro_filter import SimaProCfFilter
from ef.simapro_cf_table import SimaProEFCfTable

# ----------------------------------------------------------------------------
# Helpers


def _write_simapro_table(tmp_path: Path, rows: list[dict]) -> SimaProEFCfTable:
    """Build a ``SimaProEFCfTable`` backed by a parquet at ``tmp_path``.

    We touch a stub xlsx and write a fresh parquet with mtime ahead of it
    so ``df`` reads the parquet without invoking openpyxl on a real file.
    """
    cache = tmp_path / "simapro.parquet"
    xlsx = tmp_path / "simapro.xlsx"
    xlsx.write_bytes(b"")
    df = pd.DataFrame(rows, columns=list(SimaProEFCfTable.COLUMNS))
    df.to_parquet(cache)
    later = time.time() + 10
    os.utime(cache, (later, later))
    return SimaProEFCfTable(xlsx_path=xlsx, cache_path=cache)


def _write_biosphere_catalog(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(
        rows,
        columns=["database", "code", "name", "categories", "unit", "cas", "synonyms"],
    )
    p = tmp_path / "biosphere_catalog.parquet"
    df.to_parquet(p)
    return p


def _write_empty_flowmap(tmp_path: Path) -> Path:
    p = tmp_path / "flowmap.json"
    p.write_text(json.dumps({"update": []}))
    return p


# ----------------------------------------------------------------------------
# Per-flow exclusion list (§A.1)


def test_keep_excludes_barite_from_ecotox_even_when_simapro_carries_it(tmp_path: Path):
    """Barite appears in SimaPro EF v3.1 (adapted) at CF=322.16 but is
    absent from JRC EF v3.1's authoritative dataset. The filter must
    drop it from the ecotox method.
    """
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Barite",
                "cas": "",
                "cf": 322.16,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    bio = _write_biosphere_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "17a7db69-d874-4ecd-af5d-48c98def445f",
                "name": "Barite",
                "categories": ["water"],
                "unit": "kg",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    fm = _write_empty_flowmap(tmp_path)
    flt = SimaProCfFilter(simapro_table=sp, biosphere_catalog_path=bio, flowmap_path=fm)

    kept = flt.keep(
        our_category="ecotoxicity: freshwater",
        our_indicator="comparative toxic unit for ecosystems (CTUe)",
        db="biosphere3",
        code="17a7db69-d874-4ecd-af5d-48c98def445f",
    )
    assert kept is False, "Barite ecotox should be filtered (not in JRC)"


def test_keep_excludes_barite_in_inorganics_submethod(tmp_path: Path):
    """SimaPro lists Barite under ``Ecotoxicity, freshwater - inorganics``
    too (CF=322.16). The exclusion must strip both root and inorganics
    sub-method rows because ``_strip_submethod`` folds them together.
    """
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater - inorganics",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Barite",
                "cas": "",
                "cf": 322.16,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    bio = _write_biosphere_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "17a7db69-d874-4ecd-af5d-48c98def445f",
                "name": "Barite",
                "categories": ["water"],
                "unit": "kg",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    fm = _write_empty_flowmap(tmp_path)
    flt = SimaProCfFilter(simapro_table=sp, biosphere_catalog_path=bio, flowmap_path=fm)

    kept = flt.keep(
        our_category="ecotoxicity: freshwater",
        our_indicator="comparative toxic unit for ecosystems (CTUe)",
        db="biosphere3",
        code="17a7db69-d874-4ecd-af5d-48c98def445f",
    )
    assert kept is False, "Barite inorganics ecotox row should also be filtered"


def test_keep_excludes_all_barite_subcompartments_in_ecotox(tmp_path: Path):
    """Bio3 carries five Barite codes (water / water,ocean / water,ground- /
    water,ground-,long-term / water,surface water). The exclusion is
    keyed by ``flow name``, not (name, sub-comp), so all variants drop.
    """
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Barite",
                "cas": "",
                "cf": 322.16,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    bio = _write_biosphere_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "barite-ground",
                "name": "Barite",
                "categories": ["water", "ground-"],
                "unit": "kg",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "barite-surface",
                "name": "Barite",
                "categories": ["water", "surface water"],
                "unit": "kg",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    fm = _write_empty_flowmap(tmp_path)
    flt = SimaProCfFilter(simapro_table=sp, biosphere_catalog_path=bio, flowmap_path=fm)

    for code in ("barite-ground", "barite-surface"):
        assert (
            flt.keep(
                our_category="ecotoxicity: freshwater",
                our_indicator="comparative toxic unit for ecosystems (CTUe)",
                db="biosphere3",
                code=code,
            )
            is False
        ), f"Barite/{code} should be filtered for ecotox regardless of sub-compartment"


def test_keep_allows_non_excluded_flow_in_ecotox(tmp_path: Path):
    """Sanity check: a flow in SimaPro's ecotox set that is NOT on the
    exclusion list passes ``keep``."""
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Copper",
                "cas": "7440-50-8",
                "cf": 1234.0,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    bio = _write_biosphere_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "copper-uuid",
                "name": "Copper",
                "categories": ["water"],
                "unit": "kg",
                "cas": "7440-50-8",
                "synonyms": [],
            },
        ],
    )
    fm = _write_empty_flowmap(tmp_path)
    flt = SimaProCfFilter(simapro_table=sp, biosphere_catalog_path=bio, flowmap_path=fm)

    kept = flt.keep(
        our_category="ecotoxicity: freshwater",
        our_indicator="comparative toxic unit for ecosystems (CTUe)",
        db="biosphere3",
        code="copper-uuid",
    )
    assert kept is True, "Copper is on SimaPro's ecotox set and not excluded"


def test_keep_excludes_sodium_chloride_from_material_resources(tmp_path: Path):
    """``Sodium chloride [natural resource, in ground]`` is absent from
    SimaPro's ``Resource use, minerals and metals`` set. The biosphere
    flowmap maps it to the elemental ``Sodium`` entry SimaPro DOES carry,
    so the synonym fallback in ``keep`` would otherwise wave it through
    at the bw2io-inherited CF (1.6e-5 kg Sb-eq/kg). The per-flow
    exclusion list must drop it: rock-salt is an ultimate-reserve flow
    JRC excludes from mater, and the synonym is a flow-identity match,
    not a CF-equivalence claim.
    """
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Resource use, minerals and metals",
                "simapro_method_unit": "kg Sb eq",
                "compartment": "Raw",
                "sub_compartment": "(in ground)",
                "name": "Sodium",
                "cas": "",
                "cf": 1.6e-5,
                "flow_unit": "kg",
                "cf_unit": "kg Sb-eq/kg",
            },
        ],
    )
    bio = _write_biosphere_catalog(
        tmp_path,
        [
            {
                "database": "ecoinvent-3.9.1-biosphere",
                "code": "0b9159dd-305d-4add-802f-f7b780ed0289",
                "name": "Sodium chloride",
                "categories": ["natural resource", "in ground"],
                "unit": "kg",
                "cas": "",
                "synonyms": [],
            },
        ],
    )
    # Flowmap declares the same synonym present in production: bio3
    # 'Sodium chloride [natural resource]' aliases SimaPro's 'sodium'.
    flowmap = {
        "update": [
            {
                "source": {"name": "sodium"},
                "target": {"identifier": "0b9159dd-305d-4add-802f-f7b780ed0289"},
            }
        ]
    }
    fm = tmp_path / "flowmap.json"
    fm.write_text(json.dumps(flowmap))
    flt = SimaProCfFilter(simapro_table=sp, biosphere_catalog_path=bio, flowmap_path=fm)

    kept = flt.keep(
        our_category="material resources: metals/minerals",
        our_indicator="abiotic depletion potential (ADP): elements (ultimate reserves)",
        db="ecoinvent-3.9.1-biosphere",
        code="0b9159dd-305d-4add-802f-f7b780ed0289",
    )
    assert kept is False, (
        "Sodium chloride [natural resource, in ground] should be filtered "
        "from mater even when the flowmap aliases it to elemental Sodium"
    )


def test_filter_rows_drops_excluded_flow_and_counts_it(tmp_path: Path):
    """``filter_rows`` must remove the excluded row and bump n_dropped by 1."""
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Barite",
                "cas": "",
                "cf": 322.16,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Copper",
                "cas": "7440-50-8",
                "cf": 1234.0,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    bio = _write_biosphere_catalog(
        tmp_path,
        [
            {
                "database": "biosphere3",
                "code": "barite-uuid",
                "name": "Barite",
                "categories": ["water"],
                "unit": "kg",
                "cas": "",
                "synonyms": [],
            },
            {
                "database": "biosphere3",
                "code": "copper-uuid",
                "name": "Copper",
                "categories": ["water"],
                "unit": "kg",
                "cas": "7440-50-8",
                "synonyms": [],
            },
        ],
    )
    fm = _write_empty_flowmap(tmp_path)
    flt = SimaProCfFilter(simapro_table=sp, biosphere_catalog_path=bio, flowmap_path=fm)

    rows = [
        {"database": "biosphere3", "code": "barite-uuid", "amount": 322.16},
        {"database": "biosphere3", "code": "copper-uuid", "amount": 1234.0},
    ]
    kept, n_dropped = flt.filter_rows(
        our_key=(
            "category",
            "indicator",
            "ecotoxicity: freshwater",
            "comparative toxic unit for ecosystems (CTUe)",
        ),
        rows=rows,
    )
    assert n_dropped == 1
    assert [r["code"] for r in kept] == ["copper-uuid"]
