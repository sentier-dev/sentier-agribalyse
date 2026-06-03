"""Tests for ``SimaProJrcDeltaAudit`` — the §A.5 sweep tool that surfaces
(sp_method_root, flow_name) pairs SimaPro EF 3.1 (adapted) characterises
but JRC EF v3.1 does not.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from ef.cf_table import EfCfTable
from ef.simapro_cf_table import SimaProEFCfTable
from ef.simapro_jrc_delta import SimaProJrcDeltaAudit


def _write_simapro_table(tmp_path: Path, rows: list[dict]) -> SimaProEFCfTable:
    cache = tmp_path / "simapro.parquet"
    xlsx = tmp_path / "simapro.xlsx"
    xlsx.write_bytes(b"")
    df = pd.DataFrame(rows, columns=list(SimaProEFCfTable.COLUMNS))
    df.to_parquet(cache)
    later = time.time() + 10
    os.utime(cache, (later, later))
    return SimaProEFCfTable(xlsx_path=xlsx, cache_path=cache)


def _write_jrc_cf(tmp_path: Path, rows: list[dict]) -> EfCfTable:
    cols = [
        "FLOW_uuid",
        "FLOW_name",
        "LCIAMethod_uuid EF3.1",
        "LCIAMethod_name",
        "CF EF3.1",
        "LCIAMethod_location",
        "FLOW_class0",
        "FLOW_class1",
        "FLOW_class2",
        "LCIAMethod_derivation",
        "LCIAMethod_direction",
    ]
    df = pd.DataFrame(rows, columns=cols)
    p = tmp_path / "jrc_cf.parquet"
    df.to_parquet(p)
    return EfCfTable(path=p)


def test_audit_surfaces_barite_ecotox_pair(tmp_path: Path):
    """The Barite case: SimaPro carries CF=322.16 in ``Ecotoxicity,
    freshwater`` (and the inorganics sub-method). JRC has zero rows for
    Barite. The audit must surface it as a candidate.
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
    # JRC has NO Barite row in ecotox.
    jrc = _write_jrc_cf(
        tmp_path,
        [
            {
                "FLOW_uuid": "uuid-copper",
                "FLOW_name": "copper",
                "LCIAMethod_uuid EF3.1": "method-ecotox",
                "LCIAMethod_name": "Ecotoxicity, freshwater",
                "CF EF3.1": 1234.0,
                "LCIAMethod_location": None,
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to water",
                "FLOW_class2": "Emissions to fresh water",
                "LCIAMethod_derivation": None,
                "LCIAMethod_direction": "input",
            }
        ],
    )
    audit = SimaProJrcDeltaAudit(simapro_table=sp, ef_cf_table=jrc)

    df = audit.candidates()
    assert len(df) == 1, df.to_string()
    row = df.iloc[0]
    assert row["sp_method_root"] == "Ecotoxicity, freshwater"
    assert row["flow_name_lower"] == "barite"
    assert row["max_abs_cf"] == pytest.approx(322.16)
    assert row["our_category"] == "ecotoxicity: freshwater"
    assert row["jrc_method"] == "Ecotoxicity, freshwater"
    assert row["in_jrc"] is False or row["in_jrc"] == False  # noqa: E712


def test_audit_omits_flows_present_in_jrc(tmp_path: Path):
    """If a flow exists in JRC for the same method, it's not a candidate."""
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Strontium",
                "cas": "7440-24-6",
                "cf": 18073.0,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    jrc = _write_jrc_cf(
        tmp_path,
        [
            {
                "FLOW_uuid": "uuid-sr",
                "FLOW_name": "strontium",
                "LCIAMethod_uuid EF3.1": "method-ecotox",
                "LCIAMethod_name": "Ecotoxicity, freshwater",
                "CF EF3.1": 18073.0,
                "LCIAMethod_location": None,
                "FLOW_class0": "Emissions",
                "FLOW_class1": "Emissions to water",
                "FLOW_class2": "Emissions to fresh water",
                "LCIAMethod_derivation": None,
                "LCIAMethod_direction": "input",
            }
        ],
    )
    audit = SimaProJrcDeltaAudit(simapro_table=sp, ef_cf_table=jrc)

    assert audit.candidates().empty


def test_audit_drops_zero_cf_simapro_rows(tmp_path: Path):
    """A SimaPro CF of 0.0 cannot inflate scores → not a candidate."""
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "Inert Filler",
                "cas": "",
                "cf": 0.0,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    jrc = _write_jrc_cf(tmp_path, [])
    audit = SimaProJrcDeltaAudit(simapro_table=sp, ef_cf_table=jrc)
    assert audit.candidates().empty


def test_audit_uses_max_abs_cf_across_subcomps(tmp_path: Path):
    """Multiple SimaPro rows for the same (method, flow_name) collapse
    to the max absolute CF — that's the worst-case scoring impact.
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
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Soil",
                "sub_compartment": "(unspecified)",
                "name": "Barite",
                "cas": "",
                "cf": 2.13,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    jrc = _write_jrc_cf(tmp_path, [])
    audit = SimaProJrcDeltaAudit(simapro_table=sp, ef_cf_table=jrc)

    df = audit.candidates()
    assert len(df) == 1
    assert df.iloc[0]["max_abs_cf"] == pytest.approx(322.16)


def test_audit_sorts_by_method_then_max_abs_cf_desc(tmp_path: Path):
    sp = _write_simapro_table(
        tmp_path,
        [
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "FlowSmall",
                "cas": "",
                "cf": 1.0,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
            {
                "simapro_method": "Ecotoxicity, freshwater",
                "simapro_method_unit": "CTUe",
                "compartment": "Water",
                "sub_compartment": "(unspecified)",
                "name": "FlowBig",
                "cas": "",
                "cf": 100.0,
                "flow_unit": "kg",
                "cf_unit": "CTUe/kg",
            },
        ],
    )
    jrc = _write_jrc_cf(tmp_path, [])
    audit = SimaProJrcDeltaAudit(simapro_table=sp, ef_cf_table=jrc)

    df = audit.candidates()
    assert list(df["flow_name_lower"]) == ["flowbig", "flowsmall"]
