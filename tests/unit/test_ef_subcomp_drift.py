"""Tests for ``MethodCfSubcompDriftAudit`` — surface JRC method+flow
groups whose CF varies meaningfully across sub-compartments. Reviewers
use this output to predict where bw2io's ``unspecified``-fallback CF
inheritance can skew scoring (the §C.3 PM / climate diagnostic).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ef.cf_table import EfCfTable
from ef.subcomp_drift_audit import MethodCfSubcompDriftAudit


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


def _row(name: str, sub: str, cf: float, method: str) -> dict:
    return {
        "FLOW_uuid": f"uuid-{name}-{sub}",
        "FLOW_name": name,
        "LCIAMethod_uuid EF3.1": "method-x",
        "LCIAMethod_name": method,
        "CF EF3.1": cf,
        "LCIAMethod_location": None,
        "FLOW_class0": "Emissions",
        "FLOW_class1": "Emissions to air",
        "FLOW_class2": sub,
        "LCIAMethod_derivation": None,
        "LCIAMethod_direction": "input",
    }


def test_surfaces_flow_with_2x_cf_variation_across_subcomps(tmp_path: Path):
    """Particles (PM2.5) ranges from CF=0 (long-term unspecified) to
    CF=2.4e-4 (urban close to ground) in JRC's PM method — a 100×
    spread that any unspecified-fallback inheritance would flatten.
    """
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _row(
                "particles (PM2.5)",
                "Emissions to urban air close to ground",
                2.4e-4,
                "EF-particulate Matter",
            ),
            _row(
                "particles (PM2.5)",
                "Emissions to non-urban air or from high stacks",
                6.0e-5,
                "EF-particulate Matter",
            ),
            _row(
                "particles (PM2.5)",
                "Emissions to air, unspecified",
                1.2e-4,
                "EF-particulate Matter",
            ),
        ],
    )
    audit = MethodCfSubcompDriftAudit(ef_cf_table=jrc)

    df = audit.candidates(method_name="EF-particulate Matter", min_ratio=2.0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["FLOW_name"] == "particles (PM2.5)"
    assert row["min_cf"] == pytest.approx(6.0e-5)
    assert row["max_cf"] == pytest.approx(2.4e-4)
    assert row["n_subcomps"] == 3
    # ratio = max / min = 4.0
    assert row["ratio"] == pytest.approx(4.0)


def test_omits_flow_with_uniform_cf_across_subcomps(tmp_path: Path):
    """Methane (fossil) ships at CF=29.8 across every sub-comp in JRC
    cc:fossil — no sub-comp variation, no audit candidate.
    """
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _row(
                "methane (fossil)",
                "Emissions to urban air close to ground",
                29.8,
                "Climate change-Fossil",
            ),
            _row(
                "methane (fossil)",
                "Emissions to non-urban air or from high stacks",
                29.8,
                "Climate change-Fossil",
            ),
            _row(
                "methane (fossil)", "Emissions to air, unspecified", 29.8, "Climate change-Fossil"
            ),
        ],
    )
    audit = MethodCfSubcompDriftAudit(ef_cf_table=jrc)
    assert audit.candidates(method_name="Climate change-Fossil", min_ratio=2.0).empty


def test_min_ratio_threshold_filters_small_variation(tmp_path: Path):
    """A 1.5× spread is below the default 2.0 ratio threshold."""
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _row("flowA", "sub1", 1.0, "EF-particulate Matter"),
            _row("flowA", "sub2", 1.5, "EF-particulate Matter"),
            _row("flowB", "sub1", 1.0, "EF-particulate Matter"),
            _row("flowB", "sub2", 5.0, "EF-particulate Matter"),
        ],
    )
    audit = MethodCfSubcompDriftAudit(ef_cf_table=jrc)
    df = audit.candidates(method_name="EF-particulate Matter", min_ratio=2.0)
    assert list(df["FLOW_name"]) == ["flowB"]


def test_zero_min_cf_uses_max_minus_min_threshold(tmp_path: Path):
    """When the smallest CF is 0 (ratio undefined), the audit must
    still surface the flow if max_cf is materially non-zero."""
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _row("flowZ", "sub1", 0.0, "EF-particulate Matter"),
            _row("flowZ", "sub2", 1.0e-4, "EF-particulate Matter"),
        ],
    )
    audit = MethodCfSubcompDriftAudit(ef_cf_table=jrc)
    df = audit.candidates(method_name="EF-particulate Matter", min_ratio=2.0)
    assert list(df["FLOW_name"]) == ["flowZ"]
    assert df.iloc[0]["min_cf"] == pytest.approx(0.0)
    assert df.iloc[0]["max_cf"] == pytest.approx(1.0e-4)


def test_skips_unknown_method(tmp_path: Path):
    jrc = _write_jrc_cf(
        tmp_path,
        [_row("flowA", "sub1", 1.0, "EF-particulate Matter")],
    )
    audit = MethodCfSubcompDriftAudit(ef_cf_table=jrc)
    df = audit.candidates(method_name="No Such Method", min_ratio=2.0)
    assert df.empty


def test_sorts_by_ratio_descending(tmp_path: Path):
    jrc = _write_jrc_cf(
        tmp_path,
        [
            _row("flowSmall", "sub1", 1.0, "EF-particulate Matter"),
            _row("flowSmall", "sub2", 3.0, "EF-particulate Matter"),
            _row("flowBig", "sub1", 1.0, "EF-particulate Matter"),
            _row("flowBig", "sub2", 100.0, "EF-particulate Matter"),
        ],
    )
    audit = MethodCfSubcompDriftAudit(ef_cf_table=jrc)
    df = audit.candidates(method_name="EF-particulate Matter", min_ratio=2.0)
    assert list(df["FLOW_name"]) == ["flowBig", "flowSmall"]
