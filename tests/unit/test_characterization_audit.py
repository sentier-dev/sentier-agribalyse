import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scoring.characterization_audit import audit_characterization
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.matrix_builder import BuiltMatrix


def fixture():
    catalog = pd.DataFrame(
        [
            {"database": "bio", "code": "ch4", "name": "Methane", "categories": ("air",)},
            {
                "database": "bio",
                "code": "co2",
                "name": "Carbon dioxide, biogenic",
                "categories": ("air",),
            },
            {"database": "bio", "code": "nh3", "name": "Ammonia", "categories": ("air",)},
        ]
    )
    ids = [ExchangeFrameBuilder.flow_id_for(("bio", c)) for c in catalog.code]
    b = BuiltMatrix(
        matrix=sparse.csr_matrix([[1.0], [1.0], [2.0]]),
        row_id_to_idx=dict(zip(ids, range(3), strict=True)),
        col_id_to_idx={1: 0},
    )
    cfs = {("test", "climate change"): pd.DataFrame({"flow_id": ids[:2], "cf": [27.0, 0.0]})}
    return b, cfs, catalog


def test_explicit_zero_is_characterized_and_ammonia_not_required():
    b, cfs, catalog = fixture()
    audit = audit_characterization(b, cfs, catalog)
    assert audit["status"] == "passed"
    assert audit["methods"][0]["explicit_zero_flows"] == 1
    assert len(audit["methods"][0]["missing_flow_ids"]) == 1


def test_unqualified_methane_missing_cf_fails_without_guessing():
    b, cfs, catalog = fixture()
    cfs[("test", "climate change")] = cfs[("test", "climate change")].iloc[1:]
    audit = audit_characterization(b, cfs, catalog)
    assert audit["status"] == "failed"
    assert audit["methods"][0]["missing_core_gases"][0]["name"] == "Methane"


def test_zero_inventory_row_not_required():
    b, cfs, catalog = fixture()
    b.matrix[0, 0] = 0
    cfs[("test", "climate change")] = cfs[("test", "climate change")].iloc[1:]
    assert audit_characterization(b, cfs, catalog)["status"] == "passed"


def test_unlabelled_used_flow_cannot_be_certified():
    b, cfs, catalog = fixture()
    assert audit_characterization(b, cfs, catalog.iloc[1:])["status"] == "failed"


def test_nonfinite_and_duplicate_factors_fail():
    b, cfs, catalog = fixture()
    key = ("test", "climate change")
    cfs[key].loc[0, "cf"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        audit_characterization(b, cfs, catalog)
    b, cfs, catalog = fixture()
    cfs[key] = pd.concat([cfs[key], cfs[key].iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate"):
        audit_characterization(b, cfs, catalog)
