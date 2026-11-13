"""Audit the final biosphere against source CF rows, including explicit zero factors.

This checks coverage, not factor correctness. It never invents a factor or resolves
unqualified methane. The core-gas policy covers CO2, CH4 and N2O emitted to air;
all other missing rows remain visible for method-specific review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.matrix_builder import BuiltMatrix

CORE_GASES = {
    "carbon dioxide",
    "carbon dioxide, fossil",
    "carbon dioxide, non-fossil",
    "carbon dioxide, biogenic",
    "carbon dioxide, peat oxidation",
    "methane",
    "methane, fossil",
    "methane, non-fossil",
    "methane, biogenic",
    "methane, peat oxidation",
    "dinitrogen monoxide",
    "dinitrogen monoxide, peat oxidation",
}


def audit_characterization(
    biosphere: BuiltMatrix,
    method_cfs: dict[tuple[str, ...], pd.DataFrame],
    catalog: pd.DataFrame,
) -> dict:
    matrix = biosphere.matrix.copy()
    matrix.eliminate_zeros()
    used = {
        key
        for key, row in biosphere.row_id_to_idx.items()
        if matrix.indptr[row + 1] > matrix.indptr[row]
    }
    labels, required = {}, set()
    for row in catalog.itertuples(index=False):
        key = ExchangeFrameBuilder.flow_id_for((row.database, row.code))
        labels[key] = {"database": str(row.database), "code": str(row.code), "name": str(row.name)}
        categories = tuple(row.categories) if row.categories is not None else ()
        if (
            categories
            and str(categories[0]).casefold() == "air"
            and str(row.name).strip().casefold() in CORE_GASES
        ):
            required.add(key)
    missing_labels = sorted(used - labels.keys())
    methods, failures = [], []
    for method, factors in sorted(method_cfs.items()):
        values = factors["cf"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite characterization factor in {method}")
        # Duplicate CF rows would be summed by CharacterizationBuilder. Reject
        # them here instead of mistaking presence for correct characterization.
        if factors["flow_id"].duplicated().any():
            raise ValueError(f"Duplicate characterization flow ids in {method}")
        covered = set(int(key) for key in factors["flow_id"])
        missing = sorted(used - covered)
        climate = any(str(part).casefold() == "climate change" for part in method)
        missing_required = sorted((used & required) - covered) if climate else []
        # Unlabelled rows could hide core gases; a climate audit cannot certify them.
        if climate and (missing_required or missing_labels):
            failures.append(list(method))
        methods.append(
            {
                "method": list(method),
                "emitted_flows": len(used),
                "characterized_flows": len(used & covered),
                "explicit_zero_flows": int(
                    factors.loc[factors.flow_id.isin(used), "cf"].eq(0).sum()
                ),
                "missing_flow_ids": missing,
                "missing_core_gases": [labels[key] for key in missing_required],
            }
        )
    return {
        "status": "failed" if failures else "passed",
        "policy": "Core CO2/CH4/N2O air emissions require source CF rows in climate-change methods; no inferred CFs.",
        "scope": "Coverage only; other gases and non-climate omissions require method-specific review.",
        "unlabelled_emitted_flow_ids": missing_labels,
        "failed_methods": failures,
        "methods": methods,
    }
