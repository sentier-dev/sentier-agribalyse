"""``RegionalCorrectionBuilder`` — per-activity correction row for one method.

The matrix scorer is built around a flat 1×N characterisation vector per
method (one CF per biosphere flow), multiplied as
``Q @ B @ supply``. Regional CFs (AWARE water consumption per ISO country)
break that assumption: the CF for an activity emitting flow F depends on
*which activity* (its location) emits it, not just on F.

To preserve the fast 1×N path for the 18 non-regional methods, we compute
a *per-activity correction vector* at build time::

    delta_vec[j] = Σ_i (regional_cf[i, location(j)] - global_cf[i]) × B[i, j]

so the score becomes ``(Q @ inventory) + (delta_vec @ supply)``. The
correction is contracted across biosphere rows at build time — at score
time it's an extra sparse dot product per method, indistinguishable in
shape from the regular characterisation step.

Methods without regional CFs simply don't get a correction row;
``ScoringPackage.corrections`` is keyed by method tuple and absent keys
mean "no correction needed".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse as sp

from scoring.matrix_builder import BuiltMatrix


@dataclass(frozen=True)
class RegionalCorrectionBuilder:
    """Compose one method's per-activity correction row from regional CFs."""

    def build(
        self,
        *,
        global_cf_df: pd.DataFrame,
        regional_cf_df: pd.DataFrame,
        biosphere: BuiltMatrix,
        technosphere: BuiltMatrix,
        col_id_to_location: dict[int, str],
    ) -> sp.csr_matrix:
        """Return a ``(1, n_activities)`` sparse correction row.

        ``global_cf_df`` columns: ``(flow_id int, cf float)``.
        ``regional_cf_df`` columns: ``(flow_id int, location str, cf float)``.
        ``col_id_to_location`` maps technosphere column ids (the
        ``output_id`` hash) to the activity's location string.
        """
        n_activities = technosphere.matrix.shape[1]
        if regional_cf_df.empty or n_activities == 0:
            return sp.csr_matrix((1, n_activities))

        # Resolve regional flow ids → biosphere row indices. Flow ids
        # that aren't in the biosphere row map (e.g. an EF-coded flow
        # the matrix doesn't carry) are dropped — they can't contribute
        # to any activity's score anyway.
        flow_to_row = biosphere.row_id_to_idx
        regional = regional_cf_df.copy()
        regional["flow_row"] = regional["flow_id"].map(flow_to_row)
        regional = regional.dropna(subset=["flow_row"])
        if regional.empty:
            return sp.csr_matrix((1, n_activities))
        regional["flow_row"] = regional["flow_row"].astype("int64")

        # Per-flow global CF lookup. A flow that exists regionally but
        # not globally takes 0.0 as the global; the delta then equals
        # the regional value — same as if a brand-new CF were applied
        # for activities at that location.
        if global_cf_df.empty:
            global_by_flow: dict[int, float] = {}
        else:
            global_by_flow = {
                int(fid): float(cf)
                for fid, cf in zip(global_cf_df["flow_id"], global_cf_df["cf"], strict=False)
            }

        # Group activity columns by location once so each regional row
        # turns into a single sparse slice + scatter-add.
        location_to_cols: dict[str, np.ndarray] = {}
        for col_id, location in col_id_to_location.items():
            if not isinstance(location, str) or not location:
                continue
            col_idx = technosphere.col_id_to_idx.get(int(col_id))
            if col_idx is None:
                continue
            location_to_cols.setdefault(location, []).append(col_idx)
        location_to_cols = {
            loc: np.asarray(sorted(cols), dtype="int64") for loc, cols in location_to_cols.items()
        }

        b_csr = biosphere.matrix
        delta = np.zeros(n_activities, dtype="float64")

        # Iterate over (flow_row, location) groups so we read each
        # B-row only once per location.
        grouped = regional.groupby(["flow_row", "location"], sort=False)
        for (flow_row, location), group in grouped:
            cols = location_to_cols.get(location)
            if cols is None or cols.size == 0:
                continue
            regional_cf = float(group["cf"].iloc[0])
            global_cf = global_by_flow.get(int(flow_row))
            if global_cf is None:
                # Bio row exists, but no global CF — recover via the
                # biosphere row inverse map. The matrix doesn't know
                # about flow_id ↔ row mappings outside ``row_id_to_idx``,
                # so iterate to find the flow_id whose row is flow_row.
                # Cost is negligible (one walk per method build).
                global_cf = 0.0
                for fid, ridx in flow_to_row.items():
                    if ridx == int(flow_row):
                        global_cf = global_by_flow.get(int(fid), 0.0)
                        break
            d = regional_cf - global_cf
            if d == 0.0:
                continue
            # Sparse row slice → dense for the few activity columns at
            # this location. ``b_row.toarray().ravel()`` materialises one
            # N-length row, then we index the small location subset.
            b_row = b_csr.getrow(int(flow_row)).toarray().ravel()
            delta[cols] += d * b_row[cols]

        if not np.any(delta):
            return sp.csr_matrix((1, n_activities))
        return sp.csr_matrix(delta.reshape(1, -1))
