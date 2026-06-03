"""``MethodCfSubcompDriftAudit`` — surface JRC method+flow groups whose
CF varies materially across sub-compartments.

This is the §C.3 PM / climate diagnostic. Bw2io's snapshot inherits
CFs onto bio3 codes by name; when JRC's CF on a flow varies between
sub-compartments (e.g. PM2.5 ranges from 0 long-term to 2.4e-4 urban-
close-to-ground — factor ~∞ across sub-comps), any inheritance that
collapses to the unspecified-fallback CF will skew scoring relative to
JRC's intended sub-comp-specific characterisation.

The audit lists every (method, FLOW_name) group whose ``max_cf / min_cf``
exceeds a configurable threshold (default 2.0×). Reviewers cross-check
the matrix's actual bio3 sub-comp targets to find rows the snapshot
mis-CFed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ef.cf_table import EfCfTable


@dataclass(frozen=True)
class MethodCfSubcompDriftAudit:
    """Per-method (FLOW_name) audit of CF spread across sub-compartments."""

    ef_cf_table: EfCfTable

    def candidates(self, *, method_name: str, min_ratio: float = 2.0) -> pd.DataFrame:
        """Return one row per FLOW_name with sub-comp CF spread ≥ ``min_ratio``.

        Columns:
        * ``FLOW_name`` — the JRC flow name.
        * ``min_cf`` / ``max_cf`` — CF range across sub-compartments
          for this method.
        * ``n_subcomps`` — count of distinct sub-comps with rows.
        * ``ratio`` — ``max_cf / min_cf`` if ``min_cf > 0`` else ``inf``.

        Sorted by ratio descending so the most-variant flows surface
        first.
        """
        df = self.ef_cf_table.raw
        df = df[df["LCIAMethod_name"] == method_name]
        if df.empty:
            return pd.DataFrame(columns=["FLOW_name", "min_cf", "max_cf", "n_subcomps", "ratio"])
        # Use null-region rows when present; else the full set. For our
        # purposes the audit just needs *any* CF samples per sub-comp so
        # we work off the raw rows (every region of a sub-comp shares the
        # same per-sub-comp CF in the JRC dataset).
        df = df.dropna(subset=["FLOW_name", "FLOW_class2", "CF EF3.1"])
        if df.empty:
            return pd.DataFrame(columns=["FLOW_name", "min_cf", "max_cf", "n_subcomps", "ratio"])
        rows: list[dict] = []
        for name, g in df.groupby("FLOW_name"):
            sub_cf = g.groupby("FLOW_class2")["CF EF3.1"].first()
            cf_min = float(sub_cf.min())
            cf_max = float(sub_cf.max())
            if cf_max == cf_min:
                continue
            if cf_min > 0.0:
                ratio = cf_max / cf_min
                if ratio < min_ratio:
                    continue
            else:
                # min=0 → ratio undefined. Surface anyway when max_cf is
                # materially non-zero (any caller-meaningful threshold).
                if cf_max == 0.0:
                    continue
                ratio = float("inf")
            rows.append(
                {
                    "FLOW_name": name,
                    "min_cf": cf_min,
                    "max_cf": cf_max,
                    "n_subcomps": len(sub_cf),
                    "ratio": ratio,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["FLOW_name", "min_cf", "max_cf", "n_subcomps", "ratio"])
        out = pd.DataFrame(rows)
        return out.sort_values("ratio", ascending=False, na_position="last").reset_index(drop=True)
