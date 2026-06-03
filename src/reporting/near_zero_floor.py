"""``NearZeroFloor`` — collapse near-zero computed/reference pairs to 0.

When both the computed score and the ADEME reference for a method are
well below the method's natural scale (default: 1 % of the median
``|reference|`` across all mapped products), the percentage delta blows
up to four-digit values that reflect numerical noise rather than
calibration error — e.g. ``Tap water cc_luc = +5631 %`` after the
multi-output waste-treatment fix. This class detects those cells and
zeros all four entries (``computed_*``, ``reference_*``, ``diff_abs``,
``diff_pct``) so the backtest dashboard surfaces only meaningful
discrepancies.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd


@dataclass(frozen=True)
class NearZeroFloor:
    """Per-method ``|x| < factor * median(|reference|)`` zero-threshold floor."""

    thresholds: Mapping[str, float]
    factor: float

    @classmethod
    def compute(
        cls,
        scores_df: pd.DataFrame,
        method_shortnames: Sequence[str],
        factor: float = 0.01,
    ) -> NearZeroFloor:
        """Derive per-method thresholds from the absolute median reference."""
        thresholds: dict[str, float] = {}
        for short in method_shortnames:
            col = f"reference_{short}"
            if col not in scores_df.columns:
                thresholds[short] = 0.0
                continue
            reference = pd.to_numeric(scores_df[col], errors="coerce").abs().dropna()
            reference = reference[reference > 0.0]
            if reference.empty:
                thresholds[short] = 0.0
                continue
            thresholds[short] = factor * float(reference.median())
        return cls(thresholds=MappingProxyType(thresholds), factor=factor)

    def apply(
        self,
        scores_df: pd.DataFrame,
        diff_abs: MutableMapping[str, pd.Series],
        diff_pct: MutableMapping[str, pd.Series],
    ) -> dict[str, int]:
        """Mutate ``scores_df`` and the diff maps in place; return zeroed counts."""
        counts: dict[str, int] = {}
        for short, thr in self.thresholds.items():
            counts[short] = 0
            if thr <= 0.0:
                continue
            ccol = f"computed_{short}"
            rcol = f"reference_{short}"
            if ccol not in scores_df.columns or rcol not in scores_df.columns:
                continue
            computed = pd.to_numeric(scores_df[ccol], errors="coerce")
            reference = pd.to_numeric(scores_df[rcol], errors="coerce")
            mask = (computed.abs() < thr) & (reference.abs() < thr)
            mask = mask.fillna(False).astype(bool)
            n = int(mask.sum())
            counts[short] = n
            if n == 0:
                continue
            scores_df.loc[mask, ccol] = 0.0
            scores_df.loc[mask, rcol] = 0.0
            if short in diff_abs:
                diff_abs[short] = diff_abs[short].mask(mask, 0.0)
            if short in diff_pct:
                diff_pct[short] = diff_pct[short].mask(mask, 0.0)
        return counts
