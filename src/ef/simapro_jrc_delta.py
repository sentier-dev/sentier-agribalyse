"""``SimaProJrcDeltaAudit`` — surface (sp_method_root, flow_name) pairs
that SimaPro EF 3.1 (adapted) characterises but JRC EF v3.1 does not.

This is the §A.5 sweep tool. The Barite case (§A.1) is the exemplar:
SimaPro carries Barite ecotox at CF=322.16 across 8 rows; JRC has zero
matching rows. Without dropping it, the CF leaks through aluminium-can
and duck-farming chains and inflates ecotox scores.

The output is a *candidate list*. Whether to add a given pair to
``SimaProCfFilter.EXCLUDED_FLOW_NAMES_BY_SP_METHOD`` requires backtest
evidence that the flow appears in the matrix with non-zero mass on at
least one AGB-side product and that dropping the SimaPro CF improves
a dominated outlier without regressing any clean product.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import pandas as pd

from ef.cf_table import EfCfTable
from ef.simapro_cf_table import SimaProEFCfTable


@dataclass(frozen=True)
class SimaProJrcDeltaAudit:
    """Compute (sp_method_root, flow_name) pairs in SimaPro but not JRC."""

    simapro_table: SimaProEFCfTable
    ef_cf_table: EfCfTable

    # SimaPro splits ``Ecotoxicity, freshwater`` and ``Human toxicity, ...``
    # into ``- inorganics`` / ``- organics`` sub-methods. We fold them into
    # the root for comparison against JRC's single combined method.
    SUBMETHOD_SUFFIXES: ClassVar[tuple[str, ...]] = (" - inorganics", " - organics")

    @cached_property
    def _sp_pairs(self) -> pd.DataFrame:
        """``(sp_method_root, flow_name_lower) → max_abs_cf`` for non-zero rows."""
        df = self.simapro_table.df.copy()
        if df.empty:
            return pd.DataFrame(columns=["sp_method_root", "flow_name_lower", "max_abs_cf"])
        df["sp_method_root"] = df["simapro_method"].map(self._strip_submethod)
        df["flow_name_lower"] = df["name"].astype(str).str.strip().str.lower()
        df = df[df["cf"].abs() > 0.0]
        if df.empty:
            return pd.DataFrame(columns=["sp_method_root", "flow_name_lower", "max_abs_cf"])
        return (
            df.groupby(["sp_method_root", "flow_name_lower"], as_index=False)["cf"]
            .agg(lambda s: s.abs().max())
            .rename(columns={"cf": "max_abs_cf"})
        )

    @cached_property
    def _jrc_method_flow_names(self) -> dict[str, set[str]]:
        """``jrc_method_name → {flow_name_lower, …}`` from the JRC CF parquet."""
        cf = self.ef_cf_table.raw
        out: dict[str, set[str]] = {}
        if cf.empty:
            return out
        for method, names in cf.groupby("LCIAMethod_name")["FLOW_name"].unique().items():
            cleaned = {str(n).strip().lower() for n in names if isinstance(n, str) and n.strip()}
            out[method] = cleaned
        return out

    def candidates(self) -> pd.DataFrame:
        """Return the audit DataFrame.

        Columns:
        * ``sp_method_root`` — SimaPro method root (post sub-method fold).
        * ``flow_name_lower`` — flow name, lowercased + stripped.
        * ``max_abs_cf`` — max |CF| across SimaPro rows for that pair.
        * ``our_category`` / ``our_indicator`` — our (category, indicator)
          tuple if the SimaPro method is known, else None.
        * ``jrc_method`` — the JRC ``LCIAMethod_name`` we intersected
          against (resolved from ``METHOD_TO_OUR_KEY`` reversed).
        * ``in_jrc`` — always False here (rows present in JRC are filtered out).
        """
        sp = self._sp_pairs
        empty_cols = [
            "sp_method_root",
            "flow_name_lower",
            "max_abs_cf",
            "our_category",
            "our_indicator",
            "jrc_method",
            "in_jrc",
        ]
        if sp.empty:
            return pd.DataFrame(columns=empty_cols)
        sp = sp.assign(
            our_category=sp["sp_method_root"].map(
                lambda m: SimaProEFCfTable.METHOD_TO_OUR_KEY.get(m, (None, None))[0]
            ),
            our_indicator=sp["sp_method_root"].map(
                lambda m: SimaProEFCfTable.METHOD_TO_OUR_KEY.get(m, (None, None))[1]
            ),
        )
        sp["jrc_method"] = sp["sp_method_root"].map(self._sp_to_jrc_method)
        jrc_lookup = self._jrc_method_flow_names

        def _has_jrc_row(row: pd.Series) -> bool:
            j_method = row["jrc_method"]
            if not isinstance(j_method, str) or not j_method:
                return False
            return row["flow_name_lower"] in jrc_lookup.get(j_method, set())

        sp["in_jrc"] = sp.apply(_has_jrc_row, axis=1)
        return (
            sp[~sp["in_jrc"]]
            .sort_values(["sp_method_root", "max_abs_cf"], ascending=[True, False])
            .reset_index(drop=True)
        )

    @classmethod
    def _strip_submethod(cls, sp_method: str) -> str:
        for suffix in cls.SUBMETHOD_SUFFIXES:
            if sp_method.endswith(suffix):
                return sp_method[: -len(suffix)]
        return sp_method

    @staticmethod
    def _sp_to_jrc_method(sp_method_root: str) -> str | None:
        """Translate a SimaPro method root to the JRC ``LCIAMethod_name``.

        The mapping is implicit: ``SimaProEFCfTable.METHOD_TO_OUR_KEY``
        maps SimaPro→our key, and ``MethodCfRegistryBuilder.EF_METHOD_MAP``
        maps our key→JRC name. The two reverse to a SimaPro→JRC bridge.
        """
        from ef.cf_registry import MethodCfRegistryBuilder

        our_key = SimaProEFCfTable.METHOD_TO_OUR_KEY.get(sp_method_root)
        if our_key is None:
            return None
        return MethodCfRegistryBuilder.EF_METHOD_MAP.get(our_key)
