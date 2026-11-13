"""``CharacterizationAuditor`` — audit the final biosphere against the method CF rows.

This checks coverage, not factor correctness. It never invents a factor or resolves
unqualified methane. The core-gas policy covers CO2, CH4 and N2O emitted to air;
all other missing rows remain visible for method-specific review.

A core gas without a factor row fails the audit, with one exception that the method
itself must prove: a species the method defines as exactly zero. EF 3.1 lists biogenic
CO2 emissions to air with a factor of 0, but under its own flow ids; the ecoinvent
biosphere flow "Carbon dioxide, non-fossil" that the inventory emits has no row of its
own. Such a flow is reported as zero by method definition, apart from a missing factor,
and only while the method's CF table carries those explicit zero rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.matrix_builder import BuiltMatrix


@dataclass(frozen=True)
class CharacterizationAuditor:
    """Stateless. ``audit(...)`` returns the receipt; ``status`` is ``passed`` or ``failed``."""

    CORE_GASES: ClassVar[frozenset[str]] = frozenset(
        {
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
    )

    #: Inventory flow name → the name the method's own table gives the same species, for
    #: species the method defines as exactly zero. Source: JRC EF 3.1 CF table, "Climate
    #: change", "carbon dioxide (biogenic)", emissions to air, CF 0 in every sub-compartment.
    #: An entry excuses nothing by itself — see :meth:`_method_defines_zero`.
    ZERO_BY_METHOD_DEFINITION: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "carbon dioxide, non-fossil": "carbon dioxide (biogenic)",
            "carbon dioxide, biogenic": "carbon dioxide (biogenic)",
        }
    )

    def audit(
        self,
        biosphere: BuiltMatrix,
        method_cfs: dict[tuple[str, ...], pd.DataFrame],
        catalog: pd.DataFrame,
    ) -> dict[str, Any]:
        matrix = biosphere.matrix.copy()
        matrix.eliminate_zeros()
        used = {
            key
            for key, row in biosphere.row_id_to_idx.items()
            if matrix.indptr[row + 1] > matrix.indptr[row]
        }
        labels, required = {}, set()
        air_flows_by_name: dict[str, set[int]] = {}
        for row in catalog.itertuples(index=False):
            key = ExchangeFrameBuilder.flow_id_for((row.database, row.code))
            labels[key] = {
                "database": str(row.database),
                "code": str(row.code),
                "name": str(row.name),
            }
            name = str(row.name).strip().casefold()
            categories = tuple(row.categories) if row.categories is not None else ()
            if categories and str(categories[0]).casefold() == "air" and name in self.CORE_GASES:
                required.add(key)
            if self._is_emission_to_air(categories):
                air_flows_by_name.setdefault(name, set()).add(key)
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
            cf_by_flow = {
                int(key): float(value)
                for key, value in zip(factors["flow_id"], values, strict=True)
            }
            covered = set(cf_by_flow)
            missing = sorted(used - covered)
            climate = any(str(part).casefold() == "climate change" for part in method)
            without_factor = sorted((used & required) - covered) if climate else []
            zero_by_definition = [
                key
                for key in without_factor
                if self._method_defines_zero(labels[key]["name"], cf_by_flow, air_flows_by_name)
            ]
            missing_required = [key for key in without_factor if key not in zero_by_definition]
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
                    "zero_by_method_definition": [labels[key] for key in zero_by_definition],
                    "missing_core_gases": [labels[key] for key in missing_required],
                }
            )
        return {
            "status": "failed" if failures else "passed",
            "policy": (
                "Core CO2/CH4/N2O air emissions require source CF rows in climate-change "
                "methods; no inferred CFs. A species the method defines as zero is accepted "
                "without a row of its own only while the method's table lists it with "
                "explicit zero factors."
            ),
            "scope": "Coverage only; other gases and non-climate omissions require method-specific review.",
            "unlabelled_emitted_flow_ids": missing_labels,
            "failed_methods": failures,
            "methods": methods,
        }

    def _method_defines_zero(
        self,
        flow_name: str,
        cf_by_flow: Mapping[int, float],
        air_flows_by_name: Mapping[str, set[int]],
    ) -> bool:
        """True when the method's own table lists this species to air with only zero factors.

        Three things must hold: the policy names the species, the catalog knows the
        method's flows for it, and every one of them carries an explicit factor of 0 in
        this method. A method that omits the species, lists it incompletely or gives it
        a factor proves nothing, and the flow stays a missing factor.
        """
        species = self.ZERO_BY_METHOD_DEFINITION.get(flow_name.strip().casefold())
        method_flows = air_flows_by_name.get(species, set()) if species else set()
        return bool(method_flows) and all(cf_by_flow.get(key) == 0.0 for key in method_flows)

    @staticmethod
    def _is_emission_to_air(categories: Iterable[Any]) -> bool:
        """ecoinvent says ``("air", ...)``; the EF nomenclature ``("Emissions", "Emissions to air", ...)``."""
        names = [str(part).strip().casefold() for part in categories]
        return bool(names) and (names[0] == "air" or "emissions to air" in names)
