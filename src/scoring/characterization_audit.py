"""``CharacterizationAuditor`` — audit the final biosphere against the method CF rows.

This checks coverage, not factor correctness. It never invents a factor or resolves
unqualified methane. The core-gas policy covers CO2, CH4 and N2O emitted to air; for
every other flow the receipt gives each method's counts, for method-specific review.
The receipt holds counts and the named core gases, never the list of flow ids: most
emitted flows have no factor in most methods. For an emitted package that list can be
rebuilt from its ``ids.json`` and characterization matrices; a withheld candidate has
no package, and its receipt names the gases that withheld it.

``status`` is ``passed`` only if at least one headline climate method (a method key
with the part "climate change") was audited, it checked at least one emitted core-gas
flow, and nothing failed. It is ``not_audited``, with ``not_audited_reason``, when the
registry holds no such method or when no emitted flow was recognised as a core gas
(``core_gas_flows_checked`` is 0 in each climate method): a pass never means "nothing
to check". An AGRIBALYSE package always emits CO2 to air, so a count of 0 says the
catalog's names or compartments were not recognised, not that the inventory is clean.

A core gas that scores zero fails the audit, whether it has no factor row or a row that
says 0, with one exception that the method itself must prove: a species the method
defines as exactly zero. EF 3.1 lists biogenic CO2 emissions to air with a factor of 0,
but under its own flow ids; the ecoinvent biosphere flow "Carbon dioxide, non-fossil"
that the inventory emits has no row of its own. Such a flow is reported as zero by
method definition, apart from a missing factor, and only while the method's CF table
carries those explicit zero rows.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.matrix_builder import BuiltMatrix


@dataclass(frozen=True)
class CharacterizationAuditor:
    """Stateless. ``audit(...)`` returns the receipt; ``status`` is ``passed``, ``failed``
    or ``not_audited``."""

    CORE_GASES: ClassVar[frozenset[str]] = frozenset(
        {
            "carbon dioxide",
            "carbon dioxide, fossil",
            "carbon dioxide, non-fossil",
            "carbon dioxide, biogenic",
            "carbon dioxide, peat oxidation",
            # Land-use change: the ecoinvent 3.9.1 name, then the older biosphere3 one.
            "carbon dioxide, from soil or biomass stock",
            "carbon dioxide, land transformation",
            "methane",
            "methane, fossil",
            "methane, non-fossil",
            "methane, biogenic",
            "methane, peat oxidation",
            "methane, from soil or biomass stock",
            "methane, land transformation",
            "dinitrogen monoxide",
            "dinitrogen monoxide, peat oxidation",
            # The method's own flow list (``ef``): the matcher may link an emission there.
            "carbon dioxide (fossil)",
            "carbon dioxide (biogenic)",
            "carbon dioxide (land use change)",
            "methane (fossil)",
            "methane (biogenic)",
            "methane (land use change)",
            "nitrous oxide",
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

    #: Flow ids listed for emitted flows that have no catalog row. Expected to be none;
    #: the count is always exact, the sample is for finding the first of them.
    SAMPLE_SIZE: ClassVar[int] = 20

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
            if self._is_emission_to_air(row.categories):
                name = str(row.name).strip().casefold()
                air_flows_by_name.setdefault(name, set()).add(key)
                if name in self.CORE_GASES:
                    required.add(key)
        if len(labels) != len(catalog):
            # Two rows for one flow: its name, and so its verdict, would be whichever came last.
            raise ValueError(
                f"Duplicate biosphere catalog rows: {len(catalog) - len(labels)} "
                "(database, code) pairs appear more than once"
            )
        missing_labels = sorted(int(key) for key in used - labels.keys())
        methods, failures, n_climate = [], [], 0
        core_gas_flows = sorted(used & required)
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
            climate = any(str(part).strip().casefold() == "climate change" for part in method)
            n_climate += climate
            zero_by_definition, zero_factor, missing_required = [], [], []
            for key in core_gas_flows if climate else []:
                name, factor = labels[key]["name"], cf_by_flow.get(key)
                if factor is None:
                    if self._method_defines_zero(name, cf_by_flow, air_flows_by_name):
                        zero_by_definition.append(key)
                    else:
                        missing_required.append(key)
                elif factor == 0.0:
                    # A row that says 0 scores the gas zero as surely as no row. It is the
                    # method's word only for a species the policy names as method-defined zero.
                    (zero_by_definition if self._is_zero_species(name) else zero_factor).append(key)
            # Unlabelled rows could hide core gases; a climate audit cannot certify them.
            if climate and (missing_required or zero_factor or missing_labels):
                failures.append(list(method))
            entry: dict[str, Any] = {
                "method": list(method),
                "characterized_flows": len(used & covered),
                "explicit_zero_flows": sum(cf_by_flow[key] == 0.0 for key in used & covered),
                "uncharacterized_flows": len(used - covered),
            }
            if climate:
                # 0 means no emitted flow was recognised as a core gas: nothing was certified.
                entry["core_gas_flows_checked"] = len(core_gas_flows)
                # Bounded by the core-gas rows of the catalog, and named: these lists are
                # the evidence a reviewer needs. Only the first is compatible with a pass.
                entry["zero_by_method_definition"] = [labels[key] for key in zero_by_definition]
                entry["zero_factor_core_gases"] = [labels[key] for key in zero_factor]
                entry["missing_core_gases"] = [labels[key] for key in missing_required]
            methods.append(entry)
        if failures:
            status, reason = "failed", None
        elif not n_climate:
            status, reason = "not_audited", "the registry holds no headline climate-change method"
        elif not core_gas_flows:
            status = "not_audited"
            reason = (
                "no emitted flow was recognised as a core gas (CO2, CH4 or N2O to air); "
                "check the names and compartments of the biosphere catalog"
            )
        else:
            status, reason = "passed", None
        return {
            "status": status,
            "not_audited_reason": reason,
            "climate_methods_audited": n_climate,
            "policy": (
                "Core CO2/CH4/N2O air emissions require non-zero source CF rows in "
                "climate-change methods; no inferred CFs. Zero is accepted only for a "
                "species the method defines as zero: on the flow's own zero row, or without "
                "a row while the method's table lists the species with explicit zero factors."
            ),
            "scope": "Coverage only; other gases and non-climate omissions require method-specific review.",
            "emitted_flows": len(used),
            "unlabelled_emitted_flows": len(missing_labels),
            "unlabelled_emitted_flow_id_sample": missing_labels[: self.SAMPLE_SIZE],
            "failed_methods": failures,
            "methods": methods,
        }

    def write_receipt(self, audit: Mapping[str, Any], candidate_hash: str, path: Path) -> Path:
        """Write the audit of one scoring-package candidate to its own file, atomically.

        Written whether the audit passed or failed; ``candidate_content_hash`` says
        which candidate it describes. An audit that cannot be made and raises (an
        inconsistent catalog or factor table) writes no receipt, and the receipt of
        the earlier candidate stays.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(
            json.dumps(
                {"candidate_content_hash": candidate_hash, **audit}, indent=2, ensure_ascii=False
            )
        )
        os.replace(partial, path)
        return path

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

    def _is_zero_species(self, flow_name: str) -> bool:
        """True for a name the policy lists, in the inventory's or the method's nomenclature."""
        name = flow_name.strip().casefold()
        return (
            name in self.ZERO_BY_METHOD_DEFINITION
            or name in self.ZERO_BY_METHOD_DEFINITION.values()
        )

    @staticmethod
    def _is_emission_to_air(categories: Any) -> bool:
        """ecoinvent says ``("air", ...)``; the EF nomenclature ``("Emissions", "Emissions to air", ...)``.

        The catalog parquet hands the compartment over as a tuple, list or array; a
        missing one arrives as ``None`` or NaN and is not an emission to air.
        """
        if isinstance(categories, str):
            categories = (categories,)
        if not isinstance(categories, Iterable):
            return False
        names = [str(part).strip().casefold() for part in categories]
        return bool(names) and (names[0] == "air" or "emissions to air" in names)
