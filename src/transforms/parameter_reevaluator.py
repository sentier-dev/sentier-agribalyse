"""``ParameterReevaluator`` — re-evaluate exchange formulas under overrides.

Mirrors ``bw_simapro_csv``'s process-level resolution
(``Process.resolve_local_parameters``): input-parameter amounts feed
``bw2parameters.ParameterSet`` for the calculated parameters, then a
``bw2parameters.Interpreter`` evaluates each exchange formula. Run with no
overrides it reproduces the baked amounts exactly — the invariant the
integration tests pin.

All AGB 3.2 parameters are process-local (the database/project blocks in the
export are empty), so the blast radius of an override is the set of processes
that define the parameter. Only **input** parameters are editable; calculated
parameters are formula-derived and refuse overrides.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from bw2parameters import Interpreter, ParameterSet

from core.logging import Logging

if TYPE_CHECKING:
    from transforms.parameter_overrides import ParameterOverride

GLOBAL_SCOPE = "*"


class UnknownParameterError(ValueError):
    """Raised when an override names a parameter that does not exist."""


class UnknownProcessError(ValueError):
    """Raised when an override targets a process that does not define the parameter."""


class CalculatedParameterOverrideError(ValueError):
    """Raised when an override targets a calculated (formula-derived) parameter."""


@dataclass(frozen=True)
class ExchangeChange:
    """One exchange amount that changed under the overrides."""

    process_code: str
    exchange_index: int
    exchange_name: str
    old_amount: float
    new_amount: float


@dataclass(frozen=True)
class ReevaluationResult:
    data: list[dict]
    changes: tuple[ExchangeChange, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ParameterReevaluator:
    """Re-evaluate exchange amounts for processes affected by overrides.

    ``parameters`` are the rows of ``ParsedSimaProCsv.parameters``.
    """

    parameters: list[dict]

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Indexes

    @cached_property
    def _input_by_process(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for row in self.parameters:
            if row["kind"] == "input":
                out.setdefault(row["process_code"], []).append(row)
        return out

    @cached_property
    def _calc_by_process(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for row in self.parameters:
            if row["kind"] == "calculated":
                out.setdefault(row["process_code"], []).append(row)
        return out

    @cached_property
    def _norm_by_user_name(self) -> dict[str, str]:
        """lower(original_name) and lower(name) → normalized ``name``."""
        out: dict[str, str] = {}
        for row in self.parameters:
            out.setdefault((row["original_name"] or "").lower(), row["name"])
            out.setdefault((row["name"] or "").lower(), row["name"])
        return out

    @cached_property
    def _kinds_by_norm(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for row in self.parameters:
            out.setdefault(row["name"], set()).add(row["kind"])
        return out

    @cached_property
    def _defining_processes(self) -> dict[str, list[str]]:
        """normalized name → process codes whose *input* block defines it."""
        out: dict[str, list[str]] = {}
        for row in self.parameters:
            if row["kind"] == "input":
                out.setdefault(row["name"], []).append(row["process_code"])
        return out

    @cached_property
    def _mf_children_by_parent(self) -> dict[str, list[str]]:
        """multifunctional parent code → allocated child codes.

        Child parameter rows carry ``mf_parent_code`` (see
        ``ProcessParameterExtractor``); a process-scoped override on the
        parent must expand to its children, whose exchanges carry the
        formulas that actually become matrix columns.
        """
        out: dict[str, list[str]] = {}
        for row in self.parameters:
            parent = row.get("mf_parent_code")
            if parent and row["process_code"] not in out.setdefault(parent, []):
                out[parent].append(row["process_code"])
        return out

    # ------------------------------------------------------------------
    # Name resolution

    def resolve_name(self, user_name: str) -> str:
        """Resolve a user-facing name to the normalized formula name.

        Raises ``UnknownParameterError`` (with closest matches) or
        ``CalculatedParameterOverrideError`` (with an example formula).
        """
        norm = self._norm_by_user_name.get(user_name.lower())
        if norm is None:
            suggestions = difflib.get_close_matches(
                user_name.lower(), list(self._norm_by_user_name), n=3, cutoff=0.6
            )
            display = sorted(
                {
                    row["original_name"]
                    for row in self.parameters
                    if (row["original_name"] or "").lower() in suggestions
                    or (row["name"] or "").lower() in suggestions
                }
            )
            hint = f" Did you mean: {', '.join(display)}?" if display else ""
            raise UnknownParameterError(
                f"Unknown parameter {user_name!r}.{hint} "
                f"Use dds-list-parameters to browse the {len(self._defining_processes)} "
                f"editable input-parameter names."
            )
        kinds = self._kinds_by_norm.get(norm, set())
        if "input" not in kinds:
            example = next(
                (
                    row["formula"]
                    for row in self.parameters
                    if row["name"] == norm and row["kind"] == "calculated"
                ),
                None,
            )
            raise CalculatedParameterOverrideError(
                f"{user_name!r} is a calculated parameter (formula: {example}). "
                f"Only input parameters can be overridden — override the input "
                f"parameters its formula references instead."
            )
        return norm

    def defining_processes(self, norm_name: str) -> list[str]:
        return list(self._defining_processes.get(norm_name, []))

    # ------------------------------------------------------------------
    # Re-evaluation

    def plan(
        self, overrides: list[ParameterOverride]
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
        """Map overrides to per-process environments: ``{code: {norm_name: value}}``.

        Precedence: a process-specific override beats a ``*`` override for
        the same parameter. Also returns ``{user_name: target codes}`` per
        override for effect reporting.
        """
        per_process: dict[str, dict[str, float]] = {}
        targets_by_name: dict[str, list[str]] = {}
        # Precedence tiers, later tiers overwrite earlier ones regardless of
        # row order in the file:
        #   0. global ("*")
        #   1. multifunctional-parent expansion onto its children
        #   2. direct process-specific scope (incl. the parent itself)
        for tier in (0, 1, 2):
            for ov in overrides:
                is_specific = ov.scope != GLOBAL_SCOPE
                norm = self.resolve_name(ov.parameter_name)
                defining = self._defining_processes.get(norm, [])
                if is_specific and ov.scope not in set(defining):
                    raise UnknownProcessError(
                        f"Process {ov.scope!r} does not define input parameter "
                        f"{ov.parameter_name!r} ({len(defining)} processes do; "
                        f"use dds-list-parameters --name-like to find them)."
                    )
                if tier == 0 and not is_specific:
                    codes = defining
                elif tier == 1 and is_specific:
                    # A multifunctional parent's allocated children inherit
                    # the override — their (child-coded) exchanges are what
                    # survive into the matrices.
                    codes = [
                        c
                        for c in self._mf_children_by_parent.get(ov.scope, [])
                        if c in set(defining)
                    ]
                elif tier == 2 and is_specific:
                    codes = [ov.scope]
                else:
                    continue
                for code in codes:
                    per_process.setdefault(code, {})[norm] = ov.value
                targets_by_name.setdefault(ov.parameter_name, [])
                targets_by_name[ov.parameter_name].extend(
                    c for c in codes if c not in targets_by_name[ov.parameter_name]
                )
        return per_process, targets_by_name

    def reevaluate(
        self,
        data: list[dict],
        overrides: list[ParameterOverride],
    ) -> ReevaluationResult:
        """Return a new dataset list with affected exchange amounts recomputed.

        Absolute mode — for parse-level data: the formula's value IS the
        new amount. Use ``ratio_patch`` for post-transform (linked) data.
        """
        return self._apply(data, overrides, ratio=False)

    def ratio_patch(
        self,
        data: list[dict],
        overrides: list[ParameterOverride],
    ) -> ReevaluationResult:
        """Ratio mode — for post-transform (linked) data.

        Between the parse and the exchange frame, transforms rescale
        amounts multiplicatively (unit conversions, MJ→kWh, ...), so the
        stored amount is no longer the formula's value. Scaling by
        ``new_eval / baseline_eval`` preserves whatever linear factor was
        applied. Exchanges whose baseline evaluation is 0 but new value
        isn't cannot be ratio-patched (unknown scale) — they surface as
        warnings and keep their baseline amount.
        """
        return self._apply(data, overrides, ratio=True)

    def _apply(
        self,
        data: list[dict],
        overrides: list[ParameterOverride],
        ratio: bool,
    ) -> ReevaluationResult:
        per_process, targets_by_name = self.plan(overrides)
        if not per_process:
            return ReevaluationResult(data=data, changes=(), warnings=())

        envs = {code: self._environment(code, values) for code, values in per_process.items()}
        base_envs = {code: self._environment(code, {}) for code in per_process} if ratio else {}
        # Only formulas that reference an overridden parameter (directly or
        # through the calculated-parameter chain) may be touched.
        # Parameter-free formulas — and edges bw_simapro_csv post-fixed
        # after evaluation (e.g. zeroed waste-model production rows) —
        # must keep their baked amounts.
        influenced = {
            code: self._influenced_names(code, set(values)) for code, values in per_process.items()
        }
        new_data: list[dict] = []
        changes: list[ExchangeChange] = []
        unpatchable: list[str] = []
        for ds in data:
            env = envs.get(ds.get("code"))
            if env is None or not ds.get("exchanges"):
                new_data.append(ds)
                continue
            interpreter = Interpreter()
            interpreter.add_symbols(env)
            base_interpreter = None
            if ratio:
                base_interpreter = Interpreter()
                base_interpreter.add_symbols(base_envs[ds["code"]])
            names = influenced[ds["code"]]
            exchanges: list[dict] = []
            for idx, exc in enumerate(ds["exchanges"]):
                formula = exc.get("formula")
                if not formula or not self._references(formula, names):
                    exchanges.append(exc)
                    continue
                try:
                    evaluated = float(interpreter(formula))
                except ZeroDivisionError as zde:
                    raise ValueError(
                        f"Override produces a non-finite amount on "
                        f"{ds['code']}[{idx}] {exc.get('name', '')!r} "
                        f"(formula: {formula}) — division by an overridden "
                        f"zero. No amounts were changed."
                    ) from zde
                if ratio:
                    baseline = float(base_interpreter(formula))
                    if baseline == 0.0:
                        if evaluated != 0.0:
                            unpatchable.append(
                                f"{ds['code']}[{idx}] {exc.get('name', '')!r}: baseline "
                                f"evaluates to 0, cannot derive the transform scale — "
                                f"amount left unchanged (use the full rebuild path)."
                            )
                        exchanges.append(exc)
                        continue
                    new_amount = float(exc.get("amount") or 0.0) * (evaluated / baseline)
                else:
                    new_amount = evaluated
                if not math.isfinite(new_amount):
                    # Overriding a denominator parameter to 0 (or a hand-
                    # edited nan/inf) must fail HERE, loudly — the pipeline's
                    # NaN guard runs before this stage and would let it
                    # corrupt the emitted scoring package silently.
                    raise ValueError(
                        f"Override produces a non-finite amount on "
                        f"{ds['code']}[{idx}] {exc.get('name', '')!r} "
                        f"(formula: {formula}) — check for division by an "
                        f"overridden zero. No amounts were changed."
                    )
                if new_amount == exc.get("amount"):
                    exchanges.append(exc)
                    continue
                changes.append(
                    ExchangeChange(
                        process_code=ds["code"],
                        exchange_index=idx,
                        exchange_name=exc.get("name", ""),
                        old_amount=exc.get("amount"),
                        new_amount=new_amount,
                    )
                )
                exchanges.append({**exc, "amount": new_amount})
            new_data.append({**ds, "exchanges": exchanges})

        zero_productions = tuple(
            f"Override zeroes the production amount of "
            f"{c.process_code}[{c.exchange_index}] {c.exchange_name!r} — this "
            f"adds a zero diagonal to the technosphere; the scipy solver may "
            f"reject the matrix (use --solver pardiso)."
            for c in changes
            if c.new_amount == 0.0 and self._is_production(data, c)
        )
        warnings = (
            self._unused_warnings(data, targets_by_name) + tuple(unpatchable) + zero_productions
        )
        self._log.info(
            "parameters.reevaluated",
            mode="ratio" if ratio else "absolute",
            processes=len(per_process),
            changed_exchanges=len(changes),
            unpatchable=len(unpatchable),
        )
        return ReevaluationResult(data=new_data, changes=tuple(changes), warnings=warnings)

    # ------------------------------------------------------------------
    # Internals

    @staticmethod
    def _is_production(data: list[dict], change: ExchangeChange) -> bool:
        for ds in data:
            if ds.get("code") == change.process_code:
                exchanges = ds.get("exchanges") or []
                if change.exchange_index < len(exchanges):
                    kind = exchanges[change.exchange_index].get("type", "")
                    return "production" in str(kind)
        return False

    @staticmethod
    def _references(formula: str, names: Iterable[str]) -> bool:
        """True when ``formula`` references any of ``names`` (word-boundary).

        The single load-bearing "does this formula use parameter n"
        predicate — every reference check must go through it.
        """
        return any(re.search(rf"\b{re.escape(n)}\b", formula) for n in names)

    def _environment(self, code: str, override_values: dict[str, float]) -> dict[str, float]:
        """Input amounts (overridden) + calculated amounts, in dependency order."""
        env: dict[str, float] = {}
        for row in self._input_by_process.get(code, []):
            if not isinstance(row["amount"], int | float):
                # Fail with the parameter's name, not a cryptic TypeError
                # deep inside bw2parameters.
                raise ValueError(
                    f"Input parameter {row['original_name']!r} on process "
                    f"{code!r} has a non-numeric amount "
                    f"({row['amount']!r}) — cannot build the evaluation "
                    f"environment."
                )
            env[row["name"]] = row["amount"]
        env.update(override_values)
        calc_rows = self._calc_by_process.get(code, [])
        if calc_rows:
            calc = {row["name"]: {"formula": row["formula"]} for row in calc_rows}
            ParameterSet(calc, global_params=dict(env)).evaluate_and_set_amount_field()
            env.update({name: spec["amount"] for name, spec in calc.items()})
        return env

    def _influenced_names(self, code: str, norm_names: set[str]) -> set[str]:
        """Transitive closure of ``norm_names`` through the process's
        calculated-parameter formulas: a calc param whose formula references
        an influenced name becomes influenced itself."""
        influenced = set(norm_names)
        calc_rows = self._calc_by_process.get(code, [])
        grew = True
        while grew:
            grew = False
            for row in calc_rows:
                if row["name"] in influenced or not row.get("formula"):
                    continue
                if self._references(row["formula"], influenced):
                    influenced.add(row["name"])
                    grew = True
        return influenced

    def _unused_warnings(
        self, data: list[dict], targets_by_name: dict[str, list[str]]
    ) -> tuple[str, ...]:
        """Warn for overrides whose parameter feeds no formula (transitively).

        A parameter is "used" in a process when its normalized name — or a
        calculated parameter transitively derived from it — appears in an
        exchange formula of that process. Most ``DQI_*`` rows are pure
        metadata and land here.
        """
        ds_by_code = {ds.get("code"): ds for ds in data if ds.get("exchanges")}
        warnings: list[str] = []
        for user_name, codes in targets_by_name.items():
            norm = self.resolve_name(user_name)
            used = False
            for code in codes:
                influenced = self._influenced_names(code, {norm})
                ds = ds_by_code.get(code)
                if ds is None:
                    continue
                for exc in ds["exchanges"]:
                    formula = exc.get("formula")
                    if formula and self._references(formula, influenced):
                        used = True
                        break
                if used:
                    break
            if not used:
                warnings.append(
                    f"Parameter {user_name!r} is not referenced by any formula in "
                    f"its target process(es) — the override has no score effect."
                )
        return tuple(warnings)
