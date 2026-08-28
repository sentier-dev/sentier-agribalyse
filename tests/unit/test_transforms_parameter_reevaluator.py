"""Unit tests for ``ParameterReevaluator``.

Pins the resolution contract: input amounts (with overrides) feed the
calculated-parameter cascade (``bw2parameters.ParameterSet``), then exchange
formulas re-evaluate through ``bw2parameters.Interpreter`` — the exact
machinery ``bw_simapro_csv`` used to bake the amounts, so a no-override run
must be a no-op.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from transforms.parameter_overrides import ParameterOverride
from transforms.parameter_reevaluator import (
    CalculatedParameterOverrideError,
    ParameterReevaluator,
    UnknownParameterError,
    UnknownProcessError,
)


def _param(code, kind, name, original, amount=None, formula=None):
    return {
        "process_code": code,
        "kind": kind,
        "name": name,
        "original_name": original,
        "amount": amount,
        "formula": formula,
        "comment": "",
    }


@pytest.fixture
def reevaluator() -> ParameterReevaluator:
    return ParameterReevaluator(
        parameters=[
            _param("PROC1", "input", "SP_RATIO", "ratio", amount=0.5),
            _param("PROC1", "input", "SP_UNUSED_DQI", "DQI_P_Thing", amount=1.0),
            _param(
                "PROC1",
                "calculated",
                "SP_LOSSES",
                "losses",
                amount=0.25,
                formula="(SP_RATIO * SP_RATIO)",
            ),
            _param("PROC2", "input", "SP_RATIO", "ratio", amount=0.8),
        ]
    )


@pytest.fixture
def data() -> list[dict]:
    return [
        {
            "code": "PROC1",
            "type": "process",
            "exchanges": [
                {
                    "name": "direct",
                    "amount": 1.0,
                    "formula": "(SP_RATIO * 2)",
                    "type": "technosphere",
                },
                {"name": "derived", "amount": 0.25, "formula": "SP_LOSSES", "type": "technosphere"},
                {"name": "constant", "amount": 7.0, "type": "technosphere"},
            ],
        },
        {
            "code": "PROC2",
            "type": "process",
            "exchanges": [
                {
                    "name": "direct2",
                    "amount": 1.6,
                    "formula": "(SP_RATIO * 2)",
                    "type": "technosphere",
                },
            ],
        },
        {"code": "PROD1", "type": "product"},
    ]


class TestReevaluate:
    def test_no_overrides_is_noop(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [])
        assert result.changes == ()
        assert result.data is data

    def test_reproduces_baked_amounts_when_value_unchanged(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [ParameterOverride("ratio", "PROC1", 0.5)])
        assert result.changes == ()

    def test_global_override_hits_all_defining_processes(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [ParameterOverride("ratio", "*", 0.1)])
        by_process = {c.process_code for c in result.changes}
        assert by_process == {"PROC1", "PROC2"}

    def test_process_scope_limits_blast_radius(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [ParameterOverride("ratio", "PROC2", 0.1)])
        assert {c.process_code for c in result.changes} == {"PROC2"}
        assert result.data[0] is data[0]  # PROC1 dataset untouched (same object)

    def test_calculated_cascade_propagates(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [ParameterOverride("ratio", "PROC1", 0.3)])
        by_name = {c.exchange_name: c for c in result.changes}
        assert by_name["direct"].new_amount == pytest.approx(0.6)
        assert by_name["derived"].new_amount == pytest.approx(0.09)  # 0.3**2

    def test_specific_scope_beats_global(self, reevaluator, data):
        result = reevaluator.reevaluate(
            data,
            [
                ParameterOverride("ratio", "*", 0.1),
                ParameterOverride("ratio", "PROC1", 0.2),
            ],
        )
        by_key = {(c.process_code, c.exchange_name): c for c in result.changes}
        assert by_key[("PROC1", "direct")].new_amount == pytest.approx(0.4)
        assert by_key[("PROC2", "direct2")].new_amount == pytest.approx(0.2)

    def test_original_data_never_mutated(self, reevaluator, data):
        reevaluator.reevaluate(data, [ParameterOverride("ratio", "*", 0.0)])
        assert data[0]["exchanges"][0]["amount"] == 1.0

    def test_matching_is_case_insensitive_on_original_name(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [ParameterOverride("RATIO", "PROC1", 0.3)])
        assert len(result.changes) == 2

    def test_normalized_name_also_accepted(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [ParameterOverride("SP_RATIO", "PROC1", 0.3)])
        assert len(result.changes) == 2

    def test_unused_parameter_warns_no_score_effect(self, reevaluator, data):
        result = reevaluator.reevaluate(data, [ParameterOverride("DQI_P_Thing", "*", 5.0)])
        assert result.changes == ()
        assert len(result.warnings) == 1
        assert "no score effect" in result.warnings[0]


class TestPostFixedEdgesUntouched:
    """Edges bw_simapro_csv post-fixed after formula evaluation must keep
    their baked amounts (the 2026-08-18 sweep finding: a zeroed waste-model
    production row whose parameter-free formula still evaluates to 12.23)."""

    _params: ClassVar[list] = [
        _param("PROC1", "input", "SP_RATIO", "ratio", amount=0.97),
    ]
    _data: ClassVar[list] = [
        {
            "code": "PROC1",
            "type": "process",
            "exchanges": [
                # References the parameter — fair game.
                {"name": "steel", "amount": 0.0105, "formula": "(0.35 * (1 - SP_RATIO))"},
                # Parameter-free formula, post-fixed to 0 — must stay 0.
                {"name": "self-production", "amount": 0.0, "formula": "(1034 * 0.0118)"},
            ],
        },
    ]

    def test_parameter_free_formula_never_touched_absolute(self):
        reev = ParameterReevaluator(parameters=self._params)
        result = reev.reevaluate(self._data, [ParameterOverride("ratio", "*", 0.5)])
        assert [c.exchange_name for c in result.changes] == ["steel"]
        assert result.data[0]["exchanges"][1]["amount"] == 0.0

    def test_parameter_free_formula_never_touched_ratio(self):
        reev = ParameterReevaluator(parameters=self._params)
        result = reev.ratio_patch(self._data, [ParameterOverride("ratio", "*", 0.5)])
        assert [c.exchange_name for c in result.changes] == ["steel"]
        assert result.data[0]["exchanges"][1]["amount"] == 0.0

    def test_zeroed_edge_referencing_param_stays_zero_in_ratio_mode(self):
        data = [
            {
                "code": "PROC1",
                "exchanges": [
                    {"name": "fixed-up", "amount": 0.0, "formula": "(2 * SP_RATIO)"},
                ],
            }
        ]
        reev = ParameterReevaluator(parameters=self._params)
        result = reev.ratio_patch(data, [ParameterOverride("ratio", "*", 0.5)])
        # 0 * (new/old) = 0 — the fix-up survives; no change reported.
        assert result.changes == ()
        assert result.data[0]["exchanges"][0]["amount"] == 0.0


class TestDefiningProcesses:
    def test_lists_input_defining_process_codes(self, reevaluator):
        assert reevaluator.defining_processes("SP_RATIO") == ["PROC1", "PROC2"]
        assert reevaluator.defining_processes("SP_GHOST") == []


class TestMultifunctionalChildScope:
    """A process-scoped override on a multifunctional PARENT must expand to
    its allocated readonly_process children — the child-coded exchanges are
    what survive into the matrices (2026-08-18 red-team finding)."""

    _params: ClassVar[list] = [
        _param("PARENT1", "input", "SP_Q", "Q", amount=10.0),
        {**_param("CHILD1", "input", "SP_Q", "Q", amount=10.0), "mf_parent_code": "PARENT1"},
        _param("OTHER", "input", "SP_Q", "Q", amount=10.0),
    ]
    _data: ClassVar[list] = [
        {
            "code": "PARENT1",
            "exchanges": [{"name": "p-edge", "amount": 20.0, "formula": "(SP_Q * 2)"}],
        },
        {
            "code": "CHILD1",
            "exchanges": [{"name": "c-edge", "amount": 5.0, "formula": "(SP_Q / 2)"}],
        },
        {
            "code": "OTHER",
            "exchanges": [{"name": "o-edge", "amount": 20.0, "formula": "(SP_Q * 2)"}],
        },
    ]

    def test_parent_scope_expands_to_children_only(self):
        reev = ParameterReevaluator(parameters=self._params)
        result = reev.reevaluate(self._data, [ParameterOverride("Q", "PARENT1", 20.0)])
        touched = {c.process_code for c in result.changes}
        assert touched == {"PARENT1", "CHILD1"}  # OTHER untouched
        by_code = {c.process_code: c for c in result.changes}
        assert by_code["CHILD1"].new_amount == pytest.approx(10.0)  # 20/2

    def test_child_scope_stays_child_only(self):
        reev = ParameterReevaluator(parameters=self._params)
        result = reev.reevaluate(self._data, [ParameterOverride("Q", "CHILD1", 20.0)])
        assert {c.process_code for c in result.changes} == {"CHILD1"}

    def test_direct_child_override_beats_parent_expansion_any_row_order(self):
        """Precedence tiers: direct child scope > parent expansion > global,
        regardless of override row order (2026-08-18 adversarial F5)."""
        reev = ParameterReevaluator(parameters=self._params)
        for order in (
            [ParameterOverride("Q", "PARENT1", 20.0), ParameterOverride("Q", "CHILD1", 40.0)],
            [ParameterOverride("Q", "CHILD1", 40.0), ParameterOverride("Q", "PARENT1", 20.0)],
        ):
            result = reev.reevaluate(self._data, order)
            by_code = {c.process_code: c for c in result.changes}
            assert by_code["CHILD1"].new_amount == pytest.approx(20.0)  # 40/2
            assert by_code["PARENT1"].new_amount == pytest.approx(40.0)  # 20*2

    def test_parent_expansion_beats_global(self):
        reev = ParameterReevaluator(parameters=self._params)
        result = reev.reevaluate(
            self._data,
            [ParameterOverride("Q", "*", 30.0), ParameterOverride("Q", "PARENT1", 20.0)],
        )
        by_code = {c.process_code: c for c in result.changes}
        assert by_code["CHILD1"].new_amount == pytest.approx(10.0)  # parent 20/2, not 30/2
        assert by_code["OTHER"].new_amount == pytest.approx(60.0)  # global 30*2


class TestNonFiniteGuard:
    """Overriding a denominator to 0 must fail loudly, never emit inf/NaN
    into the scoring package (2026-08-18 adversarial F1)."""

    _params: ClassVar[list] = [
        _param("PROC1", "input", "SP_Q", "Q", amount=10.0),
    ]
    _data: ClassVar[list] = [
        {
            "code": "PROC1",
            "exchanges": [{"name": "per-unit", "amount": 5.0, "formula": "(50 / SP_Q)"}],
        },
    ]

    def test_zero_denominator_override_raises(self):
        reev = ParameterReevaluator(parameters=self._params)
        with pytest.raises(ValueError, match="non-finite"):
            reev.reevaluate(self._data, [ParameterOverride("Q", "*", 0.0)])

    def test_zero_denominator_override_raises_in_ratio_mode(self):
        reev = ParameterReevaluator(parameters=self._params)
        with pytest.raises(ValueError, match="non-finite"):
            reev.ratio_patch(self._data, [ParameterOverride("Q", "*", 0.0)])


class TestCrashQualityEdges:
    def test_none_input_amount_fails_with_parameter_name(self):
        """(2026-08-18 adversarial F10)"""
        params = [_param("PROC1", "input", "SP_Q", "Q", amount=None)]
        data = [{"code": "PROC1", "exchanges": [{"name": "e", "amount": 1.0, "formula": "SP_Q"}]}]
        reev = ParameterReevaluator(parameters=params)
        with pytest.raises(ValueError, match=r"'Q'.*non-numeric"):
            reev.reevaluate(data, [ParameterOverride("Q", "*", 2.0)])

    def test_zeroing_a_production_amount_warns_about_zero_diagonal(self):
        params = [_param("PROC1", "input", "SP_Q", "Q", amount=2.0)]
        data = [
            {
                "code": "PROC1",
                "exchanges": [
                    {"name": "prod", "amount": 2.0, "formula": "SP_Q", "type": "production"},
                ],
            }
        ]
        reev = ParameterReevaluator(parameters=params)
        result = reev.reevaluate(data, [ParameterOverride("Q", "*", 0.0)])
        assert result.data[0]["exchanges"][0]["amount"] == 0.0
        assert any("zero diagonal" in w for w in result.warnings)


class TestRatioZeroToZero:
    def test_zero_baseline_zero_new_keeps_amount_silently(self):
        """Baseline AND overridden evaluation are both 0 → the amount is
        kept with no change and no unpatchable warning (0 → 0 is exact)."""
        params = [
            _param("PROC1", "input", "SP_X", "x", amount=1.0),
            _param("PROC1", "input", "SP_ZERO", "zero", amount=0.0),
        ]
        data = [
            {
                "code": "PROC1",
                "exchanges": [
                    {"name": "e", "amount": 0.0, "formula": "(SP_X * SP_ZERO)"},
                ],
            }
        ]
        result = ParameterReevaluator(parameters=params).ratio_patch(
            data, [ParameterOverride("x", "*", 5.0)]
        )
        assert result.changes == ()
        assert result.warnings == ()
        assert result.data[0]["exchanges"][0]["amount"] == 0.0


class TestUnusedWarningsMissingDataset:
    def test_target_process_absent_from_data_still_warns_no_effect(self, reevaluator):
        """The parameter table can carry codes the dataset list lacks (e.g.
        after producer dedup) — the override must still surface as no-effect."""
        data = [{"code": "PROC2", "type": "process", "exchanges": []}]
        result = reevaluator.reevaluate(data, [ParameterOverride("ratio", "*", 0.1)])
        assert result.changes == ()
        assert any("no score effect" in w for w in result.warnings)


class TestValidation:
    def test_unknown_parameter_raises_with_suggestions(self, reevaluator, data):
        with pytest.raises(UnknownParameterError, match="Did you mean"):
            reevaluator.reevaluate(data, [ParameterOverride("ratioo", "*", 1.0)])

    def test_calculated_parameter_refused_with_formula(self, reevaluator, data):
        with pytest.raises(CalculatedParameterOverrideError, match=r"SP_RATIO \* SP_RATIO"):
            reevaluator.reevaluate(data, [ParameterOverride("losses", "*", 1.0)])

    def test_unknown_process_scope_raises(self, reevaluator, data):
        with pytest.raises(UnknownProcessError, match="does not define"):
            reevaluator.reevaluate(data, [ParameterOverride("ratio", "GHOST", 1.0)])
