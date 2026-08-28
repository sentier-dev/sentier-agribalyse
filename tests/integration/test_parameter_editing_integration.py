"""Integration tests: parameter editing against a real AGB-derived CSV.

``tests/fixtures/agb_mini_params.csv`` is the authentic AGB 3.2 export
header plus one real process (canned tuna, ``EI3CQUNI000025017103472``)
carrying 22 input parameters and 2 calculated parameters — extracted
verbatim from ``source/AGB32_final.CSV``. Parsing it through the real
``bw_simapro_csv`` pins:

1. extraction lifts the process-local definitions the brightway dicts drop;
2. re-evaluation with no overrides reproduces the baked amounts (the
   engine is the same machinery that baked them);
3. an override propagates through the calculated-parameter cascade into
   the exchange amounts with hand-computed expected values.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from transforms.parameter_extraction import ProcessParameterExtractor
from transforms.parameter_overrides import ParameterOverride
from transforms.parameter_reevaluator import ParameterReevaluator

FIXTURE = Path(__file__).parents[1] / "fixtures" / "agb_mini_params.csv"
PROCESS_CODE = "EI3CQUNI000025017103472"


@pytest.fixture(scope="module")
def real_bw_simapro_csv():
    """Swap the session-wide ``bw_simapro_csv`` stub for the real package.

    ``tests/fixtures/bw_stubs.py`` installs a permissive stub before any
    test imports; this suite is the one place that needs the genuine
    parser. Restored on teardown so the unit suite stays hermetic.
    """
    import importlib
    import sys

    # The real parse pulls a chain of stubbed packages
    # (bw_simapro_csv → multifunctional → bw2data ...), so evict every
    # ``_dds_test_stub``-marked module, not just the parser.
    saved = {
        k: v
        for k, v in sys.modules.items()
        if getattr(v, "_dds_test_stub", False)
        or getattr(sys.modules.get(k.split(".")[0]), "_dds_test_stub", False)
    }
    for key in list(saved):
        del sys.modules[key]
    real = importlib.import_module("bw_simapro_csv")
    yield real
    for key in list(sys.modules):
        root = key.split(".")[0]
        if root in {s.split(".")[0] for s in saved}:
            del sys.modules[key]
    sys.modules.update(saved)


@pytest.fixture(scope="module")
def parsed(real_bw_simapro_csv):
    spcsv = real_bw_simapro_csv.SimaProCSV(
        path_or_stream=StringIO(FIXTURE.read_text(encoding="latin-1")),
        database_name="agb-test",
        stderr_logs=False,
        write_logs=False,
    )
    bw = spcsv.to_brightway(separate_products=True, shorten_names=True)
    data = list(bw["processes"]) + list(bw["products"])
    parameters = ProcessParameterExtractor().extract(spcsv)
    return data, parameters


@pytest.mark.integration
class TestExtraction:
    def test_extracts_input_and_calculated_definitions(self, parsed):
        _, parameters = parsed
        kinds = {row["kind"] for row in parameters}
        assert kinds == {"input", "calculated"}
        assert all(row["process_code"] == PROCESS_CODE for row in parameters)
        by_original = {row["original_name"]: row for row in parameters}
        assert by_original["ratio_ingredient1"]["amount"] == pytest.approx(0.67)
        assert by_original["ratio_ingredient1"]["name"] == "SP_RATIO_INGREDIENT1"
        assert "SP_RATIO_INGREDIENT1" in by_original["fish_losses_kg"]["formula"]

    def test_exchange_formulas_survive_the_parse(self, parsed):
        data, _ = parsed
        process = next(ds for ds in data if ds.get("code") == PROCESS_CODE)
        with_formula = [e for e in process["exchanges"] if e.get("formula")]
        assert len(with_formula) == 7


@pytest.mark.integration
class TestReevaluationEquivalence:
    def test_no_overrides_reproduces_baked_amounts(self, parsed):
        data, parameters = parsed
        result = ParameterReevaluator(parameters=parameters).reevaluate(data, [])
        assert result.changes == ()

    def test_unchanged_value_reproduces_baked_amounts(self, parsed):
        """Overriding with the ORIGINAL value must change nothing — the
        engine is bit-equivalent to the machinery that baked the amounts."""
        data, parameters = parsed
        result = ParameterReevaluator(parameters=parameters).reevaluate(
            data, [ParameterOverride("ratio_ingredient1", "*", 0.67)]
        )
        assert result.changes == ()


@pytest.mark.integration
class TestOverrideEffect:
    def test_override_cascades_into_exchanges(self, parsed):
        data, parameters = parsed
        result = ParameterReevaluator(parameters=parameters).reevaluate(
            data, [ParameterOverride("ratio_ingredient1", "*", 0.5)]
        )
        by_name = {c.exchange_name: c for c in result.changes}
        # tuna input = ratio * yield_filling(1.02) * yield_fishpreparation(1.76)
        tuna = next(v for k, v in by_name.items() if "tuna" in k.lower())
        assert tuna.new_amount == pytest.approx(0.5 * 1.02 * 1.76)
        # fish_losses_kg = ratio*1.02*1.76 - ratio  (calculated parameter)
        losses = next(v for k, v in by_name.items() if "Food waste" in k)
        assert losses.new_amount == pytest.approx(0.5 * 1.02 * 1.76 - 0.5)
        # Baseline untouched.
        process = next(ds for ds in data if ds.get("code") == PROCESS_CODE)
        tuna_baseline = next(
            e for e in process["exchanges"] if "tuna" in str(e.get("name", "")).lower()
        )
        assert tuna_baseline["amount"] == pytest.approx(1.202784)

    def test_dqi_override_warns_and_changes_nothing(self, parsed):
        data, parameters = parsed
        result = ParameterReevaluator(parameters=parameters).reevaluate(
            data, [ParameterOverride("DQI_P_Processing", "*", 5.0)]
        )
        assert result.changes == ()
        assert any("no score effect" in w for w in result.warnings)
