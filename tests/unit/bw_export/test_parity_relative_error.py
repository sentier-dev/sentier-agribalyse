"""The error measure of the Brightway export's parity check."""

import pytest

pytest.importorskip("bw_processing", reason="needs the [bw] extra")
pytest.importorskip("bw2calc", reason="needs the [bw] extra")

from bw_export.parity_verifier import ParityVerifier


class TestRelativeError:
    def test_a_score_without_cancellation_is_measured_against_itself(self):
        verifier = ParityVerifier()

        assert verifier.relative_error(1.0 + 1e-6, 1.0, gross=1.0) == pytest.approx(1e-6)

    def test_a_rounding_residue_against_an_exact_zero_is_not_an_error_of_one(self):
        # Two terms of 40 and -40: one side sums them to 0, the other leaves 1e-15.
        verifier = ParityVerifier()

        assert verifier.relative_error(0.0, 1e-15, gross=80.0) < verifier.tolerance

    def test_a_small_score_that_is_wrong_still_fails(self):
        # A real score of 1e-3 made of terms of that size, missed by 1 %.
        verifier = ParityVerifier()

        assert verifier.relative_error(0.99e-3, 1e-3, gross=1e-3) == pytest.approx(0.01)

    def test_a_dropped_term_fails_even_when_zero_is_expected(self):
        # Expected 0 = 40 - 40; the export lost one term and scores 40.
        verifier = ParityVerifier()

        assert verifier.relative_error(40.0, 0.0, gross=40.0) > 1.0

    def test_a_score_without_any_term_keeps_the_absolute_measure(self):
        verifier = ParityVerifier()

        assert verifier.relative_error(0.0, 0.0, gross=0.0) == 0.0
        assert verifier.relative_error(2e-3, 0.0, gross=0.0) == pytest.approx(2e-3)
