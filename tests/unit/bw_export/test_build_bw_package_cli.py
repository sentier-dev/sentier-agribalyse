from pathlib import Path

import pytest

pytest.importorskip("bw_processing", reason="needs the [bw] extra")
pytest.importorskip("bw2calc", reason="needs the [bw] extra")

from cli.build_bw_package import BuildBwPackageCli
from tests.fixtures.bw_synthetic import PRODUCT_IDS


def test_parser_defaults():
    args = BuildBwPackageCli.parser().parse_args([])
    assert args.out == Path("bw_package")
    assert args.parity_n == 3
    assert args.parity_full is False
    assert args.skip_parity is False


def test_no_bundle_only_flags():
    # The bundle CLI's skeleton-fill machinery does not exist here: the export
    # consumes the locally built ScoringPackage, so --no-fill must be gone.
    with pytest.raises(SystemExit):
        BuildBwPackageCli.parser().parse_args(["--no-fill"])


def test_sample_product_ids_is_deterministic(synthetic_package):
    cli = BuildBwPackageCli()
    ids_a = cli.sample_product_ids(synthetic_package, n=1)
    ids_b = cli.sample_product_ids(synthetic_package, n=1)
    assert ids_a == ids_b
    assert len(ids_a) == 1
    assert ids_a[0] in PRODUCT_IDS


def test_sample_full_returns_all(synthetic_package):
    cli = BuildBwPackageCli()
    ids = cli.sample_product_ids(synthetic_package, n=None, full=True)
    assert set(ids) == set(PRODUCT_IDS)


def test_sample_zero_returns_empty_without_dividing_by_zero(synthetic_package):
    cli = BuildBwPackageCli()
    assert cli.sample_product_ids(synthetic_package, n=0) == []


class TestSolverHint:
    @staticmethod
    def _importable(monkeypatch, *names: str, failure: type[ImportError] = ModuleNotFoundError):
        def import_module(name):
            if name in names:
                return object()
            raise failure(name)

        monkeypatch.setattr("cli.build_bw_package.importlib.import_module", import_module)

    def test_names_the_extras_when_no_shared_solver_imports(self, monkeypatch):
        self._importable(monkeypatch)

        hint = BuildBwPackageCli.solver_hint()

        assert "can be imported" in hint
        assert "pardiso extra" in hint
        assert "umfpack extra" in hint

    @pytest.mark.parametrize("installed", ["pypardiso", "scikits.umfpack"])
    def test_is_empty_when_one_shared_solver_imports(self, monkeypatch, installed):
        self._importable(monkeypatch, installed)

        assert BuildBwPackageCli.solver_hint() == ""

    def test_an_installed_solver_that_does_not_load_counts_as_missing(self, monkeypatch):
        # scikit-umfpack present but its library broken: scipy and bw2calc
        # catch the ImportError and fall back to SuperLU, so the hint applies.
        self._importable(monkeypatch, failure=ImportError)

        assert "umfpack extra" in BuildBwPackageCli.solver_hint()
