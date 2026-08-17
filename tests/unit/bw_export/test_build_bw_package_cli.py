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
