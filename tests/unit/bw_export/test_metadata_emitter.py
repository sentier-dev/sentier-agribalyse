import json

import pandas as pd
import pytest

pytest.importorskip("bw_processing", reason="needs the [bw] extra")

from bw_export.bw_node_types import ActivityMeta, BioMeta
from bw_export.correction_embedder import CorrectionEmbedder
from bw_export.datapackage_writer import DatapackageWriter
from bw_export.metadata_emitter import MetadataEmitter
from tests.fixtures.bw_synthetic import ACTIVITY_IDS, BIO_FLOW_ID, METHOD, PRODUCT_IDS


def _product_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "database": ["agribalyse-3.2", "agribalyse-3.2"],
            "code": ["c101", "c102"],
            "name": ["prod one", "prod two"],
            "type": ["product", "product"],
            "unit": ["kg", "kg"],
            "product_id": PRODUCT_IDS,
        }
    )


def _activity(col_id: int) -> ActivityMeta:
    return ActivityMeta(
        key=("db", f"act-{col_id}"),
        name=f"activity {col_id}",
        unit="kg",
        location="GLO",
        reference_product="ref",
    )


def _biosphere(bid: int) -> BioMeta:
    return BioMeta(
        key=("bio", f"flow-{bid}"),
        name=f"flow {bid}",
        categories=("air", "urban"),
        unit="kg",
        is_synthetic_correction=bid != BIO_FLOW_ID,
    )


def _emit(tmp_path, synthetic_package, *, activity=_activity, bio=_biosphere, catalog=None):
    embedded = CorrectionEmbedder().embed(synthetic_package)
    written = DatapackageWriter().write(embedded, out_root=tmp_path)
    MetadataEmitter().emit(
        out_root=tmp_path,
        embedded=embedded,
        written=written,
        activity_resolver=activity,
        bio_resolver=bio,
        product_catalog=_product_catalog() if catalog is None else catalog,
        parity_scores={PRODUCT_IDS[0]: {METHOD: 20.5}},
        parity_tolerance=1e-9,
        manifest_extra={"ecoinvent_version": "3.9.1", "content_hash": "synthetic"},
        parity={"n_checked": 1, "max_rel_error": 0.0, "tolerance": 1e-9, "passed": True},
    )
    return embedded


def test_emits_all_metadata_files(tmp_path, synthetic_package):
    _emit(tmp_path, synthetic_package)

    for name in ("activities", "products", "biosphere"):
        assert (tmp_path / "metadata" / f"{name}.parquet").is_file()
    for name in ("methods", "manifest", "parity_samples"):
        assert (tmp_path / "metadata" / f"{name}.json").is_file()
    assert (tmp_path / "run_example.py").exists()
    assert (tmp_path / "README.md").exists()


def test_activities_are_fully_named_with_production_map(tmp_path, synthetic_package):
    _emit(tmp_path, synthetic_package)
    acts = pd.read_parquet(tmp_path / "metadata" / "activities.parquet")

    assert set(acts["col_id"]) == set(ACTIVITY_IDS)
    for col in ("name", "database", "code", "production_product_id"):
        assert acts[col].notna().all()
        assert (acts[col].astype("string").str.len() > 0).all()
    # Matrix index i pairs activity-col i with product-row i: 201->101, 202->102.
    mapping = dict(zip(acts["col_id"], acts["production_product_id"], strict=True))
    assert mapping == {ACTIVITY_IDS[0]: PRODUCT_IDS[0], ACTIVITY_IDS[1]: PRODUCT_IDS[1]}


def test_biosphere_carries_keys_and_flags_correction(tmp_path, synthetic_package):
    _emit(tmp_path, synthetic_package)
    bio = pd.read_parquet(tmp_path / "metadata" / "biosphere.parquet")

    assert BIO_FLOW_ID in set(bio["bioflow_id"])
    assert int(bio["is_synthetic_correction"].sum()) == 1  # the AWARE flow
    for col in ("database", "code"):
        assert (bio[col].astype("string").str.len() > 0).all()
    real = bio.loc[bio["bioflow_id"] == BIO_FLOW_ID].iloc[0]
    assert real["categories"] == "air::urban"


def test_parity_samples_map_product_to_producing_activity(tmp_path, synthetic_package):
    _emit(tmp_path, synthetic_package)
    payload = json.loads((tmp_path / "metadata" / "parity_samples.json").read_text())

    assert payload["tolerance"] == 1e-9
    assert len(payload["samples"]) == 1
    sample = payload["samples"][0]
    assert sample["product_id"] == PRODUCT_IDS[0]
    assert sample["activity_col_id"] == ACTIVITY_IDS[0]
    assert sample["expected"] == [[list(METHOD), 20.5]]


def test_manifest_counts_and_methods(tmp_path, synthetic_package):
    _emit(tmp_path, synthetic_package)
    manifest = json.loads((tmp_path / "metadata" / "manifest.json").read_text())
    assert manifest["parity"]["passed"] is True
    assert manifest["counts"]["n_methods"] == 1
    assert manifest["content_hash"] == "synthetic"

    methods = json.loads((tmp_path / "metadata" / "methods.json").read_text())
    assert methods["methods"][0]["path"].startswith("methods/")


def test_readme_has_public_licence_wording(tmp_path, synthetic_package):
    _emit(tmp_path, synthetic_package)
    readme = (tmp_path / "README.md").read_text()
    assert "must not be redistributed" in readme
    assert "your own ecoinvent licence" in readme
    # No bundle-era distribution channels in the public export.
    assert "Release" not in readme
    assert "tar.gz" not in readme


def test_build_fails_loudly_on_unnamed_activity(tmp_path, synthetic_package):
    def blank(col_id: int) -> ActivityMeta:
        return ActivityMeta(
            ("db", f"act-{col_id}"), name="", unit="kg", location="GLO", reference_product="ref"
        )

    with pytest.raises(ValueError, match="empty/NA 'name'"):
        _emit(tmp_path, synthetic_package, activity=blank)


def test_readme_tells_importer_users_to_install_pyarrow(tmp_path, synthetic_package):
    _emit(tmp_path, synthetic_package)
    readme = (tmp_path / "README.md").read_text()
    section_b = readme.split("## B. Import into Brightway")[1]
    assert "pyarrow" in section_b
    assert "conda install pyarrow" in section_b


def test_readme_gives_the_legacy_install_command(tmp_path, synthetic_package):
    """Legacy bw2data 3.x does not cap NumPy but needs NumPy 1, and pyarrow 26
    needs NumPy 2 without declaring it: a legacy virtualenv needs both pins,
    in one command a user can copy (the integration test uses the same pins)."""
    _emit(tmp_path, synthetic_package)
    section_b = (tmp_path / "README.md").read_text().split("## B. Import into Brightway")[1]
    assert 'pip install "numpy<2" "pyarrow<26"' in section_b


def test_products_keep_the_catalog_names_and_carry_the_producer_location(
    tmp_path, synthetic_package
):
    _emit(tmp_path, synthetic_package)
    products = pd.read_parquet(tmp_path / "metadata" / "products.parquet")

    assert list(products["name"]) == ["prod one", "prod two"]
    assert list(products["database"]) == ["agribalyse-3.2", "agribalyse-3.2"]
    assert list(products["location"]) == ["GLO", "GLO"]


def test_a_product_missing_from_the_catalog_is_named_after_its_producer(
    tmp_path, synthetic_package
):
    """The product catalog holds the AGRIBALYSE products only. An ecoinvent
    product takes the reference product, unit, key and location of the
    activity that produces it, as the activity table does."""

    def producer(col_id: int) -> ActivityMeta:
        return ActivityMeta(
            key=("ecoinvent-3.9.1-cutoff", f"eco-{col_id}"),
            name=f"market for thing {col_id}",
            unit="kilowatt hour",
            location="CH",
            reference_product=f"thing {col_id}",
        )

    _emit(tmp_path, synthetic_package, activity=producer, catalog=_product_catalog().iloc[1:])
    products = pd.read_parquet(tmp_path / "metadata" / "products.parquet").set_index("product_id")

    background = products.loc[PRODUCT_IDS[0]]
    assert background["name"] == f"thing {ACTIVITY_IDS[0]}"
    assert (background["database"], background["code"]) == (
        "ecoinvent-3.9.1-cutoff",
        f"eco-{ACTIVITY_IDS[0]}",
    )
    assert (background["type"], background["unit"], background["location"]) == (
        "product",
        "kilowatt hour",
        "CH",
    )
    assert products.loc[PRODUCT_IDS[1], "name"] == "prod two"


def test_build_fails_loudly_on_a_product_nothing_names(tmp_path, synthetic_package):
    def nameless(col_id: int) -> ActivityMeta:
        return ActivityMeta(
            ("db", f"act-{col_id}"), name="", unit="kg", location="GLO", reference_product=""
        )

    empty = _product_catalog().iloc[0:0]
    with pytest.raises(ValueError, match=r"products\.parquet"):
        _emit(tmp_path, synthetic_package, activity=nameless, catalog=empty)
