"""Unit tests for :class:`CatalogKeyResolver` — id → (db, code) recovery."""

from __future__ import annotations

import pandas as pd

from bw_export.catalog_key_resolver import CatalogKeyResolver
from scoring.exchange_frame_builder import ExchangeFrameBuilder


def _resolver(**kwargs) -> CatalogKeyResolver:
    base = dict(
        product_catalog=pd.DataFrame(
            {
                "database": ["agb"],
                "code": ["apple"],
                "name": ["Apple, at farm"],
                "unit": ["kg"],
            }
        ),
        biosphere_catalog=pd.DataFrame(
            {
                "database": ["biosphere3"],
                "code": ["co2"],
                "name": ["Carbon dioxide"],
                "categories": ["air::urban"],
                "unit": ["kg"],
            }
        ),
        ecoinvent_catalog=pd.DataFrame(
            {"code": ["apple"], "location": ["FR"], "reference_product": ["apple"]}
        ),
    )
    base.update(kwargs)
    return CatalogKeyResolver(**base)


def test_activity_id_resolves_to_catalog_key_and_labels():
    resolver = _resolver()
    cid = ExchangeFrameBuilder.flow_id_for(("agb", "apple"))
    meta = resolver.activity(cid)
    assert meta.key == ("agb", "apple")
    assert meta.name == "Apple, at farm"
    assert meta.unit == "kg"
    assert meta.location == "FR"  # joined from the ecoinvent catalog
    assert meta.reference_product == "apple"


def test_unknown_activity_id_falls_back_to_synthetic_key():
    resolver = _resolver()
    meta = resolver.activity(999_999)
    assert meta.key == ("agribalyse-ef31", "999999")


def test_biosphere_id_resolves_and_splits_categories():
    resolver = _resolver()
    bid = ExchangeFrameBuilder.flow_id_for(("biosphere3", "co2"))
    meta = resolver.biosphere(bid)
    assert meta.key == ("biosphere3", "co2")
    assert meta.categories == ("air", "urban")
    assert meta.is_synthetic_correction is False


def test_synthetic_correction_flow_uses_supplied_code():
    resolver = _resolver(synthetic_flow_codes={42: "aware-water-use"})
    meta = resolver.biosphere(42)
    assert meta.key == ("agribalyse-ef31-correction", "aware-water-use")
    assert meta.is_synthetic_correction is True


def test_activity_catalog_is_authoritative_and_names_synthetics():
    # The activity_catalog is keyed by the column id directly, so it can
    # name columns the (db, code) hash can't — notably Allocator synthetic
    # splits whose ids are not flow_id_for hashes.
    synthetic_col_id = 7_000_000_000_000
    activity_catalog = pd.DataFrame(
        {
            "activity_id": [synthetic_col_id],
            "database": ["agribalyse-3.2"],
            "code": ["parent::123"],
            "name": ["Milk, raw | Cream"],
            "type": ["multifunctional_split"],
            "unit": ["kg"],
            "location": ["FR"],
            "reference_product": ["Cream"],
            "product_id": [123],
        }
    )
    resolver = _resolver(activity_catalog=activity_catalog)
    meta = resolver.activity(synthetic_col_id)
    assert meta.name == "Milk, raw | Cream"
    assert meta.unit == "kg"
    assert meta.location == "FR"
    assert meta.reference_product == "Cream"


def test_activity_catalog_absent_falls_back_to_product_catalog():
    # No activity_catalog → legacy (db, code)-hash path still works.
    resolver = _resolver(activity_catalog=None)
    cid = ExchangeFrameBuilder.flow_id_for(("agb", "apple"))
    assert resolver.activity(cid).name == "Apple, at farm"
