"""``SimaproFormResolver`` — SimaPro-form names resolved on the ecoinvent catalog."""

from __future__ import annotations

from matching.ecoinvent_catalog import EcoinventActivityRef, EcoinventCatalog
from matching.simapro_form_resolver import SimaproFormResolver

EI = "ecoinvent-3.9.1-cutoff"


def _ref(
    code: str, name: str, location: str, db: str = EI, product: str = "x"
) -> EcoinventActivityRef:
    return EcoinventActivityRef(
        db=db, code=code, name=name, unit="kilogram", location=location, reference_product=product
    )


def _resolver(*refs: EcoinventActivityRef) -> SimaproFormResolver:
    catalog = EcoinventCatalog(db_names=(EI,), activities=tuple(refs))
    return SimaproFormResolver(catalog=catalog, ei_db=EI)


class TestResolve:
    def test_complete_activity_name(self):
        resolver = _resolver(_ref("c1", "carrot production", "RoW"))

        hit = resolver.resolve(
            "Carrot {RoW}| carrot production | Cut-off, U - Copied from Ecoinvent U"
        )

        assert hit.code == "c1"

    def test_truncated_market_gets_the_product_appended(self):
        resolver = _resolver(_ref("m1", "market for soybean seed, for sowing", "GLO"))

        hit = resolver.resolve(
            "Soybean seed, for sowing {GLO}| market for | Cut-off, U - Copied from Ecoinvent U"
        )

        assert hit.code == "m1"

    def test_truncated_transformer_gets_the_product_prepended(self):
        resolver = _resolver(_ref("t1", "tofu production", "RoW"))

        assert resolver.resolve("Tofu {RoW}| production | Cut-off, U").code == "t1"

    def test_falls_back_to_the_global_dataset(self):
        resolver = _resolver(_ref("g1", "market for p-dichlorobenzene", "GLO"))

        hit = resolver.resolve(
            "P-dichlorobenzene {RER}| market for p-dichlorobenzene | Cut-off, S - Copied from "
            "Ecoinvent {GLO} U"
        )

        assert hit.code == "g1"

    def test_own_location_wins_over_the_fallback(self):
        resolver = _resolver(
            _ref("glo", "onion production", "GLO"), _ref("row", "onion production", "RoW")
        )

        assert resolver.resolve("Onion {RoW}| onion production | Cut-off, U").code == "row"


class TestCoProducts:
    def test_display_name_picks_the_co_product(self):
        name = "soybean meal and crude oil production"
        resolver = _resolver(
            _ref("meal", name, "BR", product="soybean meal"),
            _ref("oil", name, "BR", product="soybean oil, crude"),
        )

        hit = resolver.resolve(
            f"Soybean meal {{BR}}| {name} | Cut-off, U - Copied from Ecoinvent U"
        )

        assert hit.code == "meal"

    def test_truncated_name_with_co_products(self):
        resolver = _resolver(
            _ref("tofu", "tofu production", "RoW", product="tofu"),
            _ref("okara", "tofu production", "RoW", product="okara"),
        )

        assert resolver.resolve("Tofu {RoW}| production | Cut-off, U").code == "tofu"


class TestNoGuessing:
    def test_two_activities_with_the_same_name_and_location_stay_unlinked(self):
        resolver = _resolver(
            _ref("a", "soybean production", "US"), _ref("b", "soybean production", "US")
        )

        assert resolver.resolve("Soybean {US}| production | Cut-off, U") is None

    def test_ambiguity_at_the_own_location_does_not_fall_through(self):
        resolver = _resolver(
            _ref("a", "soybean production", "US"),
            _ref("b", "soybean production", "US"),
            _ref("g", "soybean production", "GLO"),
        )

        assert resolver.resolve("Soybean {US}| production | Cut-off, U") is None

    def test_unknown_activity(self):
        resolver = _resolver(_ref("c1", "carrot production", "RoW"))

        assert resolver.resolve("Core board {GLO}| recycling of core board | Cut-off, U") is None

    def test_name_not_in_simapro_form(self):
        resolver = _resolver(_ref("c1", "carrot production", "RoW"))

        assert resolver.resolve("carrot production") is None
        assert resolver.resolve("") is None

    def test_other_databases_are_ignored(self):
        resolver = _resolver(_ref("x", "carrot production", "RoW", db="other-db"))

        assert resolver.resolve("Carrot {RoW}| carrot production | Cut-off, U") is None
