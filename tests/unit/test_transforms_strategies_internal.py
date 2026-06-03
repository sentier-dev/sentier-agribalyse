"""Unit tests for the lifted internal-link strategies (Phase F6).

These four classes replace the last bw2io callables the linker used to
drive — pure dict transforms, no bw2 dependency.
"""

from __future__ import annotations

from transforms.strategies.internal import (
    DropUnlinkedExchanges,
    LinkIterableByFields,
    SetMetadataUsingSingleFunctionalExchange,
    SplitSimaproNameGeo,
)

# ============================================================================
# SetMetadataUsingSingleFunctionalExchange


class TestSetMetadataUsingSingleFunctionalExchange:
    def test_lifts_name_and_unit_when_missing(self):
        data = [
            {
                "exchanges": [
                    {
                        "functional": True,
                        "name": "Tap water",
                        "unit": "kilogram",
                        "amount": 1.0,
                    }
                ]
            }
        ]
        SetMetadataUsingSingleFunctionalExchange()(data)
        ds = data[0]
        assert ds["name"] == "Tap water"
        assert ds["reference product"] == "Tap water"
        assert ds["unit"] == "kilogram"
        assert ds["production amount"] == 1.0

    def test_skips_when_multiple_functional_edges(self):
        data = [
            {
                "exchanges": [
                    {"functional": True, "name": "A", "unit": "kg", "amount": 1.0},
                    {"functional": True, "name": "B", "unit": "kg", "amount": 1.0},
                ]
            }
        ]
        SetMetadataUsingSingleFunctionalExchange()(data)
        # Both unset because the rule requires exactly one functional edge.
        assert "name" not in data[0]

    def test_does_not_overwrite_existing_values(self):
        data = [
            {
                "name": "Existing",
                "exchanges": [
                    {"functional": True, "name": "Different", "unit": "kg", "amount": 1.0}
                ],
            }
        ]
        SetMetadataUsingSingleFunctionalExchange()(data)
        assert data[0]["name"] == "Existing"


# ============================================================================
# SplitSimaproNameGeo


class TestSplitSimaproNameGeo:
    def test_splits_dataset_and_exchange_names(self):
        data = [
            {
                "name": "foo/CH U",
                "exchanges": [{"name": "bar/US U", "type": "technosphere"}],
            }
        ]
        SplitSimaproNameGeo()(data)
        ds = data[0]
        assert ds["name"] == "foo"
        assert ds["reference product"] == "foo"
        assert ds["location"] == "CH"
        assert ds["simapro name"] == "foo/CH U"
        exc = ds["exchanges"][0]
        assert exc["name"] == "bar"
        assert exc["location"] == "US"
        assert exc["simapro name"] == "bar/US U"

    def test_does_nothing_when_pattern_absent(self):
        data = [{"name": "no_geo_suffix", "exchanges": []}]
        SplitSimaproNameGeo()(data)
        assert data[0]["name"] == "no_geo_suffix"
        assert "location" not in data[0]


# ============================================================================
# DropUnlinkedExchanges


class TestDropUnlinkedExchanges:
    def test_removes_exchanges_without_input(self):
        data = [
            {
                "exchanges": [
                    {"input": ("a", "b"), "amount": 1.0},
                    {"amount": 2.0},
                    {"input": ("c", "d"), "amount": 3.0},
                ]
            }
        ]
        DropUnlinkedExchanges()(data)
        assert len(data[0]["exchanges"]) == 2
        assert all(e.get("input") for e in data[0]["exchanges"])


# ============================================================================
# LinkIterableByFields


class TestLinkIterableByFields:
    def _data(self) -> list[dict]:
        return [
            {
                "type": "process",
                "database": "agb",
                "code": "p1",
                "name": "P1",
                "exchanges": [
                    {"name": "X", "unit": "kg", "type": "technosphere"},
                    {"name": "Y", "unit": "kg", "type": "technosphere"},
                ],
            },
            {
                "type": "product",
                "database": "agb",
                "code": "prod-x",
                "name": "X",
                "unit": "kg",
            },
        ]

    def test_internal_link_matches_processes_to_products(self):
        data = self._data()
        LinkIterableByFields(
            fields=("name", "unit"),
            this_node_kinds=("process",),
            other_node_kinds=("product",),
            internal=True,
        ).apply(data)
        # X exchange links to prod-x; Y stays unlinked (no matching product).
        assert data[0]["exchanges"][0]["input"] == ("agb", "prod-x")
        assert "input" not in data[0]["exchanges"][1]

    def test_skips_already_linked_when_relink_false(self):
        data = self._data()
        data[0]["exchanges"][0]["input"] = ("other", "preset")
        LinkIterableByFields(
            fields=("name", "unit"),
            this_node_kinds=("process",),
            other_node_kinds=("product",),
            internal=True,
        ).apply(data)
        # Pre-set input preserved.
        assert data[0]["exchanges"][0]["input"] == ("other", "preset")

    def test_relinks_when_flag_true(self):
        data = self._data()
        data[0]["exchanges"][0]["input"] = ("other", "preset")
        LinkIterableByFields(
            fields=("name", "unit"),
            this_node_kinds=("process",),
            other_node_kinds=("product",),
            internal=True,
            relink=True,
        ).apply(data)
        # Re-linked to the matching product.
        assert data[0]["exchanges"][0]["input"] == ("agb", "prod-x")

    def test_duplicate_keys_skip_match_silently(self):
        # Two products with the same (name, unit) — bw2io would raise; our
        # in-house version "leaves unlinked" to match the catalog matchers.
        data = [
            {
                "type": "process",
                "database": "agb",
                "code": "p1",
                "exchanges": [{"name": "X", "unit": "kg", "type": "technosphere"}],
            },
            {
                "type": "product",
                "database": "agb",
                "code": "prod-a",
                "name": "X",
                "unit": "kg",
            },
            {
                "type": "product",
                "database": "agb",
                "code": "prod-b",
                "name": "X",
                "unit": "kg",
            },
        ]
        LinkIterableByFields(
            fields=("name", "unit"),
            this_node_kinds=("process",),
            other_node_kinds=("product",),
            internal=True,
        ).apply(data)
        assert "input" not in data[0]["exchanges"][0]

    def test_edge_kinds_filter_limits_candidates(self):
        data = [
            {
                "type": "process",
                "database": "agb",
                "code": "p1",
                "exchanges": [
                    {"name": "X", "unit": "kg", "type": "technosphere"},
                    {"name": "X", "unit": "kg", "type": "biosphere"},
                ],
            },
            {
                "type": "product",
                "database": "agb",
                "code": "prod-x",
                "name": "X",
                "unit": "kg",
            },
        ]
        LinkIterableByFields(
            fields=("name", "unit"),
            edge_kinds=("technosphere",),
            this_node_kinds=("process",),
            other_node_kinds=("product",),
            internal=True,
        ).apply(data)
        # Technosphere exchange linked; biosphere (filtered out) is not.
        assert data[0]["exchanges"][0]["input"] == ("agb", "prod-x")
        assert "input" not in data[0]["exchanges"][1]
