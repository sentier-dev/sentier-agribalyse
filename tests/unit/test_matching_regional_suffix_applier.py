"""Unit tests for ``RegionalSuffixApplier``."""

from __future__ import annotations

from matching.regional_suffix_applier import RegionalSuffixApplier


class TestRegionalSuffixApplier:
    def test_rewrites_regional_water_flow_to_synthetic_code(self):
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Water, well, CN",
                        "input": ("ecoinvent-3.9.1-biosphere", "water-well-uuid"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "water-well-uuid@CN",
        )
        assert stats.n_rewritten == 1

    def test_pre_flowmap_snapshot_preferred_when_present(self):
        """The flowmap rewrites ``exc.name`` to the canonical bio3
        spelling. The snapshot field carries the original regional
        spelling and must take precedence."""
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Water",  # post-flowmap canonical
                        "_regional_source_name": "Water, IN",  # pre-flowmap
                        "input": ("biosphere3", "water-uuid"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "biosphere3",
            "water-uuid@IN",
        )
        assert stats.n_rewritten == 1

    def test_simapro_name_preferred_over_stripped_name(self):
        """In-region flow: name was stripped to ``Water, well`` but the
        original ``Water, well, FR`` survives under ``simapro name``."""
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Water, well",
                        "simapro name": "Water, well, FR",
                        "input": ("ecoinvent-3.9.1-biosphere", "water-well-uuid"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "water-well-uuid@FR",
        )
        assert stats.n_rewritten == 1

    def test_non_regional_flow_left_untouched(self):
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Carbon dioxide",
                        "input": ("biosphere3", "co2-uuid"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == ("biosphere3", "co2-uuid")
        assert stats.n_rewritten == 0

    def test_non_water_canonical_name_not_rewritten(self):
        """``Nitrogen oxides, DE`` is an air-pollutant regional flow.
        SimaPro EF v3.1 (adapted) only carries per-region CFs for the
        Water-use method; rewriting NOx onto a synthetic ``@DE`` row
        would silently zero its PM / acidification contribution. The
        applier keeps the canonical-name filter tight to ``Water``."""
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Nitrogen oxides",  # post-flowmap canonical
                        "_regional_source_name": "Nitrogen oxides, DE",
                        "input": ("ecoinvent-3.9.1-biosphere", "nox-uuid"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "nox-uuid",
        )
        assert stats.n_rewritten == 0
        assert stats.n_skipped_non_water == 1

    def test_chemical_formula_suffix_not_rewritten(self):
        """``HCC-30`` is a chemical formula token, not a region."""
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Methane, dichloro-, HCC-30",
                        "input": ("ecoinvent-3.9.1-biosphere", "ch2cl2-uuid"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "ch2cl2-uuid",
        )
        assert stats.n_rewritten == 0

    def test_already_synthetic_code_left_untouched(self):
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Water, well, CN",
                        "input": (
                            "ecoinvent-3.9.1-biosphere",
                            "water-well-uuid@CN",
                        ),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == (
            "ecoinvent-3.9.1-biosphere",
            "water-well-uuid@CN",
        )
        assert stats.n_rewritten == 0
        assert stats.n_already_synthetic == 1

    def test_ef_database_not_rewritten(self):
        """EF database is outside the regionalisable allowlist — leave alone."""
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Water, well, CN",
                        "input": ("ef", "ef-water-uuid"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert sp_data[0]["exchanges"][0]["input"] == ("ef", "ef-water-uuid")
        assert stats.n_unsupported_db == 1

    def test_unlinked_exchange_skipped(self):
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "biosphere",
                        "name": "Water, well, FR",
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert "input" not in sp_data[0]["exchanges"][0]
        assert stats.n_no_input == 1

    def test_technosphere_exchange_ignored(self):
        sp_data = [
            {
                "name": "P",
                "exchanges": [
                    {
                        "type": "technosphere",
                        "name": "Some, FR",
                        "input": ("foo", "bar"),
                    }
                ],
            }
        ]
        stats = RegionalSuffixApplier().apply(sp_data)
        assert stats.n_visited == 0
