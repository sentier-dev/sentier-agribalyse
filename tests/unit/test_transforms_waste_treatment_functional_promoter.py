"""Unit tests for ``WasteTreatmentFunctionalPromoter``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from transforms.waste_treatment_functional_promoter import (
    WasteTreatmentFunctionalPromoter,
)


@dataclass
class _Sp:
    data: list[dict[str, Any]]


class TestWasteTreatmentFunctionalPromoter:
    def test_promotes_functional_tech_when_no_real_production(self):
        # Mimics ``Compost, of biowaste (amendment) {RER}``: a zero-amount
        # production stub plus a functional=True technosphere row that
        # carries the actual reference flow (900 kg of biowaste treated).
        ds = {
            "name": "Compost, of biowaste (amendment) {RER}",
            "exchanges": [
                {
                    "type": "production",
                    "amount": 0,
                    "functional": False,
                    "name": "Treatment of biowaste",
                },
                {
                    "type": "technosphere",
                    "amount": 900.0,
                    "functional": True,
                    "name": "Treatment of biowaste",
                },
                {"type": "technosphere", "amount": 4.5, "functional": False, "name": "Diesel"},
            ],
        }
        stats = WasteTreatmentFunctionalPromoter().apply(_Sp(data=[ds]))
        assert stats == {"promoted": 1, "activities": 1, "zero_stubs_dropped": 1}
        # The functional=True row is now a production row...
        productions = [e for e in ds["exchanges"] if e["type"] == "production"]
        assert len(productions) == 1
        assert productions[0]["amount"] == 900.0
        assert productions[0]["functional"] is True
        # ...and the zero-amount stub is gone.
        assert all(e.get("amount") != 0 or e["type"] == "technosphere" for e in ds["exchanges"])
        # Other rows untouched.
        diesel = next(e for e in ds["exchanges"] if e.get("name") == "Diesel")
        assert diesel["type"] == "technosphere"
        assert diesel["amount"] == pytest.approx(4.5)

    def test_skips_when_real_production_exists(self):
        # Multifunctional activity — has real production AND a
        # functional=True technosphere consumption (a treatment service
        # it uses, not provides). Nothing should change.
        ds = {
            "name": "Some real activity",
            "exchanges": [
                {"type": "production", "amount": 1.0, "functional": True, "name": "Cocoa butter"},
                {
                    "type": "technosphere",
                    "amount": 0.3,
                    "functional": True,
                    "name": "Treatment of biowaste",
                },
            ],
        }
        original = [dict(e) for e in ds["exchanges"]]
        stats = WasteTreatmentFunctionalPromoter().apply(_Sp(data=[ds]))
        assert stats == {"promoted": 0, "activities": 0, "zero_stubs_dropped": 0}
        assert ds["exchanges"] == original

    def test_no_functional_tech_no_change(self):
        ds = {
            "name": "Activity without any functional=True tech",
            "exchanges": [
                {"type": "production", "amount": 1.0, "functional": True, "name": "Product"},
                {"type": "technosphere", "amount": 0.5, "functional": False, "name": "Input"},
            ],
        }
        original = [dict(e) for e in ds["exchanges"]]
        stats = WasteTreatmentFunctionalPromoter().apply(_Sp(data=[ds]))
        assert stats == {"promoted": 0, "activities": 0, "zero_stubs_dropped": 0}
        assert ds["exchanges"] == original

    def test_handles_activity_with_no_production_rows_at_all(self):
        # No production rows, just a functional tech row + consumption.
        ds = {
            "name": "Pure tech-only activity",
            "exchanges": [
                {"type": "technosphere", "amount": 1.0, "functional": True, "name": "Service"},
                {"type": "technosphere", "amount": 0.1, "functional": False, "name": "Input"},
            ],
        }
        stats = WasteTreatmentFunctionalPromoter().apply(_Sp(data=[ds]))
        assert stats == {"promoted": 1, "activities": 1, "zero_stubs_dropped": 0}
        productions = [e for e in ds["exchanges"] if e["type"] == "production"]
        assert len(productions) == 1
