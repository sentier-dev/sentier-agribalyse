"""``RebuiltRecipe`` — the amount-free recipe of the processes rebuilt from ecoinvent.

The skeleton zeroes the columns of the processes :class:`UnitCopyRebuilder`
rebuilt from the user's snapshot. The recipe says, for every exchange of those
processes that reached the package, which ecoinvent exchange the amount came
from, with its unit multiplier and its sign, so a licensed user can recompute
the columns from their own exchanges. Made-up data throughout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoring.dangling_edge_pruner import DanglingEdgePruner
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.rebuilt_recipe import RebuiltRecipe

AGB = "agribalyse-3.2"
EI = "ecoinvent-3.9.1-cutoff"
BIO = "ecoinvent-3.9.1-biosphere"
ATTR = RebuiltRecipe.ATTRIBUTE


def _src(output, db, code, kind, multiplier=1.0) -> dict:
    return {
        "output_code": output,
        "input_database": db,
        "input_code": code,
        "type": kind,
        "multiplier": multiplier,
    }


def _fid(db: str, code: str) -> int:
    return ExchangeFrameBuilder.flow_id_for((db, code))


def _sp_data() -> list[dict]:
    """A rebuilt market, ADEME's adapted dataset it draws from, and a consumer."""
    return [
        {
            "database": AGB,
            "code": "ecoinvent-unit-copy-mkt",
            "type": "process",
            "exchanges": [
                {
                    "type": "production",
                    "amount": 1.0,
                    "input": (AGB, "ecoinvent-unit-copy-product-mkt"),
                    ATTR: _src("mkt", EI, "mkt", "production"),
                },
                {
                    "type": "technosphere",
                    "amount": 0.0007,
                    "input": (AGB, "adapted-product"),
                    ATTR: _src("mkt", EI, "soy", "technosphere", 0.001),
                },
                {
                    "type": "technosphere",
                    "amount": 0.2,
                    "input": (EI, "nobody-makes-this"),
                    ATTR: _src("mkt", EI, "nobody-makes-this", "technosphere"),
                },
                {
                    "type": "biosphere",
                    "amount": 0.05,
                    "input": (BIO, "co2"),
                    ATTR: _src("mkt", BIO, "co2", "biosphere"),
                },
            ],
        },
        {
            "database": AGB,
            "code": "adapted",
            "type": "process",
            "exchanges": [
                {"type": "production", "amount": 1.0, "input": (AGB, "adapted-product")},
                {"type": "biosphere", "amount": 0.3, "input": (BIO, "co2")},
            ],
        },
        {
            "database": AGB,
            "code": "bread",
            "type": "process",
            "exchanges": [
                {"type": "production", "amount": 1.0, "input": (AGB, "bread-product")},
                {
                    "type": "technosphere",
                    "amount": 0.5,
                    "input": (AGB, "ecoinvent-unit-copy-product-mkt"),
                },
            ],
        },
    ]


def _final_frame(sp_data):
    frame, _ = DanglingEdgePruner().prune(ExchangeFrameBuilder().from_sp_data(sp_data))
    return frame


class TestFromSpData:
    def test_tagged_exchanges_become_rows_keyed_like_the_package(self):
        sp_data = _sp_data()

        recipe = RebuiltRecipe.from_sp_data(sp_data, _final_frame(sp_data)).frame

        market = _fid(AGB, "ecoinvent-unit-copy-mkt")
        rows = {
            (r.matrix, r.output_id, r.input_id, r.edge_type, r.sign, r.multiplier): (
                r.source_output_code,
                r.source_input_database,
                r.source_input_code,
                r.source_type,
            )
            for r in recipe.itertuples()
        }
        assert rows == {
            (
                "technosphere",
                market,
                _fid(AGB, "ecoinvent-unit-copy-product-mkt"),
                "production",
                1.0,
                1.0,
            ): ("mkt", EI, "mkt", "production"),
            (
                "technosphere",
                market,
                _fid(AGB, "adapted-product"),
                "technosphere",
                -1.0,
                0.001,
            ): ("mkt", EI, "soy", "technosphere"),
            ("biosphere", market, _fid(BIO, "co2"), "biosphere", 1.0, 1.0): (
                "mkt",
                BIO,
                "co2",
                "biosphere",
            ),
        }

    def test_an_exchange_the_final_frame_dropped_is_not_in_the_recipe(self):
        sp_data = _sp_data()

        recipe = RebuiltRecipe.from_sp_data(sp_data, _final_frame(sp_data)).frame

        # The draw on a product nobody produces was pruned from the package.
        assert "nobody-makes-this" not in set(recipe["source_input_code"])

    def test_the_recipe_holds_no_amount(self):
        sp_data = _sp_data()

        recipe = RebuiltRecipe.from_sp_data(sp_data, _final_frame(sp_data)).frame

        assert tuple(recipe.columns) == RebuiltRecipe.COLUMNS
        assert "amount" not in recipe.columns


class TestEntryValues:
    @staticmethod
    def _recipe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sign": [1.0, -1.0, -1.0, 1.0],
                "multiplier": [1.0, 0.001, 1.0, 1.0],
                "source_output_code": ["mkt", "mkt", "mkt", "mkt"],
                "source_input_database": [EI, EI, EI, BIO],
                "source_input_code": ["mkt", "soy", "waste-treatment", "co2"],
                "source_type": ["production", "technosphere", "technosphere", "biosphere"],
            }
        )

    @staticmethod
    def _snapshot() -> pd.DataFrame:
        rows = [
            ("mkt", EI, "mkt", 1.0, "production"),
            ("mkt", EI, "soy", 700.0, "technosphere"),
            # ecoinvent's waste convention: the treatment produces -1, its
            # producer draws a negative amount. The rebuilder reads both flipped.
            ("mkt", EI, "waste-treatment", -0.25, "technosphere"),
            ("mkt", BIO, "co2", 0.05, "biosphere"),
            ("waste-treatment", EI, "waste-treatment", -1.0, "production"),
        ]
        return pd.DataFrame(
            [
                {
                    "output_database": EI,
                    "output_code": out,
                    "input_database": db,
                    "input_code": code,
                    "amount": amount,
                    "type": kind,
                }
                for out, db, code, amount, kind in rows
            ]
        )

    def test_value_is_the_normalised_amount_times_multiplier_times_sign(self):
        values = RebuiltRecipe.entry_values(self._recipe(), self._snapshot())

        np.testing.assert_allclose(values, [1.0, -0.7, -0.25, 0.05])

    def test_a_source_missing_from_the_snapshot_is_refused(self):
        snapshot = self._snapshot()
        snapshot = snapshot[snapshot["input_code"] != "soy"]

        with pytest.raises(ValueError, match="not in the ecoinvent snapshot"):
            RebuiltRecipe.entry_values(self._recipe(), snapshot)

    def test_a_source_that_is_not_one_exchange_is_refused(self):
        snapshot = self._snapshot()
        snapshot = pd.concat([snapshot, snapshot.iloc[[1]]], ignore_index=True)

        with pytest.raises(ValueError, match="one exchange each"):
            RebuiltRecipe.entry_values(self._recipe(), snapshot)


class TestReadWrite:
    def test_write_then_read_returns_the_same_recipe(self, tmp_path: Path):
        sp_data = _sp_data()
        recipe = RebuiltRecipe.from_sp_data(sp_data, _final_frame(sp_data))

        recipe.write(tmp_path)

        again = RebuiltRecipe.read(tmp_path)
        assert again is not None
        pd.testing.assert_frame_equal(again.frame, recipe.frame)
        assert not (tmp_path / (RebuiltRecipe.FILENAME + ".partial")).exists()

    def test_a_package_without_a_recipe_reads_as_none(self, tmp_path: Path):
        assert RebuiltRecipe.read(tmp_path) is None
