"""Unit tests for ``PerennialLifecycleFilter``.

The filter drops only the acid-relevant biosphere edges (NH3) on
perennial-crop non-productive / grubbing-up lifecycle activities.
ADEME's published synthese excludes these emissions (capital-goods
convention) but does *include* their biogenic CO2 and land use
contributions, so the filter must be surgical at the (activity,
flow) level — not a wholesale technosphere edge cut.
"""

from __future__ import annotations

import pandas as pd

from scoring.exchange_frame import ExchangeFrame
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.perennial_lifecycle_filter import PerennialLifecycleFilter

_DB = "agribalyse-3.2"
_AGGREGATOR_CODE = "AGRIBALU000000003101544"
_NON_PROD_CODE = "1872cd1e5d1046fa842f11496a6165b2"
_GRUBBING_CODE = "b0fd3897b1b14a7997e03df6b7951585"
_PRODUCTIVE_CODE = "EI3CQUNI000025017100986"

_AGGREGATOR_ID = ExchangeFrameBuilder.flow_id_for((_DB, _AGGREGATOR_CODE))
_NON_PROD_ID = ExchangeFrameBuilder.flow_id_for((_DB, _NON_PROD_CODE))
_GRUBBING_ID = ExchangeFrameBuilder.flow_id_for((_DB, _GRUBBING_CODE))
_PRODUCTIVE_ID = ExchangeFrameBuilder.flow_id_for((_DB, _PRODUCTIVE_CODE))

_NH3_BIO3 = ("biosphere3", "nh3-uuid")
_NH3_EI = ("ecoinvent-3.9.1-biosphere", "nh3-uuid")
_NOX_BIO3 = ("biosphere3", "nox-uuid")
_CO2_BIO3 = ("biosphere3", "co2-bio-uuid")

_NH3_ID = ExchangeFrameBuilder.flow_id_for(_NH3_BIO3)
_NH3_EI_ID = ExchangeFrameBuilder.flow_id_for(_NH3_EI)
_NOX_ID = ExchangeFrameBuilder.flow_id_for(_NOX_BIO3)
_CO2_ID = ExchangeFrameBuilder.flow_id_for(_CO2_BIO3)


def _frame(rows: list[tuple]) -> ExchangeFrame:
    df = pd.DataFrame(
        rows,
        columns=["output_id", "input_id", "amount", "edge_type", "is_biosphere"],
    )
    df["output_id"] = df["output_id"].astype("int64")
    df["input_id"] = df["input_id"].astype("int64")
    df["amount"] = df["amount"].astype("float64")
    df["edge_type"] = df["edge_type"].astype("string")
    df["is_biosphere"] = df["is_biosphere"].astype(bool)
    return ExchangeFrame(df=df)


def _bio_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "database": [_NH3_BIO3[0], _NH3_EI[0], _NOX_BIO3[0], _CO2_BIO3[0]],
            "code": [_NH3_BIO3[1], _NH3_EI[1], _NOX_BIO3[1], _CO2_BIO3[1]],
            "name": ["Ammonia", "Ammonia", "Nitrogen oxides", "Carbon dioxide, biogenic"],
        }
    )


def _sp_data() -> list[dict]:
    return [
        {
            "database": _DB,
            "code": _AGGREGATOR_CODE,
            "name": "Black pepper, conventional, at farm gate {VN}",
            "type": "process",
            "exchanges": [],
        },
        {
            "database": _DB,
            "code": _NON_PROD_CODE,
            "name": "Black pepper berries, fresh, non productive years {VN} U",
            "type": "process",
            "exchanges": [],
        },
        {
            "database": _DB,
            "code": _GRUBBING_CODE,
            "name": "Black pepper berries, fresh, grubbing up {VN} U",
            "type": "process",
            "exchanges": [],
        },
        {
            "database": _DB,
            "code": _PRODUCTIVE_CODE,
            "name": "Black pepper bells, fresh, productive years {VN}",
            "type": "process",
            "exchanges": [],
        },
    ]


class TestPerennialLifecycleFilter:
    def test_drops_ammonia_edges_on_target_activities(self):
        frame = _frame(
            [
                (_NON_PROD_ID, _NON_PROD_ID, 1.0, "production", False),
                (_GRUBBING_ID, _GRUBBING_ID, 1.0, "production", False),
                (_PRODUCTIVE_ID, _PRODUCTIVE_ID, 1.0, "production", False),
                # Target: NH3 on non-prod (drop) — bio3 + ecoinvent-bio.
                (_NON_PROD_ID, _NH3_ID, 45.4, "biosphere", True),
                (_NON_PROD_ID, _NH3_EI_ID, 0.1, "biosphere", True),
                # Target: NH3 on grubbing-up (drop).
                (_GRUBBING_ID, _NH3_ID, 12.3, "biosphere", True),
                # Survives: NH3 on productive (productive is not a target).
                (_PRODUCTIVE_ID, _NH3_ID, 108.1, "biosphere", True),
                # Survives: NOx / biogenic CO2 / etc. on non-prod (we
                # only drop acid-relevant flows, not all biosphere edges).
                (_NON_PROD_ID, _NOX_ID, 24.7, "biosphere", True),
                (_NON_PROD_ID, _CO2_ID, 100.0, "biosphere", True),
            ]
        )
        purger = PerennialLifecycleFilter(biosphere_catalog=_bio_catalog())
        out, stats = purger.purge(frame, sp_data=_sp_data())

        assert stats["dropped"] == 3
        assert stats["n_distinct_activities"] == 2
        assert stats["n_distinct_flows"] == 2  # NH3 bio3 + NH3 ei

        bio = out.df[out.df["edge_type"] == "biosphere"]
        pairs = set(zip(bio["output_id"], bio["input_id"], strict=False))

        # Dropped: NH3 on non-prod and grubbing-up.
        assert (_NON_PROD_ID, _NH3_ID) not in pairs
        assert (_NON_PROD_ID, _NH3_EI_ID) not in pairs
        assert (_GRUBBING_ID, _NH3_ID) not in pairs

        # Survived: NH3 on productive, NOx/biogenic CO2 on non-prod.
        assert (_PRODUCTIVE_ID, _NH3_ID) in pairs
        assert (_NON_PROD_ID, _NOX_ID) in pairs
        assert (_NON_PROD_ID, _CO2_ID) in pairs

    def test_passthrough_when_no_target_activities_in_sp_data(self):
        sp = [
            {
                "database": _DB,
                "code": _PRODUCTIVE_CODE,
                "name": "Black pepper bells, fresh, productive years {VN}",
                "type": "process",
                "exchanges": [],
            }
        ]
        frame = _frame(
            [
                (_PRODUCTIVE_ID, _PRODUCTIVE_ID, 1.0, "production", False),
                (_PRODUCTIVE_ID, _NH3_ID, 108.1, "biosphere", True),
            ]
        )
        out, stats = PerennialLifecycleFilter(biosphere_catalog=_bio_catalog()).purge(
            frame, sp_data=sp
        )
        assert stats["dropped"] == 0
        assert out is frame

    def test_passthrough_when_no_matching_edges_in_frame(self):
        frame = _frame(
            [
                (_NON_PROD_ID, _NON_PROD_ID, 1.0, "production", False),
                # Only NOx / CO2 on the target activity — neither in
                # ACID_FLOW_NAMES → nothing dropped.
                (_NON_PROD_ID, _NOX_ID, 24.7, "biosphere", True),
                (_NON_PROD_ID, _CO2_ID, 100.0, "biosphere", True),
            ]
        )
        out, stats = PerennialLifecycleFilter(biosphere_catalog=_bio_catalog()).purge(
            frame, sp_data=_sp_data()
        )
        assert stats["dropped"] == 0
        assert out is frame

    def test_handles_empty_biosphere_catalog(self):
        frame = _frame(
            [
                (_NON_PROD_ID, _NON_PROD_ID, 1.0, "production", False),
                (_NON_PROD_ID, _NH3_ID, 45.4, "biosphere", True),
            ]
        )
        empty = pd.DataFrame({"database": [], "code": [], "name": []})
        out, stats = PerennialLifecycleFilter(biosphere_catalog=empty).purge(
            frame, sp_data=_sp_data()
        )
        assert stats["dropped"] == 0
        assert out is frame

    def test_does_not_touch_technosphere_edges(self):
        # Even if an aggregator -> non-prod technosphere edge exists,
        # the filter leaves it alone. The non-prod activity stays
        # consumed (its CO2 + land use contributions ride through).
        frame = _frame(
            [
                (_NON_PROD_ID, _NON_PROD_ID, 1.0, "production", False),
                (_AGGREGATOR_ID, _NON_PROD_ID, 3300.0, "technosphere", False),
                (_NON_PROD_ID, _NH3_ID, 45.4, "biosphere", True),
            ]
        )
        out, stats = PerennialLifecycleFilter(biosphere_catalog=_bio_catalog()).purge(
            frame, sp_data=_sp_data()
        )
        # NH3 dropped, technosphere intact.
        assert stats["dropped"] == 1
        tech = out.df[out.df["edge_type"] == "technosphere"]
        assert ((tech["output_id"] == _AGGREGATOR_ID) & (tech["input_id"] == _NON_PROD_ID)).any()
