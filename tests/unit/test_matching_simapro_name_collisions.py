"""``SimaproNameCollisions`` — the block SimaPro holds behind a shared name."""

from __future__ import annotations

import json

from matching.ecoinvent_catalog import EcoinventActivityRef, EcoinventCatalog
from matching.simapro_form_resolver import SimaproFormResolver
from matching.simapro_name_collisions import SimaproNameCollisions

EI = "ecoinvent-3.9.1-cutoff"


def _ref(code, name, location="GLO", product="pesticide, unspecified") -> EcoinventActivityRef:
    return EcoinventActivityRef(
        db=EI, code=code, name=name, unit="kilogram", location=location, reference_product=product
    )


REAL = _ref("real", "market for pesticide, unspecified")
DCP = _ref("dcp", "dichloropropene to generic market for pesticide, unspecified")
OTHER = _ref("other", "glyphosate to generic market for pesticide, unspecified")


def _collisions(tmp_path, entries, *refs) -> SimaproNameCollisions:
    path = tmp_path / "edge-labels.json"
    path.write_text(
        json.dumps(
            {"replace": [{"source": {"name": s}, "target": {"name": t}} for s, t in entries]}
        )
    )
    catalog = EcoinventCatalog(db_names=(EI,), activities=tuple(refs))
    return SimaproNameCollisions(
        resolver=SimaproFormResolver(catalog=catalog, ei_db=EI), edge_labels_path=path
    )


SHORT = "Pesticide, unspecified {GLO}| market for | Cut-off, S - Copied from Ecoinvent U"
DCP_LABEL = (
    "Pesticide, unspecified {GLO}| dichloropropene to generic market for pesticide, "
    "unspecified | Cut-off, U"
)
REAL_LABEL = "Pesticide, unspecified {GLO}| market for pesticide, unspecified | Cut-off, U"


class TestStandIn:
    def test_activity_without_a_block_gets_the_block_filed_under_its_short_name(self, tmp_path):
        collisions = _collisions(tmp_path, [(SHORT, DCP_LABEL)], REAL, DCP)

        assert collisions.stand_in(REAL).code == "dcp"

    def test_activity_with_its_own_block_keeps_it(self, tmp_path):
        full = "Pesticide, unspecified {GLO}| market for pesticide, unspecified | Cut-off, S"
        collisions = _collisions(tmp_path, [(SHORT, DCP_LABEL), (full, REAL_LABEL)], REAL, DCP)

        assert collisions.stand_in(REAL) is None
        assert collisions.stand_in(DCP) is None

    def test_label_must_be_the_start_of_the_activity_name(self, tmp_path):
        collisions = _collisions(tmp_path, [(SHORT, DCP_LABEL)], REAL, DCP, OTHER)

        assert collisions.stand_in(OTHER) is None

    def test_other_location_is_not_a_stand_in(self, tmp_path):
        elsewhere = _ref("ch", "market for pesticide, unspecified", location="CH")
        collisions = _collisions(tmp_path, [(SHORT, DCP_LABEL)], REAL, DCP, elsewhere)

        assert collisions.stand_in(elsewhere) is None

    def test_block_of_another_product_is_not_a_stand_in(self, tmp_path):
        feed = _ref("feed", "barley grain to generic market for energy feed", product="energy feed")
        protein = _ref("protein", "market for protein feed", product="pesticide, unspecified")
        entries = [
            (
                SHORT,
                "Pesticide, unspecified {GLO}| barley grain to generic market for energy feed | "
                "Cut-off, U",
            )
        ]
        collisions = _collisions(tmp_path, entries, feed, protein)

        assert collisions.stand_in(protein) is None

    def test_two_candidate_blocks_give_no_opinion(self, tmp_path):
        second = "Pesticide, unspecified {GLO}| market | Cut-off, S - Copied from Ecoinvent U"
        other_label = (
            "Pesticide, unspecified {GLO}| glyphosate to generic market for pesticide, "
            "unspecified | Cut-off, U"
        )
        collisions = _collisions(
            tmp_path, [(SHORT, DCP_LABEL), (second, other_label)], REAL, DCP, OTHER
        )

        assert collisions.stand_in(REAL) is None

    def test_missing_edge_label_file(self, tmp_path):
        catalog = EcoinventCatalog(db_names=(EI,), activities=(REAL, DCP))
        collisions = SimaproNameCollisions(
            resolver=SimaproFormResolver(catalog=catalog, ei_db=EI),
            edge_labels_path=tmp_path / "ghost.json",
        )

        assert collisions.stand_in(REAL) is None
