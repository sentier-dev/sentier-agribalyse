"""``WasteTreatmentDummyFixer`` — convert ``bw_simapro_csv`` waste-treatment stubs to real productions.

When a SimaPro CSV process has a ``Waste treatment`` block but no
``Products`` block, ``bw_simapro_csv.brightway.lci_to_brightway`` injects
a stub production exchange:

    {"type": "production", "functional": False, "amount": 0,
     "comment": "Dummy edge inserted to stop auto-generation of unitary production edge"}

The stub exists purely to suppress bw2io's auto-generated unitary edge.
Two downstream consequences cause a singular technosphere matrix:

* ``ProductionReclassifier`` flips every ``functional=False`` production
  to ``technosphere``, so the activity loses its production exchange.
* ``drop_unlinked`` strips zero-amount exchanges, so the stub disappears
  even if the reclassifier is bypassed.

Either way, the activity ends up with no production and a zero diagonal
in the technosphere matrix → ``MatrixRankWarning: Matrix is exactly
singular``. This transform identifies the stubs by their canonical
comment and rewrites them in-place to ``functional=True, amount=1``,
which makes them survive both the reclassifier and ``drop_unlinked``
and gives the matrix a +1 pivot on the diagonal.

Because ``bw_simapro_csv`` already marks the waste-input technosphere
exchange as ``functional=True`` (the SimaPro convention for waste
treatment), promoting the production to functional gives the dataset two
functional edges. ``multifunctional`` then classifies the dataset as
``type=multifunctional`` and walks every functional edge looking for a
``properties["manual_allocation"]`` value (the default allocation
property for SimaPro-derived imports). To satisfy that walk without
altering impacts (dummies are empty by construction), we attach
``manual_allocation: 1.0`` to the promoted production and
``manual_allocation: 0.0`` to every other functional edge in the same
dataset, sending 100% of the (zero) impacts to the dummy production.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.logging import Logging

_DUMMY_MARKER = "dummy edge inserted to stop auto-generation of unitary production edge"
_ALLOC_PROPERTY = "manual_allocation"


@dataclass(frozen=True)
class WasteTreatmentDummyFixer:
    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(self, sp) -> dict[str, int]:
        n = 0
        for ds in sp.data:
            promoted: list[dict] = []
            for e in ds.get("exchanges", []):
                if e.get("type") != "production":
                    continue
                if e.get("functional") is not False:
                    continue
                comment = (e.get("comment") or "").lower()
                if _DUMMY_MARKER not in comment:
                    continue
                e["amount"] = 1.0
                e["functional"] = True
                e["properties"] = {**(e.get("properties") or {}), _ALLOC_PROPERTY: 1.0}
                promoted.append(e)
                n += 1

            if not promoted:
                continue

            for other in ds.get("exchanges", []):
                if other in promoted:
                    continue
                if other.get("functional") is not True:
                    continue
                other["properties"] = {
                    **(other.get("properties") or {}),
                    _ALLOC_PROPERTY: 0.0,
                }
        self._log.info("transforms.waste_treatment_dummy_fixed", n=n)
        return {"dummies_promoted": n}
