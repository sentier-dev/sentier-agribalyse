"""``WasteTreatmentFunctionalPromoter`` — flip ``functional=True`` technosphere rows to production.

SimaPro encodes waste-treatment activities backwards: the "functional
unit" of a treatment process is the waste being *consumed* (i.e., the
tonne of biowaste sent to compost), not a positive product output. The
CSV therefore carries the activity's reference flow as a
``type='technosphere'`` row with ``functional=True`` and a positive
amount, and either ships no production row at all or ships a stub
``production`` row with ``amount=0`` (sometimes labelled
``functional=False``) just to keep the parser happy.

[`WasteTreatmentDummyFixer`](waste_treatment_dummy_fixer.py) already
rewrites the *commented* "Dummy edge inserted to stop auto-generation
of unitary production edge" stubs. Many AGB waste-treatment activities
(e.g. ``Compost, of biowaste (amendment) {RER}``,
``Treatment of biowaste, co-composting biowaste-greenwaste``,
``[Dummy] Composting grape marcs``, the various `[Dummy] Disposal …`
records — 421 cases in AGRIBALYSE 3.2) use the same pattern but lack
the canonical comment, so they slip past the fixer.

This transform handles the remainder. For any process whose only
``functional=True`` exchange sits on the technosphere side (and whose
production rows are absent or zero-amount), we promote that
technosphere row to ``type='production'`` in-place. The amount stays
positive, the ``functional`` flag stays ``True`` — the matrix builder
then puts the amount on the diagonal as a real production exchange,
preventing ``ProductionReclassifier`` and the dangling-edge pruner
from interpreting the activity as having no output.

Without this fix the activity sits in the matrix as a column with no
positive entry; downstream products that consume it can only satisfy
their demand by inverting an artificial loop through the unfixed
``technosphere`` edge, blowing up supply by 1000× or more (the
coconut-oil 54× backtest outlier was traced to AGB ``Compost, of
biowaste (amendment) {RER}`` here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logging import Logging


@dataclass(frozen=True)
class WasteTreatmentFunctionalPromoter:
    """Flip ``functional=True`` technosphere rows to production rows
    when the activity lacks a non-zero production exchange."""

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def apply(self, sp: Any) -> dict[str, int]:
        n_promoted = 0
        n_activities = 0
        n_zero_stubs_dropped = 0
        for ds in sp.data:
            excs = ds.get("exchanges", []) or ()
            production_rows = [
                e for e in excs if e.get("type") in ("production", "generic production")
            ]
            # Only treat the activity as already having a real functional
            # output when there is a non-zero ``functional=True`` production
            # row. Plain "any non-zero production" misclassifies SimaPro's
            # multi-output waste-treatment activities: the ecoinvent System
            # libraries emit a long tail of ``functional=False`` co-product
            # productions (sewage sludge, waste streams, etc.) for tracking
            # purposes, and the ONE actual functional unit sits on the
            # technosphere side as ``functional=True`` (the wastewater
            # being treated). Without promoting that technosphere row, the
            # ``ProductionReclassifier`` then flips every co-product back
            # to technosphere and the column ends up with no positive
            # diagonal — the matrix builder drops the activity and every
            # consumer's input dangles. Concrete case: ``treatment of
            # wastewater, average, wastewater treatment CH``
            # (EI3CQUNI…139) used by shrimp aquaculture.
            functional_production = any(
                (e.get("amount") or 0) != 0 and e.get("functional") is True for e in production_rows
            )
            if functional_production:
                # The activity already has a real production output —
                # whatever functional=True technosphere edges it carries
                # are genuine consumption (e.g. waste-treatment services
                # that the activity uses, not provides).
                continue
            functional_tech = [
                e for e in excs if e.get("type") == "technosphere" and e.get("functional") is True
            ]
            if not functional_tech:
                continue
            for e in functional_tech:
                e["type"] = "production"
                # Keep the positive amount and the functional flag — the
                # matrix builder reads ``edge_type`` for sign and the
                # Allocator reads ``allocation_factor`` for splitting.
                n_promoted += 1
            # Drop zero-amount production stubs left over from
            # ``bw_simapro_csv``'s parser. If they survive, the Allocator
            # would see two production rows (the new promoted one + the
            # zero stub) for the same activity and incorrectly classify
            # it as multifunctional. The pre-existing
            # ``WasteTreatmentDummyFixer`` would have caught these but
            # only when they carried the canonical "dummy edge" comment.
            kept = []
            for e in excs:
                if (
                    e.get("type") in ("production", "generic production")
                    and (e.get("amount") or 0) == 0
                ):
                    n_zero_stubs_dropped += 1
                    continue
                kept.append(e)
            ds["exchanges"] = kept
            n_activities += 1
        self._log.info(
            "transforms.waste_treatment_functional_promoted",
            n_promoted=n_promoted,
            n_activities=n_activities,
            n_zero_stubs_dropped=n_zero_stubs_dropped,
        )
        return {
            "promoted": n_promoted,
            "activities": n_activities,
            "zero_stubs_dropped": n_zero_stubs_dropped,
        }
