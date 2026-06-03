"""``ProductionReclassifier`` — flips ``functional=False`` production rows to technosphere.

Agribalyse 3.2 SimaPro CSVs encode waste outputs and unallocated
co-products as ``production`` exchanges with ``functional: False``. In
the ecoinvent cutoff convention these belong on the technosphere side
(positive inputs into the corresponding waste-treatment activity), not
on the production side. Sign stays positive — the waste polluter is
also the consumer of the treatment activity.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.logging import Logging


@dataclass(frozen=True)
class ProductionReclassifier:
    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(self, sp) -> dict[str, int]:
        n = 0
        for ds in sp.data:
            for e in ds.get("exchanges", []):
                if e.get("type") == "production" and e.get("functional") is False:
                    e["type"] = "technosphere"
                    n += 1
        self._log.info("transforms.production_reclassified", n=n)
        return {"reclassified": n}
