"""``OrphanActivityPurger`` — drop process activities that cannot produce a non-singular matrix column.

After ``ProductionReclassifier`` has flipped ``functional=False``
productions to technosphere, some datasets are left with **no
production exchange and no ``functional=True`` technosphere edge** —
meaning they have no functional output of any kind. In the technosphere
matrix these become a column of negative consumption entries with a
zero diagonal, which makes the matrix singular and the LU
factorization fail with ``MatrixRankWarning``.

These activities can't be solved by any LCA-side fix — there's no
producer to wire. The only safe option is to drop them before the
``ScoringPackage`` emit step so the matrix stays solvable.

This runs AFTER ``ProductionReclassifier`` (which is what creates this
state) and AFTER ``WasteTreatmentDummyFixer`` (which rescues the
waste-treatment stubs that would otherwise look like orphans).
Consumers of dropped activities become unlinked technosphere
exchanges, which the existing ``UnlinkedExporter`` will report —
nothing is silently lost.

``type='product'`` datasets are **explicitly excluded** from purge
candidacy. ``bw_simapro_csv`` materialises every SimaPro product as a
zero-exchange ``ActivityDataset`` whose only purpose is to be the
target of process exchanges' ``input`` tuples. Products don't
contribute matrix columns and don't need a production exchange — they
are vertices, not processes. Dropping them leaves dangling references
in surviving processes that survive ``drop_unlinked`` (the ``input``
field is set, so the strategy doesn't see them as unlinked) and then
fail in ``Database.process`` with
``UnknownObject: ... is invalid - one of these objects is unknown``.

When given a :class:`Settings`, the purger also writes
``dashboard/orphan_activities.parquet`` listing every dropped activity
with enough context (name, code, type, unit, exchange counts) to
decide which deserve a curated rescue (synthetic production / mapping
to ecoinvent) and which are genuine data artifacts to leave purged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import Settings
from core.logging import Logging


@dataclass(frozen=True)
class OrphanActivityPurger:
    settings: Settings | None = None

    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(self, sp) -> dict[str, int]:
        kept: list[dict] = []
        dropped_records: list[dict[str, Any]] = []
        for ds in sp.data:
            if ds.get("type") == "product" or self._has_functional_output(ds):
                kept.append(ds)
            else:
                dropped_records.append(self._dropped_row(ds))
        n_dropped = len(dropped_records)
        if n_dropped:
            sp.data = kept
            self._log.info(
                "transforms.orphan_activities_dropped",
                n=n_dropped,
                samples=[r["name"] for r in dropped_records[:5]],
            )
            self._write_diagnostic(dropped_records)
        else:
            self._log.info("transforms.orphan_activities_dropped", n=0)
        return {"orphans_dropped": n_dropped}

    @staticmethod
    def _has_functional_output(ds: dict) -> bool:
        for e in ds.get("exchanges", []):
            if e.get("type") == "production":
                return True
            if e.get("type") == "technosphere" and e.get("functional") is True:
                return True
        return False

    @staticmethod
    def _dropped_row(ds: dict) -> dict[str, Any]:
        """Snapshot what we know about an orphan, so a reviewer can decide
        whether to curate a rescue or leave it purged."""
        exchanges = list(ds.get("exchanges", []))
        edge_counts: dict[str, int] = {}
        for e in exchanges:
            t = e.get("type", "<unknown>")
            edge_counts[t] = edge_counts.get(t, 0) + 1
        # First few input names give a reviewer a quick sense of what the
        # dataset *consumed* — usually enough to recognise it.
        sample_inputs = [
            e.get("name", "") for e in exchanges if e.get("type") in ("technosphere", "biosphere")
        ][:5]
        return {
            "name": ds.get("name", "<no name>"),
            "code": ds.get("code", ""),
            "database": ds.get("database", ""),
            "type": ds.get("type", ""),
            "unit": ds.get("unit", ""),
            "location": ds.get("location", ""),
            "reference_product": ds.get("reference product", ""),
            "n_exchanges": len(exchanges),
            "edge_counts": edge_counts,
            "sample_inputs": sample_inputs,
            "comment": (ds.get("comment", "") or "")[:300],
        }

    def _write_diagnostic(self, dropped_records: list[dict[str, Any]]) -> None:
        if self.settings is None:
            return
        path = self.settings.paths.dashboard_orphan_activities
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(dropped_records)
        # ``edge_counts``/``sample_inputs`` are dict / list — cast to JSON
        # strings so parquet round-trips cleanly across pyarrow versions.
        for col in ("edge_counts", "sample_inputs"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        df.to_parquet(path, index=False)
        self._log.info("transforms.orphan_diagnostic.written", path=str(path), n=len(df))
