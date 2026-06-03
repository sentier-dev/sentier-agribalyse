"""``RegionalSourceNameSnapshotter`` — preserve the pre-flowmap name.

The biosphere flowmap (``BiosphereFlowmapApplier``) rewrites
``exc["name"]`` from the AGB-side SimaPro spelling (``Water, IN``) to
the bio3 canonical name (``Water``), erasing the country-of-extraction
suffix that downstream
:class:`matching.regional_suffix_applier.RegionalSuffixApplier` needs
to construct the synthetic ``@<region>`` matrix-row identity.

This transform runs once, RIGHT BEFORE the flowmap, and records every
biosphere exchange's name under ``exc["_regional_source_name"]`` so
the applier can recover the regional suffix even after the canonical
rename has happened. Standalone class, frozen dataclass, no
constructor args — pure batch mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class RegionalSourceNameSnapshotter:
    """Tag every biosphere exchange with its pre-flowmap source name."""

    FIELD: ClassVar[str] = "_regional_source_name"

    def apply(self, sp_data: list[dict]) -> dict[str, int]:
        n_snapshotted = 0
        for ds in sp_data:
            for exc in ds.get("exchanges", ()):
                if exc.get("type") != "biosphere":
                    continue
                name = exc.get("name")
                if not isinstance(name, str) or not name:
                    continue
                if self.FIELD in exc:
                    continue
                exc[self.FIELD] = name
                n_snapshotted += 1
        return {"n_snapshotted": n_snapshotted}
