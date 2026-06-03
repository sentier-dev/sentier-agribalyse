"""``UnlinkedExporter`` — write residual unlinked tech/bio exchanges to ``unlinked/``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from config import Settings
from core.logging import Logging
from domain import Bucket


@dataclass(frozen=True)
class UnlinkedExporter:
    settings: Settings
    registry: object = field(default=None)  # MappingRegistry | None — avoids circular import

    @property
    def _log(self):
        return Logging.get(__name__)

    def export(self, sp_data: list[dict]) -> dict[str, int]:
        out_dir = self.settings.paths.unlinked
        out_dir.mkdir(parents=True, exist_ok=True)

        tech: dict[str, int] = {}
        bio: dict[str, dict] = {}
        for proc in sp_data:
            for exc in proc.get("exchanges", []):
                if "input" in exc:
                    continue
                name = exc.get("name", "")
                if not name:
                    continue
                if exc.get("type") == "technosphere":
                    if "[dummy]" in name.lower():
                        continue
                    tech[name] = tech.get(name, 0) + 1
                elif exc.get("type") == "biosphere":
                    bio.setdefault(
                        name,
                        {
                            "name": name,
                            "unit": exc.get("unit", ""),
                            "categories": list(exc.get("categories", ())),
                        },
                    )

        tech_path = out_dir / "technosphere_unlinked.json"
        tech_path.write_text(
            json.dumps(
                [{"name": n, "count": c} for n, c in sorted(tech.items(), key=lambda x: -x[1])],
                indent=1,
                ensure_ascii=False,
            )
        )

        if bio:
            rows = []
            for name, info in sorted(bio.items()):
                cats = info.get("categories", [])
                top = cats[0] if len(cats) >= 1 else ""
                sub = cats[1] if len(cats) >= 2 else ""
                reason, provenance = self._unlink_reason(name, top, sub)
                rows.append(
                    {
                        "source_name": name,
                        "source_unit": info.get("unit", ""),
                        "source_top_cat": top,
                        "source_sub_cat": sub,
                        "unlink_reason": reason,
                        "unlink_provenance": provenance,
                    }
                )
            bio_path = out_dir / "biosphere_unlinked.xlsx"
            pd.DataFrame(rows).to_excel(bio_path, index=False)

        self._log.info(
            "unlinked.exported",
            tech_unique=len(tech),
            tech_exchanges=sum(tech.values()),
            bio_unique=len(bio),
        )
        return {"tech_unique": len(tech), "bio_unique": len(bio)}

    def _unlink_reason(self, name: str, top_cat: str, sub_cat: str) -> tuple[str, str]:
        """Return (human-readable reason, provenance) for why this flow is unlinked."""
        if self.registry is None:
            return "", ""

        bucket = Bucket.from_categories((top_cat, sub_cat))
        df = self.registry.unmatchable_index.df

        if not df.empty:
            mask = (df["source_name"].str.strip().str.lower() == name.strip().lower()) & (
                df["source_top_bucket"] == bucket.value
            )
            hits = df[mask]
            if not hits.empty:
                row = hits.iloc[0]
                notes = str(row.get("notes", "") or "").strip()
                prov = str(row.get("provenance", "") or "").strip()
                if notes:
                    return notes, prov
                # synthesise a reason from provenance when notes is absent
                if "neither" in prov or "placeholder" in prov:
                    return (
                        "Neither in ecoinvent v3.9.1 nor EF v3.1 "
                        "(Agribalyse placeholder flow classification)",
                        prov,
                    )
                return f"Marked unmatchable by {prov}", prov

        return "No matching biosphere flow found in any registered database", ""
