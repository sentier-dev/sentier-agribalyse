"""``CoverageReporter`` — pre-purge and post-purge biosphere/technosphere link rates.

Closes fix 1.h: report both states so the user sees what survived the
matrix-squareness purge. After REFACTOR_FINAL F5 there's no on-disk DB
walk left — coverage is computed from the in-memory ``sp.data`` only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageSnapshot:
    label: str
    bio_total: int
    bio_linked: int
    tech_total: int
    tech_linked: int

    @property
    def bio_rate(self) -> float:
        return round(100 * self.bio_linked / max(self.bio_total, 1), 2)

    @property
    def tech_rate(self) -> float:
        return round(100 * self.tech_linked / max(self.tech_total, 1), 2)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "bio_total": self.bio_total,
            "bio_linked": self.bio_linked,
            "bio_rate": self.bio_rate,
            "tech_total": self.tech_total,
            "tech_linked": self.tech_linked,
            "tech_rate": self.tech_rate,
        }


@dataclass(frozen=True)
class CoverageReporter:
    """Compute coverage from an in-memory ``sp.data`` snapshot."""

    @staticmethod
    def from_sp_data(label: str, sp_data: list[dict]) -> CoverageSnapshot:
        bio_total = bio_linked = tech_total = tech_linked = 0
        for p in sp_data:
            for e in p.get("exchanges", []):
                t = e.get("type")
                if t == "biosphere":
                    bio_total += 1
                    if "input" in e:
                        bio_linked += 1
                elif t == "technosphere":
                    tech_total += 1
                    if "input" in e:
                        tech_linked += 1
        return CoverageSnapshot(
            label=label,
            bio_total=bio_total,
            bio_linked=bio_linked,
            tech_total=tech_total,
            tech_linked=tech_linked,
        )
