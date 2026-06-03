"""Pipeline-level settings. Frozen dataclass; pass to every component."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar

from config.paths import Paths


@dataclass(frozen=True)
class Settings:
    """End-to-end pipeline configuration. Immutable; use ``with_*`` to evolve."""

    DEFAULT_ECOINVENT_FOR_AGRIBALYSE: ClassVar[Mapping[str, str]] = MappingProxyType(
        {"3.2": "3.9.1", "4.0-beta": "3.11"}
    )

    agribalyse_version: str = "3.2"
    ecoinvent_version: str | None = None
    ecoinvent_system_model: str = "cutoff"
    ef_version: str = "3.1"

    project_name: str | None = None
    biosphere_db_name: str = "ecoinvent-3.9.1-biosphere"
    agribalyse_db_name: str | None = None
    ef_db_name: str = "ef"

    apply_llm_overrides: bool = True
    """``--no-llm`` toggles this. Gates BOTH LLM overrides (tier 10) AND the
    curated synonym fallback (tier 11) — fix 1.j."""

    apply_transitive_layer: bool = False
    """The 'ecoinvent flows - EF v3.1 map' transitive sheet (806 rows). Off by default."""

    paths: Paths = field(default_factory=Paths)

    @classmethod
    def default_ecoinvent_for(cls, agribalyse_version: str) -> str:
        try:
            return cls.DEFAULT_ECOINVENT_FOR_AGRIBALYSE[agribalyse_version]
        except KeyError as exc:
            supported = ", ".join(sorted(cls.DEFAULT_ECOINVENT_FOR_AGRIBALYSE))
            raise ValueError(
                f"No ADEME-native ecoinvent pairing registered for Agribalyse "
                f"{agribalyse_version!r}. Known versions: {supported}."
            ) from exc

    @property
    def resolved_ecoinvent_version(self) -> str:
        return self.ecoinvent_version or self.default_ecoinvent_for(self.agribalyse_version)

    @property
    def resolved_project_name(self) -> str:
        return self.project_name or f"agribalyse-{self.agribalyse_version}-sentier"

    @property
    def resolved_agribalyse_db_name(self) -> str:
        return self.agribalyse_db_name or f"agribalyse-{self.agribalyse_version}"

    @property
    def ecoinvent_db_name(self) -> str:
        return f"ecoinvent-{self.resolved_ecoinvent_version}-{self.ecoinvent_system_model}"

    def with_ecoinvent(self, version: str | None) -> Settings:
        return replace(self, ecoinvent_version=version)

    def with_no_llm(self) -> Settings:
        return replace(self, apply_llm_overrides=False)
