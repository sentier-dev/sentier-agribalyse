"""``RegistryBuildPipeline`` — own-pipeline wrapper around ``RegistryBuilder``.

Exists so the build is a first-class pipeline addressable by CLI
(``python -m cli.build_registry``) and by other pipelines that depend
on the registry being current.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import Settings
from core import StepTimer
from core.logging import Logging
from registry import RegistryBuilder


@dataclass(frozen=True)
class RegistryBuildPipeline:
    settings: Settings

    @property
    def _log(self):
        return Logging.get(__name__)

    def run(self) -> dict[str, Any]:
        with StepTimer(self._log, "pipeline.registry.build"):
            return RegistryBuilder(self.settings).build()
