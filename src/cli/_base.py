"""``BaseCli`` — shared scaffolding for command-line entrypoint classes.

Subclasses override ``parser()`` and ``execute(args)``. ``run(argv)``
configures logging, parses argv, dispatches to ``execute``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

from config import Settings
from core.logging import Logging

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass
class BaseCli:
    """Common CLI behaviour. Concrete CLIs subclass and override."""

    settings: Settings = field(default_factory=Settings)

    PROG: ClassVar[str] = "cli"
    DESCRIPTION: ClassVar[str] = ""

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        return argparse.ArgumentParser(prog=cls.PROG, description=cls.DESCRIPTION)

    def configure_logging(self) -> None:
        Logging.configure(level=logging.INFO)

    def run(self, argv: list[str] | None = None) -> int:
        argv = argv if argv is not None else sys.argv[1:]
        args = self.parser().parse_args(argv)
        self.configure_logging()
        try:
            self.execute(args)
        except Exception as exc:
            Logging.get(self.PROG).exception("cli.failed", error=str(exc))
            return 1
        return 0

    def execute(self, args: argparse.Namespace) -> None:  # pragma: no cover — overridden
        raise NotImplementedError
