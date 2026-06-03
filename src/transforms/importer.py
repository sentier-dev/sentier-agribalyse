"""``SimaProImporter`` — wraps ``SimaProCsvParser`` with a pickle cache.

The parser itself lives in ``transforms/sp_csv_parser.py`` and uses
``bw_simapro_csv`` directly — bw2io is no longer imported here.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

from config import Settings
from core.logging import Logging, StepTimer
from transforms.sp_csv_parser import ParsedSimaProCsv, SimaProCsvParser


@dataclass(frozen=True)
class SimaProImporter:
    """Read AGB SimaPro CSV. Caches the parsed object to a pickle for re-runs."""

    settings: Settings

    @property
    def _log(self):
        return Logging.get(__name__)

    def load(self) -> ParsedSimaProCsv:
        cache_path = self.settings.paths.importer_cache_pkl
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with StepTimer(self._log, "csv.load.cache_hit", path=cache_path.name):
                t0 = time.time()
                sp = pickle.loads(cache_path.read_bytes())
                self._log.info(
                    "csv.load.cache_hit",
                    processes=len(sp.data),
                    elapsed_s=round(time.time() - t0, 2),
                )
            return sp

        csv = self.settings.paths.agribalyse_csv
        with StepTimer(
            self._log, "csv.load.parse", path=csv.name, size_mb=round(csv.stat().st_size / 1e6, 1)
        ):
            sp = SimaProCsvParser(
                csv_path=csv,
                database_name=self.settings.resolved_agribalyse_db_name,
            ).parse()
        cache_path.write_bytes(pickle.dumps(sp))
        self._log.info("csv.cache.written", path=cache_path.name)
        return sp
