"""From ecoinvent credentials to the four ``source/`` snapshots.

The runtime never touches Brightway: it reads four files under
``source/`` that hold the licensed ecoinvent content (exchanges, activity
labels, elementary-flow labels, the EF v3.1 factors ecoinvent publishes
against its own flows). This module produces them:

1. ``EcoinventReleaseImporter`` — download ecoinvent with the user's
   credentials and import it into a private Brightway project (``bw2io``).
2. ``EcoinventExchangesSnapshotter`` (in ``cli.snapshot_ecoinvent_exchanges``)
   — exchanges to parquet, straight from that project's SQLite.
3. ``EcoinventMetadataSnapshotter`` — the three JSON files, in the exact
   shapes ``EcoinventCatalogBuilder``, ``BiosphereRegistryBuilder`` and
   ``MethodCfRegistryBuilder`` read.

``bw2data`` and ``bw2io`` are passed in, never imported here: the
``bootstrap`` extra is the only place they are needed, and the CLI points
Brightway at a private directory before it imports them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from cli.snapshot_ecoinvent_exchanges import EcoinventExchangesSnapshotter, EcoinventSqliteReader
from config import Settings
from core.logging import Logging


class EcoinventCredentialsMissingError(RuntimeError):
    """``ECOINVENT_USERNAME`` / ``ECOINVENT_PASSWORD`` are not set."""


@dataclass(frozen=True)
class EcoinventReleaseImporter:
    """Make sure the ecoinvent release is in the bootstrap Brightway project."""

    settings: Settings
    bd: Any
    bi: Any

    PROJECT: ClassVar[str] = "dds-bootstrap"

    def ensure(self, username: str | None, password: str | None) -> bool:
        """Import the release unless it is already there. Returns ``True`` if it imported."""
        self.bd.projects.set_current(self.PROJECT)
        if self.settings.ecoinvent_db_name in self.bd.databases:
            return False
        if not username or not password:
            raise EcoinventCredentialsMissingError(
                "ECOINVENT_USERNAME and ECOINVENT_PASSWORD must be set (in .env at the "
                "repository root, or in the environment). They are the login of your "
                "ecoinvent licence on https://ecoquery.ecoinvent.org/."
            )
        # Single process: with the spawn start method (Windows, macOS) bw2io's
        # workers re-import the main module and re-run the import recursively.
        self.bi.import_ecoinvent_release(
            version=self.settings.resolved_ecoinvent_version,
            system_model=self.settings.ecoinvent_system_model,
            username=username,
            password=password,
            use_mp=False,
        )
        if self.settings.ecoinvent_db_name not in self.bd.databases:
            raise RuntimeError(
                f"bw2io finished but {self.settings.ecoinvent_db_name!r} is not in the project."
            )
        return True


@dataclass(frozen=True)
class EcoinventMetadataSnapshotter:
    """Write the activity, elementary-flow and EF-method snapshots."""

    settings: Settings
    bd: Any

    def snapshot(self) -> dict[str, int]:
        activities = self._activities()
        flows = self._flows()
        methods = self._methods()
        paths = self.settings.paths
        self._write(
            paths.ecoinvent_activities_json,
            {"db_name": self.settings.ecoinvent_db_name, "activities": activities},
        )
        self._write(paths.ecoinvent_biosphere_flows_json, {"flows": flows})
        self._write(
            paths.ef_methods_snapshot_json,
            {"ef_db_name": self.settings.ef_db_name, "methods": methods},
        )
        return {
            "activities": len(activities),
            "biosphere_flows": len(flows),
            "methods": len(methods),
            "method_cfs": sum(len(m["inherited_cfs"]) for m in methods.values()),
        }

    def _activities(self) -> list[dict[str, Any]]:
        rows = [
            {
                "code": node["code"],
                "name": node["name"],
                "unit": node.get("unit"),
                "location": node.get("location"),
                "reference_product": node.get("reference product"),
            }
            for node in self.bd.Database(self.settings.ecoinvent_db_name)
        ]
        return sorted(rows, key=lambda r: r["code"])

    def _flows(self) -> list[dict[str, Any]]:
        rows = [
            {
                "code": node["code"],
                "name": node["name"],
                "categories": list(node.get("categories") or ()),
                "unit": node.get("unit"),
                "cas": node.get("CAS number"),
                "synonyms": list(node.get("synonyms") or ()),
            }
            for node in self.bd.Database(self.settings.biosphere_db_name)
        ]
        return sorted(rows, key=lambda r: r["code"])

    def _methods(self) -> dict[str, dict[str, Any]]:
        prefix = (
            f"ecoinvent-{self.settings.resolved_ecoinvent_version}",
            f"EF v{self.settings.ef_version}",
        )
        keys = sorted(key for key in self.bd.methods if tuple(key[:2]) == prefix)
        if not keys:
            raise RuntimeError(
                f"No {prefix[1]} method under {prefix[0]!r} in the Brightway project: "
                f"the ecoinvent import did not load its LCIA methods."
            )
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            cfs = [self._cf(row) for row in self.bd.Method(key).load()]
            out[" / ".join(key)] = {
                "key": list(key),
                "inherited_cfs": sorted(cfs, key=lambda r: (r["db"], r["code"])),
            }
        return out

    def _cf(self, row: Any) -> dict[str, Any]:
        flow, amount = row[0], row[1]
        if isinstance(flow, int):  # bw2data 4 keys factors by node id
            node = self.bd.get_node(id=flow)
            flow = (node["database"], node["code"])
        return {"db": str(flow[0]), "code": str(flow[1]), "amount": float(amount)}

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class EcoinventBootstrapper:
    """Import if needed, then write the four snapshots."""

    settings: Settings
    bd: Any
    bi: Any

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def run(self, username: str | None, password: str | None) -> dict[str, Any]:
        imported = EcoinventReleaseImporter(settings=self.settings, bd=self.bd, bi=self.bi).ensure(
            username, password
        )
        sqlite_path = Path(self.bd.projects.dir) / "lci" / "databases.db"
        exchanges = EcoinventExchangesSnapshotter(
            reader=EcoinventSqliteReader(
                sqlite_path=sqlite_path, db_name=self.settings.ecoinvent_db_name
            ),
            output_path=self.settings.paths.ecoinvent_exchanges_parquet,
        ).snapshot()
        counts = EcoinventMetadataSnapshotter(settings=self.settings, bd=self.bd).snapshot()
        report = {"imported": imported, "exchanges": exchanges["n_rows"], **counts}
        self._log.info("bootstrap.ecoinvent.done", **report)
        return report
