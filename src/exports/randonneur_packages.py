"""``RandonneurPackagesExporter`` — author the AGB-3.2 randonneur datapackages.

Replaces the legacy ``scripts/build_randonneur_packages.py``. Reads the
SimaPro CSV via ``SimaProImporter`` and writes:

* ``source/randonneur_packages/agribalyse-3.2-delete-aggregated-ecoinvent.json``
* ``source/randonneur_packages/agribalyse-3.2-restore-simapro-ecoinvent-names.json``
* ``to_review/agribalyse-3.2-biosphere-residuals-for-review.xlsx``

Once human review fills in the residuals xlsx, the resulting
``-biosphere-manual-matches.json`` is promoted by hand into
``source/randonneur_packages/`` and the registry picks it up on the next
build (REFACTOR.md §0.2).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from config import Settings
from core.logging import Logging
from readers import RandonneurDataLoader
from transforms import SimaProImporter


@dataclass(frozen=True)
class RandonneurPackagesExporter:
    settings: Settings

    CC_BY_LICENSE: ClassVar[dict] = {
        "name": "CC-BY-4.0",
        "path": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "title": "Creative Commons Attribution 4.0 International",
    }

    @property
    def _log(self):
        return Logging.get(__name__)

    def export(self) -> dict[str, str]:
        s = self.settings
        out_dir = s.paths.randonneur_packages
        out_dir.mkdir(parents=True, exist_ok=True)

        sp = SimaProImporter(s).load()
        loader = RandonneurDataLoader()

        out: dict[str, str] = {}
        out["delete_aggregated"] = str(self._write_delete_aggregated(sp.data, out_dir, loader))
        out["restore_names"] = str(self._write_restore_names(sp.data, out_dir, loader))
        out["residuals_review"] = str(self._write_residuals_review(sp.data, loader))
        return out

    # ------------------------------------------------------------------

    def _header(
        self,
        name: str,
        description: str,
        graph_context: str,
        source_id: str,
        target_id: str,
    ) -> dict:
        return {
            "name": name,
            "description": description,
            "contributors": [{"title": "Laurenz Bougan", "role": "author"}],
            "created": dt.datetime.now(tz=dt.UTC).isoformat(),
            "version": "1.0.0",
            "licenses": [self.CC_BY_LICENSE],
            "graph_context": [graph_context],
            "mapping": {
                "source": {"expression language": "like JSONPath", "labels": {}},
                "target": {"expression language": "like JSONPath", "labels": {}},
            },
            "source_id": source_id,
            "target_id": target_id,
            "homepage": "https://doc.agribalyse.fr/documentation-en/agribalyse-data/data-access",
        }

    def _write_delete_aggregated(self, data, out_dir, loader: RandonneurDataLoader):
        entries = []
        for proc in data:
            name = proc.get("name", "")
            category = str(proc.get("simapro_category", ""))
            identifier = proc.get("simapro_identifier", proc.get("code", ""))
            if "Copied from Ecoinvent" in category or "Copied from Ecoinvent" in name:
                entry = {"source": {"name": name}}
                if identifier:
                    entry["source"]["identifier"] = identifier
                entries.append(entry)
        entries.sort(key=lambda e: e["source"]["name"])

        pkg = self._header(
            name="agribalyse-3.2-delete-aggregated-ecoinvent",
            description=(
                "Delete ecoinvent activities from Agribalyse 3.2 given as aggregated processes. "
                "You need to use the real ecoinvent database to resolve these exchanges."
            ),
            graph_context="nodes",
            source_id="agribalyse-3.2",
            target_id="agribalyse-3.2",
        )
        pkg["delete"] = entries

        path = out_dir / "agribalyse-3.2-delete-aggregated-ecoinvent.json"
        path.write_text(json.dumps(pkg, indent=2, ensure_ascii=False))
        return path

    def _write_restore_names(self, data, out_dir, loader: RandonneurDataLoader):
        SUFFIX = ", S - Copied from Ecoinvent U"
        entries = []
        seen: set[str] = set()
        for proc in data:
            for exc in proc.get("exchanges", []):
                exc_name = exc.get("name", "")
                if exc_name.endswith(SUFFIX) and exc_name not in seen:
                    clean = exc_name[: -len(SUFFIX)]
                    entries.append({"source": {"name": exc_name}, "target": {"name": clean}})
                    seen.add(exc_name)
        entries.sort(key=lambda e: e["source"]["name"])

        pkg = self._header(
            name="agribalyse-3.2-restore-simapro-ecoinvent-names",
            description=(
                "Restore names of linked ecoinvent processes to original SimaPro form. "
                "Changes 'Foo, S - Copied from Ecoinvent U' to 'Foo'."
            ),
            graph_context="edges",
            source_id="agribalyse-3.2",
            target_id="agribalyse-3.2",
        )
        pkg["replace"] = entries

        path = out_dir / "agribalyse-3.2-restore-simapro-ecoinvent-names.json"
        path.write_text(json.dumps(pkg, indent=2, ensure_ascii=False))
        return path

    def _write_residuals_review(self, data, loader: RandonneurDataLoader):
        unlinked: dict[str, dict] = {}
        for proc in data:
            for exc in proc.get("exchanges", []):
                if exc.get("type") != "biosphere" or exc.get("input"):
                    continue
                name = exc.get("name", "")
                if name and name not in unlinked:
                    unlinked[name] = {
                        "name": name,
                        "unit": exc.get("unit", ""),
                        "categories": exc.get("categories", ()),
                    }

        # Reuse the AGB 3.1.1 manual-matches as a pre-fill.
        prior = (
            loader.load("agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches")
            if loader.has("agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches")
            else {"replace": []}
        )
        prior_index = {e["source"]["name"]: e for e in prior.get("replace", [])}

        rows = []
        for name, info in sorted(unlinked.items()):
            cats = info.get("categories", ())
            existing = prior_index.get(name)
            target_name = existing.get("target", {}).get("name", "") if existing else ""
            rows.append(
                {
                    "source_name": name,
                    "source_cas": "",
                    "source_top_cat": cats[0] if len(cats) >= 1 else "",
                    "source_sub_cat": cats[1] if len(cats) >= 2 else "",
                    "source_unit": info.get("unit", ""),
                    "target_name": target_name,
                    "target_cas": "",
                    "target_top_cat": "",
                    "target_sub_cat": "",
                    "target_unit": "",
                    "rule": "manual",
                    "confidence": 1.0,
                    "decision": "accept" if existing and target_name else "",
                    "multiplier": 1.0,
                    "reviewer_note": "carried from 3.1.1" if existing else "",
                }
            )

        path = self.settings.paths.to_review / "agribalyse-3.2-biosphere-residuals-for-review.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_excel(path, index=False)
        return path
