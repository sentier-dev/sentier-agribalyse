"""``UnitCopyRebuilder`` — rebuild the stripped ecoinvent unit processes.

ADEME's export embeds two kinds of ecoinvent copies. The aggregated ones
(``Cut-off, S - Copied from Ecoinvent``) are closed inventories: a
consumer can be pointed at the user's own ecoinvent activity instead.
The 125 **unit** copies (``Cut-off, U - Copied from Ecoinvent``, mostly
markets and crop processes) are different: inside the Agribalyse library
their inputs resolve to ADEME's ``Adapted from Ecoinvent`` datasets where
one exists (``market for soybean {GLO}`` draws ADEME's corrected Brazilian
soybean, not ecoinvent's). They are how the adaptations reach the rest of
the database. Pointing a consumer at ecoinvent's own market bypasses
every adaptation behind it.

The ecoinvent-free CSV does not ship those blocks. This transform
rebuilds each of them from the user's ecoinvent snapshot, as an
Agribalyse-side activity:

* production and elementary exchanges: as in ecoinvent;
* an input whose ecoinvent activity has an ``Adapted from Ecoinvent``
  counterpart in the CSV (same product, location and activity): the
  adapted dataset;
* an input that is itself a stripped unit copy: its rebuilt twin;
* an input whose SimaPro name is shared with another activity, when the
  library holds that other activity's block under the name
  (``SimaproNameCollisions``): the activity SimaPro actually used;
* any other input: the ecoinvent activity.

Two stripped blocks are not copies of a whole dataset but ADEME's
*combustion extracts* (``Combustion diesel (from data: <ecoinvent
transport activity> - Copied from Ecoinvent) U``): the air emissions of
that activity, per kilogram of the fuel it burns. They are rebuilt the
same way: the activity's elementary exchanges to air, with its largest
kilogram input as the reference amount.

Only the *names* of the stripped unit processes ship
(``source/agribalyse-3.2-copied-unit-processes.json``); every amount comes
from the user's licensed snapshot. ``TechnosphereMatcher`` sends
references to a stripped unit copy to its rebuilt twin through
:pyattr:`UnitCopyRebuild.redirects`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from config import Settings
from core.logging import Logging
from matching.ecoinvent_catalog import EcoinventActivityRef, EcoinventCatalog
from matching.simapro_form_resolver import SimaproFormResolver
from matching.simapro_name_collisions import SimaproNameCollisions
from registry.indexes import UnitConverter
from scoring.ecoinvent_exchanges import EcoinventExchangesIngester

Key = tuple[str, str]


@dataclass(frozen=True)
class UnitCopyRebuild:
    """Outcome: where references to a stripped unit copy must point."""

    redirects: dict[Key, Key] = field(default_factory=dict)
    by_name: dict[str, Key] = field(default_factory=dict)
    """SimaPro name → rebuilt product, for the blocks that are not a whole
    ecoinvent activity (combustion extracts)."""
    stats: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class UnitCopyRebuilder:
    settings: Settings
    catalog: EcoinventCatalog
    unit_converter: UnitConverter

    ADAPTED_MARKER: ClassVar[str] = "adapted from ecoinvent"
    PROCESS_PREFIX: ClassVar[str] = "ecoinvent-unit-copy-"
    PRODUCT_PREFIX: ClassVar[str] = "ecoinvent-unit-copy-product-"
    EXTRACT_PREFIX: ClassVar[str] = "ecoinvent-combustion-extract-"
    EXTRACT_PRODUCT_PREFIX: ClassVar[str] = "ecoinvent-combustion-extract-product-"
    _NAME: ClassVar[re.Pattern[str]] = re.compile(r"^(.+?)\s*\{([^}]+)\}\s*\|")
    _EXTRACT: ClassVar[re.Pattern[str]] = re.compile(
        r"^Combustion\b[^(]*\(\s*from data\s*:\s*(.+)\)\s*U\s*$", re.IGNORECASE
    )
    MASS_UNIT: ClassVar[str] = "kilogram"
    AIR: ClassVar[str] = "air"

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ── public API ───────────────────────────────────────────────────────

    def apply(self, sp) -> UnitCopyRebuild:
        """Append the rebuilt unit processes to ``sp.data``."""
        names = self._unit_copy_names()
        exchanges_path = self.settings.paths.ecoinvent_exchanges_parquet
        if not names or not exchanges_path.exists():
            return UnitCopyRebuild(stats={"unit_copies_listed": len(names), "rebuilt": 0})

        agb_db = self.settings.resolved_agribalyse_db_name
        ei_db = self.settings.ecoinvent_db_name
        resolver = SimaproFormResolver(catalog=self.catalog, ei_db=ei_db)
        collisions = SimaproNameCollisions(
            resolver=resolver, edge_labels_path=self.settings.paths.edge_label_corrections_json
        )
        activities = {a.code: a for a in self.catalog.activities if a.db == ei_db}

        copies: dict[str, tuple[str, EcoinventActivityRef]] = {}
        extracts: dict[str, EcoinventActivityRef] = {}
        for name in names:
            inner = self._EXTRACT.match(name)
            if inner is not None:
                source = resolver.resolve(inner.group(1))
                if source is not None:
                    extracts[name] = source
                continue
            hit = resolver.resolve(name)
            if hit is not None and hit.code not in copies:
                copies[hit.code] = (name, hit)
        adapted, n_adapted_ambiguous = self._adapted_products(sp.data, resolver, agb_db)

        frame = EcoinventExchangesIngester._normalise_waste_signs(pd.read_parquet(exchanges_path))
        wanted = set(copies) | {ref.code for ref in extracts.values()}
        frame = frame[(frame["output_database"] == ei_db) & frame["output_code"].isin(wanted)]
        by_code = {code: rows for code, rows in frame.groupby("output_code", sort=False)}
        units = {a.code: a.unit for a in self.catalog.activities if a.db == ei_db}

        datasets: list[dict] = []
        redirects: dict[Key, Key] = {}
        counts = {
            "to_adapted": 0,
            "to_rebuilt": 0,
            "to_shared_name_block": 0,
            "to_ecoinvent": 0,
            "unit_mismatch": 0,
        }
        stand_ins: dict[str, str] = {}
        for code in frame.loc[frame["type"] != "biosphere", "input_code"].unique():
            activity = activities.get(str(code))
            other = collisions.stand_in(activity) if activity is not None else None
            if other is not None:
                stand_ins[str(code)] = other.code
        for code in sorted(copies):
            rows = by_code.get(code)
            if rows is None:
                continue
            name, ref = copies[code]
            process, product = self._datasets(
                name, ref, rows, agb_db, ei_db, copies, adapted, units, counts, stand_ins
            )
            datasets += [process, product]
            redirects[(ei_db, code)] = (agb_db, product["code"])
        by_name: dict[str, Key] = {}
        air = self._air_flows()
        for name in sorted(extracts):
            built = self._extract_datasets(name, extracts[name], by_code, agb_db, ei_db, units, air)
            if built is not None:
                datasets += list(built)
                by_name[name] = (agb_db, built[1]["code"])
        sp.data = [*sp.data, *datasets]

        stats = {
            "unit_copies_listed": len(names),
            "unit_copies_resolved": len(copies) + len(extracts),
            "rebuilt": len(redirects),
            "combustion_extracts_rebuilt": len(by_name),
            "adapted_counterparts": len(adapted),
            "adapted_ambiguous": n_adapted_ambiguous,
            "inputs_to_adapted": counts["to_adapted"],
            "inputs_to_rebuilt": counts["to_rebuilt"],
            "inputs_to_shared_name_block": counts["to_shared_name_block"],
            "inputs_to_ecoinvent": counts["to_ecoinvent"],
            "inputs_unit_mismatch": counts["unit_mismatch"],
        }
        self._log.info("transforms.unit_copies.rebuilt", **stats)
        return UnitCopyRebuild(redirects=redirects, by_name=by_name, stats=stats)

    def prune_unused(self, sp, rebuild: UnitCopyRebuild) -> int:
        """Drop rebuilt processes no linked exchange reaches. Returns the count dropped."""
        agb_db = self.settings.resolved_agribalyse_db_name
        products = set(rebuild.redirects.values()) | set(rebuild.by_name.values())
        if not products:
            return 0
        process_of = {
            (agb_db, str(ds["code"])): tuple(ds["reference process"])
            for ds in sp.data
            if ds.get("type") == "product" and (agb_db, str(ds.get("code"))) in products
        }
        inputs_of: dict[Key, set[Key]] = {}
        reached: set[Key] = set()
        for ds in sp.data:
            key = (ds.get("database") or "", ds.get("code") or "")
            is_rebuilt = key in process_of.values()
            for exc in ds.get("exchanges", ()) or ():
                target = tuple(exc["input"]) if exc.get("input") else None
                if target not in products or exc.get("type") == "production":
                    continue
                if is_rebuilt:
                    inputs_of.setdefault(key, set()).add(target)
                else:
                    reached.add(target)
        frontier = list(reached)
        while frontier:
            for nxt in inputs_of.get(process_of[frontier.pop()], ()):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append(nxt)
        unused = products - reached
        if not unused:
            return 0
        drop = unused | {process_of[p] for p in unused}
        sp.data = [
            ds for ds in sp.data if (ds.get("database") or "", ds.get("code") or "") not in drop
        ]
        self._log.info("transforms.unit_copies.pruned", dropped=len(unused), kept=len(reached))
        return len(unused)

    # ── internals ────────────────────────────────────────────────────────

    def _unit_copy_names(self) -> list[str]:
        path = self.settings.paths.copied_unit_processes_json
        if not path.exists():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")).get("names", []))

    def _adapted_products(
        self, sp_data: list[dict], resolver: SimaproFormResolver, agb_db: str
    ) -> tuple[dict[str, tuple[Key, str]], int]:
        """ecoinvent code → ``(product key, unit)`` of ADEME's adapted dataset.

        The adapted dataset must be the same product at the same location
        as the ecoinvent activity it resolves to: a proxy (``Mustard,
        proxy from Rape seed``) or a regional variant filed under another
        location is not a counterpart. Two candidates for one activity:
        neither is used.
        """
        found: dict[str, list[tuple[Key, str]]] = {}
        for ds in sp_data:
            if ds.get("database") != agb_db or ds.get("type") == "product":
                continue
            label = str(ds.get("reference product") or "")
            if self.ADAPTED_MARKER not in label.lower():
                continue
            parsed = self._NAME.match(label)
            hit = resolver.resolve(label)
            if parsed is None or hit is None:
                continue
            display, location = parsed.group(1).strip().lower(), parsed.group(2).strip()
            if hit.location != location or hit.reference_product.lower() != display:
                continue
            product = self._production_input(ds)
            if product is not None:
                found.setdefault(hit.code, []).append((product, str(ds.get("unit") or "")))
        unique = {code: rows[0] for code, rows in found.items() if len(rows) == 1}
        return unique, len(found) - len(unique)

    def _datasets(
        self,
        name: str,
        ref: EcoinventActivityRef,
        rows: pd.DataFrame,
        agb_db: str,
        ei_db: str,
        copies: dict[str, tuple[str, EcoinventActivityRef]],
        adapted: dict[str, tuple[Key, str]],
        units: dict[str, str],
        counts: dict[str, int],
        stand_ins: dict[str, str],
    ) -> tuple[dict, dict]:
        process_key = (agb_db, self.PROCESS_PREFIX + ref.code)
        product_key = (agb_db, self.PRODUCT_PREFIX + ref.code)
        exchanges: list[dict] = []
        production = 1.0
        for row in rows.itertuples(index=False):
            amount = float(row.amount)
            if row.type == "production":
                production = amount
                continue
            if row.type == "biosphere":
                exchanges.append(
                    {
                        "type": "biosphere",
                        "amount": amount,
                        "input": (str(row.input_database), str(row.input_code)),
                    }
                )
                continue
            target, amount = self._input_target(
                str(row.input_database),
                str(row.input_code),
                amount,
                agb_db,
                ei_db,
                copies,
                adapted,
                units,
                counts,
                stand_ins,
            )
            exchanges.append({"type": str(row.type), "amount": amount, "input": target})
        exchanges.insert(
            0,
            {
                "type": "production",
                "functional": True,
                "name": name,
                "unit": ref.unit,
                "amount": production,
                "input": product_key,
            },
        )
        common = {
            "database": agb_db,
            "name": name,
            "unit": ref.unit,
            "location": ref.location,
            "tags": {"rebuilt_from": f"{ei_db}:{ref.code}"},
        }
        process = {
            **common,
            "code": process_key[1],
            "type": "process",
            "reference product": name,
            "production amount": production,
            "exchanges": exchanges,
        }
        product = {**common, "code": product_key[1], "type": "product", "exchanges": []}
        product["reference process"] = process_key
        return process, product

    def _input_target(
        self,
        input_db: str,
        input_code: str,
        amount: float,
        agb_db: str,
        ei_db: str,
        copies: dict[str, tuple[str, EcoinventActivityRef]],
        adapted: dict[str, tuple[Key, str]],
        units: dict[str, str],
        counts: dict[str, int],
        stand_ins: dict[str, str],
    ) -> tuple[Key, float]:
        if input_db != ei_db:
            return (input_db, input_code), amount
        if input_code in adapted:
            product, adapted_unit = adapted[input_code]
            multiplier = self.unit_converter.multiplier(units.get(input_code, ""), adapted_unit)
            if multiplier is not None:
                counts["to_adapted"] += 1
                return product, amount * multiplier
            counts["unit_mismatch"] += 1
        if input_code in copies:
            counts["to_rebuilt"] += 1
            return (agb_db, self.PRODUCT_PREFIX + input_code), amount
        if input_code in stand_ins:
            counts["to_shared_name_block"] += 1
            return (input_db, stand_ins[input_code]), amount
        counts["to_ecoinvent"] += 1
        return (input_db, input_code), amount

    def _air_flows(self) -> set[Key]:
        """``(database, code)`` of every elementary flow whose top compartment is air."""
        path = self.settings.paths.registry_biosphere_catalog
        if not path.exists():
            return set()
        catalog = pd.read_parquet(path, columns=["database", "code", "categories"])
        return {
            (str(row.database), str(row.code))
            for row in catalog.itertuples(index=False)
            if len(row.categories) > 0 and str(row.categories[0]).lower() == self.AIR
        }

    def _extract_datasets(
        self,
        name: str,
        ref: EcoinventActivityRef,
        by_code: dict[str, pd.DataFrame],
        agb_db: str,
        ei_db: str,
        units: dict[str, str],
        air: set[Key],
    ) -> tuple[dict, dict] | None:
        """Air emissions of ``ref`` per kilogram of the fuel it burns."""
        rows = by_code.get(ref.code)
        if rows is None:
            return None
        fuel = [
            float(row.amount)
            for row in rows.itertuples(index=False)
            if row.type == "technosphere"
            and row.input_database == ei_db
            and units.get(str(row.input_code), "").lower() == self.MASS_UNIT
            and float(row.amount) > 0
        ]
        emissions = [
            {
                "type": "biosphere",
                "amount": float(row.amount),
                "input": (str(row.input_database), str(row.input_code)),
            }
            for row in rows.itertuples(index=False)
            if row.type == "biosphere" and (str(row.input_database), str(row.input_code)) in air
        ]
        if not fuel or not emissions:
            return None
        process_key = (agb_db, self.EXTRACT_PREFIX + ref.code)
        product_key = (agb_db, self.EXTRACT_PRODUCT_PREFIX + ref.code)
        production = {
            "type": "production",
            "functional": True,
            "name": name,
            "unit": self.MASS_UNIT,
            "amount": max(fuel),
            "input": product_key,
        }
        common = {
            "database": agb_db,
            "name": name,
            "unit": self.MASS_UNIT,
            "location": ref.location,
            "tags": {"rebuilt_from": f"{ei_db}:{ref.code}"},
        }
        process = {
            **common,
            "code": process_key[1],
            "type": "process",
            "reference product": name,
            "production amount": max(fuel),
            "exchanges": [production, *emissions],
        }
        product = {**common, "code": product_key[1], "type": "product", "exchanges": []}
        product["reference process"] = process_key
        return process, product

    @staticmethod
    def _production_input(ds: dict) -> Key | None:
        inputs = [
            tuple(exc["input"])
            for exc in ds.get("exchanges", ()) or ()
            if exc.get("type") == "production" and exc.get("input")
        ]
        return inputs[0] if len(inputs) == 1 else None
