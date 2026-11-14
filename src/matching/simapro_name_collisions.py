"""``SimaproNameCollisions`` — which block SimaPro really used behind a shared name.

SimaPro cuts some ecoinvent activity names down to a generic label, so
several activities making the same product at the same place end up under
one SimaPro name. ``Pesticide, unspecified {GLO}| market for`` stands for
ecoinvent's real market *and* for every ``… to generic market for
pesticide, unspecified``. The library holds a single block under that
name, and every reference to the name gets that block, whatever activity
ecoinvent meant. For this product it is ``dichloropropene to generic
market for pesticide, unspecified``, 46 times lower on ozone depletion
than the real market.

The edge-label package records, for each embedded copy, the ecoinvent
activity its block is. This class turns that into the question the
unit-copy rebuild needs answered: *ecoinvent says this process consumes
activity Y; which activity did the SimaPro library give it?*

* Y has a block of its own: Y.
* Y has none, and exactly one block makes the same product at the same
  location under a label Y's name starts with: that block's activity.
* otherwise: no opinion (``None``); the caller keeps Y.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

from matching.ecoinvent_catalog import EcoinventActivityRef
from matching.simapro_form_resolver import SimaproFormResolver


@dataclass(frozen=True)
class SimaproNameCollisions:
    resolver: SimaproFormResolver
    edge_labels_path: Path

    _NAME: ClassVar[re.Pattern[str]] = re.compile(r"^(.+?)\s*\{([^}]+)\}\s*\|\s*(.*?)\s*\|\s*")

    def stand_in(self, activity: EcoinventActivityRef) -> EcoinventActivityRef | None:
        """The activity whose block SimaPro used for ``activity``, if it differs."""
        if activity.code in self._with_own_block:
            return None
        name = activity.name.lower()
        blocks = {
            target.code: target
            for label, target in self._blocks.get(
                (activity.reference_product.lower(), activity.location), ()
            )
            if label
            and name.startswith(label)
            and target.reference_product.lower() == activity.reference_product.lower()
        }
        if len(blocks) != 1:
            return None
        return next(iter(blocks.values()))

    @cached_property
    def _entries(self) -> list[tuple[str, str, str, EcoinventActivityRef]]:
        """``(product_lower, location, label_lower, block activity)`` per embedded copy."""
        if not self.edge_labels_path.exists():
            return []
        data = json.loads(self.edge_labels_path.read_text(encoding="utf-8"))
        out: list[tuple[str, str, str, EcoinventActivityRef]] = []
        for entry in data.get("replace", ()) or ():
            source = str((entry.get("source") or {}).get("name") or "")
            target = self.resolver.resolve(str((entry.get("target") or {}).get("name") or ""))
            parsed = self._NAME.match(source)
            if target is None or parsed is None:
                continue
            display, location, label = (g.strip() for g in parsed.groups())
            out.append((display.lower(), location, label.lower(), target))
        return out

    @cached_property
    def _with_own_block(self) -> frozenset[str]:
        return frozenset(target.code for _d, _l, _a, target in self._entries)

    @cached_property
    def _blocks(self) -> dict[tuple[str, str], tuple[tuple[str, EcoinventActivityRef], ...]]:
        index: dict[tuple[str, str], list[tuple[str, EcoinventActivityRef]]] = defaultdict(list)
        for display, location, label, target in self._entries:
            index[(display, location)].append((label, target))
        return {key: tuple(rows) for key, rows in index.items()}
