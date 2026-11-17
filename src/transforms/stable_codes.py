"""``StableCodeAssigner`` — content-derived codes for what the SimaPro reader draws at random.

``bw_simapro_csv`` (and ``multifunctional``, which it calls for the
allocation of multi-output processes) give four things a ``uuid4`` value:

* the code of every product node (``separate_products=True``);
* the code of every process whose block carries no ``Process identifier``;
* the code of every ``readonly_process`` the allocation step adds, one per
  product of a multi-output process;
* the ``mf_allocation_run_uuid`` stamped on a multi-output process and its
  allocated children.

Those values changed on every parse. Every id of the scoring package is a
hash of ``(database, code)``, so two builds of the same inputs disagreed on
the product ids, on the index order of the matrices and on the content
hash. They also disagreed on content: ``ProductDeduplicator`` keeps the
producer with the smallest id, so which of two columns stood for an
allocated product (the reader's read-only child or the ``Allocator``'s
split) was drawn anew on each build. The two hold the same exchanges, but
only the child has a location, so the regional correction of an allocated
product followed the draw. With stable codes every build makes the same
choice; the choice itself is still made by id order (see
``ProductDeduplicator``).

This step runs right after the parse and replaces each of those values by
a digest of what the node is:

=============================  ================================================
node                           derived from
=============================  ================================================
process without identifier     its name, and the names and units of its products
product                        the code of its process, and its own name
allocated (read-only) process  the code of its parent, and the name of its product
allocation run                 the code of the multi-output process
=============================  ================================================

No derivation looks at where a block sits in the file, so a process gets
the same codes from ADEME's full export and from the ecoinvent-free one.
A process that carries a ``Process identifier`` keeps it as its code.

Two nodes of one kind that share the whole derivation (SimaPro does not
allow it for product names, a hand-edited file might) are told apart by
their order of appearance, counted among those nodes only.

Every reference follows its node: the ``input`` and ``output`` of
exchanges, ``reference process``, ``mf_parent_key`` and
``mf_allocated_process_code``. Parameter rows are extracted afterwards, from
the rewritten datasets.

The derived codes keep the shape of the ones they replace (32 lower-case
hexadecimal characters), so nothing downstream sees a new kind of code.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, ClassVar

from core.logging import Logging

Key = tuple[str, str]


@dataclass(frozen=True)
class StableCodeAssigner:
    """Rewrite the random codes of a ``lci_to_brightway`` result in place."""

    # Bump when a derivation changes: a parse cached under another scheme is
    # then re-done instead of mixing its codes with the new ones.
    SCHEME: ClassVar[int] = 1
    CODE_LENGTH: ClassVar[int] = 32

    _NODE_REFERENCES: ClassVar[tuple[str, ...]] = ("reference process", "mf_parent_key")
    _EDGE_REFERENCES: ClassVar[tuple[str, ...]] = ("input", "output")
    _ALLOCATED_CODE: ClassVar[str] = "mf_allocated_process_code"
    _ALLOCATION_RUN: ClassVar[str] = "mf_allocation_run_uuid"
    _EMPTY_IDENTIFIERS: ClassVar[frozenset[str]] = frozenset({"", '""', "''"})

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    def apply(self, spcsv: Any, bw_data: dict) -> dict[str, int]:
        """Assign the codes; returns how many nodes of each kind got one."""
        processes = list(bw_data.get("processes", []))
        products = list(bw_data.get("products", []))
        stats = {"processes": 0, "products": 0, "allocated_processes": 0, "told_apart_by_order": 0}
        raw_blocks = getattr(spcsv, "blocks", None)
        if raw_blocks is None:
            # Stub parsers of the unit suite expose no blocks and no codes.
            return stats
        blocks = [b for b in raw_blocks if self._is_process(b)]
        if not blocks:
            if processes:
                # The reader parsed processes, yet no block is recognised as
                # one: ``bw_simapro_csv`` renamed or reshaped its ``Process``
                # block. Returning here would leave the drawn codes in place
                # under a parse that claims the stable scheme.
                raise ValueError(
                    f"StableCodeAssigner: the reader returned {len(processes)} processes "
                    "but none of its blocks is recognised as a SimaPro Process block "
                    "(class name 'Process' with dict 'parsed' and 'blocks'). The installed "
                    "bw_simapro_csv may differ from the locked version; the codes would "
                    "stay random."
                )
            return stats

        parents = [ds for ds in processes if "mf_parent_key" not in ds]
        children = [ds for ds in processes if "mf_parent_key" in ds]
        seen: Counter[tuple[str, ...]] = Counter()
        codes: dict[Key, str] = {}

        for ds in self._without_identifier(blocks, parents):
            products_of = [
                part
                for exc in ds.get("exchanges", ())
                if exc.get("functional")
                for part in (str(exc.get("name") or ""), str(exc.get("unit") or ""))
            ]
            codes[self._key(ds)] = self._derive(
                seen, "process", str(ds.get("name") or ""), *products_of
            )
        stats["processes"] = len(codes)

        for ds in products:
            process = self._resolved(codes, ds["reference process"])
            codes[self._key(ds)] = self._derive(
                seen, "product", process[1], str(ds.get("name") or "")
            )
        stats["products"] = len(products)

        for ds in children:
            parent = self._resolved(codes, ds["mf_parent_key"])
            product = next((exc for exc in ds.get("exchanges", ()) if exc.get("functional")), {})
            codes[self._key(ds)] = self._derive(
                seen, "allocated process", parent[1], str(product.get("name") or "")
            )
        stats["allocated_processes"] = len(children)
        stats["told_apart_by_order"] = sum(n - 1 for n in seen.values())

        for ds in (*processes, *products):
            self._rewrite(ds, codes)
        self._check_unique((*processes, *products))
        self._log.info("csv.codes.assigned", scheme=self.SCHEME, **stats)
        return stats

    # ── which processes carry a random code ──────────────────────────────

    @staticmethod
    def _is_process(block: Any) -> bool:
        # Duck-typed like ``ProcessParameterExtractor._is_process``: the unit
        # suite stubs ``bw_simapro_csv``, an isinstance check would bind the stub.
        return (
            type(block).__name__ == "Process"
            and isinstance(getattr(block, "parsed", None), dict)
            and isinstance(getattr(block, "blocks", None), dict)
        )

    @classmethod
    def _identifier(cls, block: Any) -> str:
        """The block's ``Process identifier``, or ``""`` where the reader draws a code."""
        raw = str(block.parsed.get("metadata", {}).get("Process identifier") or "").strip()
        return "" if raw in cls._EMPTY_IDENTIFIERS else raw

    @classmethod
    def _without_identifier(cls, blocks: list[Any], parents: list[dict]) -> list[dict]:
        """The parent datasets whose code is not an identifier of the file.

        The reader uses the identifier as the code, so a code that no block
        names was drawn. This needs no pairing of blocks and datasets by
        position; the counts are compared so a surprise raises.
        """
        identifiers = {cls._identifier(b) for b in blocks} - {""}
        expected = sum(1 for b in blocks if not cls._identifier(b))
        drawn = [ds for ds in parents if str(ds.get("code") or "").strip() not in identifiers]
        if len(drawn) != expected:
            raise ValueError(
                f"{expected} process block(s) carry no identifier, but {len(drawn)} dataset "
                f"code(s) are not an identifier of the file: cannot tell which codes the "
                f"reader drew."
            )
        return drawn

    # ── derivation ───────────────────────────────────────────────────────

    @classmethod
    def _derive(cls, seen: Counter[tuple[str, ...]], kind: str, *content: str) -> str:
        """The code of the next node of ``kind`` with this ``content``."""
        identity = (kind, *content)
        seen[identity] += 1
        occurrence = () if seen[identity] == 1 else (str(seen[identity]),)
        return cls._digest(*identity, *occurrence)

    @classmethod
    def _digest(cls, *parts: str) -> str:
        # JSON keeps the parts apart whatever characters a name holds.
        payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[: cls.CODE_LENGTH]

    # ── rewriting ────────────────────────────────────────────────────────

    @staticmethod
    def _key(ds: dict) -> Key:
        return (str(ds.get("database") or ""), str(ds.get("code") or ""))

    @staticmethod
    def _resolved(codes: dict[Key, str], reference: Any) -> Key:
        """``reference`` with its code replaced when that code was drawn."""
        key = (str(reference[0]), str(reference[1]))
        return (key[0], codes.get(key, key[1]))

    def _rewrite(self, ds: dict, codes: dict[Key, str]) -> None:
        own = self._key(ds)
        if own in codes:
            ds["code"] = codes[own]
        for label in self._NODE_REFERENCES:
            if ds.get(label) is not None:
                ds[label] = self._resolved(codes, ds[label])
        if self._ALLOCATION_RUN in ds:
            multi_output = ds.get("mf_parent_key") or (own[0], ds["code"])
            ds[self._ALLOCATION_RUN] = self._digest("allocation run", str(multi_output[1]))
        for exc in ds.get("exchanges", ()):
            for label in self._EDGE_REFERENCES:
                if exc.get(label) is not None:
                    exc[label] = self._resolved(codes, exc[label])
            if exc.get(self._ALLOCATED_CODE) is not None:
                exc[self._ALLOCATED_CODE] = self._resolved(
                    codes, (own[0], exc[self._ALLOCATED_CODE])
                )[1]

    @classmethod
    def _check_unique(cls, datasets: tuple[dict, ...]) -> None:
        counts = Counter(cls._key(ds) for ds in datasets)
        repeated = sorted(key for key, n in counts.items() if n > 1)
        if repeated:
            raise ValueError(
                f"{len(repeated)} code(s) are held by more than one dataset: {repeated[:5]}. "
                "Either two blocks of the file carry the same Process identifier, or a "
                "derived code equals an identifier of the file."
            )
