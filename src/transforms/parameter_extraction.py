"""``ProcessParameterExtractor`` — lift process-local parameter definitions.

``bw_simapro_csv``'s ``lci_to_brightway`` exports only the database/project
parameter blocks — which are **empty** in the AGB 3.2 export. Every real
parameter is process-local (13 725 processes with input parameters, 1 422
with calculated parameters), living on ``Process.blocks["Input parameters"]``
/ ``["Calculated parameters"]`` and dropped from the brightway dicts.

This extractor walks ``SimaProCSV.blocks`` after resolution and captures the
definitions so they survive into ``ParsedSimaProCsv`` (and its pickle cache).
Names come in two forms:

* ``name`` — the normalized form referenced by exchange formulas
  (``ratio_ingredient1`` → ``SP_RATIO_INGREDIENT1``);
* ``original_name`` — the SimaPro-facing name users know
  (``Packaging_Weight``); matching is case-insensitive on this form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.logging import Logging

_BLOCK_KINDS: tuple[tuple[str, str], ...] = (
    ("input", "Input parameters"),
    ("calculated", "Calculated parameters"),
)


@dataclass(frozen=True)
class ProcessParameterExtractor:
    """Extract per-process parameter definitions from a parsed ``SimaProCSV``."""

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    @staticmethod
    def _is_process(block: Any) -> bool:
        # Duck-typed on purpose: ``bw_simapro_csv`` is stubbed in the unit
        # suite (see ``tests/fixtures/bw_stubs.py``), so importing
        # ``bw_simapro_csv.blocks.Process`` for an isinstance check would
        # bind the stub. The class name + shape is unambiguous.
        return (
            type(block).__name__ == "Process"
            and isinstance(getattr(block, "parsed", None), dict)
            and isinstance(getattr(block, "blocks", None), dict)
        )

    def extract(self, spcsv: Any, bw_processes: list[dict] | None = None) -> list[dict]:
        """Extract parameter rows, keyed by the emitted datasets' codes.

        Two joins happen here, both against ``bw_processes`` (the
        ``lci_to_brightway`` output):

        * **Parents.** ~700 AGB 3.2 processes have no ``Process
          identifier``; the export synthesises uuid4 codes that exist only
          on the output dicts. The export emits exactly one PARENT dataset
          per ``Process`` block, in block order (the multifunctional
          allocation step appends extra ``readonly_process`` children but
          never drops or reorders parents), so parent codes are joined by
          position — cross-checked on every identifier-carrying pair, and
          raising on any disagreement rather than mis-keying silently.
        * **Children.** Multifunctional parents get per-product
          ``readonly_process`` children (uuid4 codes, ``mf_parent_key``
          linkage) whose exchange formulas embed the allocation factor
          textually — evaluating them with the PARENT's parameter
          environment reproduces their baked amounts. Each parent's rows
          are mirrored onto its children so overrides reach the allocated
          datasets too (without this, a stale child column can survive
          producer dedup and silently drop an override).

        Without ``bw_processes`` (legacy callers/tests) it falls back to
        the metadata identifier and skips identifier-less blocks.
        """
        blocks = [b for b in (getattr(spcsv, "blocks", None) or []) if self._is_process(b)]
        codes: list[str | None]
        children_by_parent: dict[str, list[str]] = {}
        if bw_processes is not None:
            if not blocks:
                # Stub/fake parsers (unit suite) expose no blocks — there is
                # nothing to extract, and no join to get wrong.
                self._log.info("csv.parameters.extracted", n_rows=0)
                return []
            parents = [ds for ds in bw_processes if "mf_parent_key" not in ds]
            for ds in bw_processes:
                parent_key = ds.get("mf_parent_key")
                if parent_key is not None:
                    children_by_parent.setdefault(str(parent_key[1]), []).append(
                        str(ds.get("code") or "")
                    )
            if len(parents) != len(blocks):
                raise ValueError(
                    f"Process block/dataset count mismatch: {len(blocks)} blocks "
                    f"vs {len(parents)} parent datasets — cannot join parameters "
                    f"by position."
                )
            codes = []
            uuid_hex = re.compile(r"^[0-9a-f]{32}$")
            for block, ds in zip(blocks, parents, strict=True):
                meta_code = (
                    block.parsed.get("metadata", {}).get("Process identifier") or ""
                ).strip()
                ds_code = str(ds.get("code") or "")
                if meta_code and meta_code not in {'""', "''"}:
                    if meta_code != ds_code.strip():
                        raise ValueError(
                            f"Block/dataset order drift: block identifier "
                            f"{meta_code!r} vs dataset code {ds_code!r} — "
                            f"refusing to join parameters by position."
                        )
                elif ds_code and not uuid_hex.match(ds_code):
                    # An identifier-less block must pair with a SYNTHESISED
                    # (uuid4-hex) dataset code; a real identifier here means
                    # the positional alignment slipped inside an anchor gap.
                    raise ValueError(
                        f"Block/dataset order drift: identifier-less block "
                        f"paired with real-identifier dataset {ds_code!r} — "
                        f"refusing to join parameters by position."
                    )
                codes.append(ds_code or None)
        else:
            codes = []
            for block in blocks:
                meta_code = (
                    block.parsed.get("metadata", {}).get("Process identifier") or ""
                ).strip()
                codes.append(meta_code if meta_code and meta_code not in {'""', "''"} else None)

        rows: list[dict] = []
        skipped_no_code = 0
        n_child_rows = 0
        for block, code in zip(blocks, codes, strict=True):
            if not code:
                skipped_no_code += 1
                continue
            block_rows: list[dict] = []
            for kind, key in _BLOCK_KINDS:
                sub = block.blocks.get(key)
                if sub is None:
                    continue
                for obj in sub.parsed:
                    block_rows.append(
                        {
                            "process_code": code,
                            "kind": kind,
                            "name": obj.get("name"),
                            "original_name": obj.get("original_name") or obj.get("name"),
                            "amount": obj.get("amount"),
                            "formula": obj.get("formula"),
                            "comment": obj.get("comment") or "",
                        }
                    )
            rows.extend(block_rows)
            for child_code in children_by_parent.get(code, []):
                for row in block_rows:
                    # ``mf_parent_code`` preserves the linkage so a
                    # process-scoped override targeting the PARENT expands
                    # to its allocated children (see ``ParameterReevaluator.plan``).
                    rows.append({**row, "process_code": child_code, "mf_parent_code": code})
                    n_child_rows += 1
        if skipped_no_code:
            self._log.warning(
                "csv.parameters.skipped_no_code",
                n_processes=skipped_no_code,
            )
        self._log.info(
            "csv.parameters.extracted",
            n_rows=len(rows),
            n_child_rows=n_child_rows,
        )
        return rows
