"""``AggregateDeleter`` — applies the ``deletions.parquet`` registry file.

Replaces the ad-hoc ``delete-aggregated-ecoinvent-{processes,products}.json``
loop in the legacy linker. Tightens matching by code where available
(fix 1.m): for processes we use ``code``, for products we fall back to
``name``-only matching only when no code is registered.

System-process safeguard (fix for FIX_DATA.md § 2 — ionising radiation
30–60× under-score). The deletions list mixes two kinds of AGB activities:

* **unit processes** — direct biosphere emissions plus technosphere
  inputs to upstream activities. These ARE genuine duplicates of an
  ecoinvent namesake; deleting them and routing to ecoinvent loses
  nothing because ecoinvent carries the same direct emissions and
  ecoinvent's supply chain produces the upstream contributions.
* **system processes** — direct biosphere emissions only, *no*
  technosphere inputs. AGB exported these with the cradle-to-gate
  upstream chain pre-aggregated into the direct biosphere edges
  (e.g. ``chemical factory construction, organics RER`` carries 2 971
  direct biosphere edges including 1.3×10¹⁰ kBq Radon-222 from
  upstream uranium-bearing cement and steel; ecoinvent's namesake has
  zero direct radon and reaches it only via its technosphere chain).
  Deleting these strips ~4.6×10¹⁰ kBq of Radon-222 (and similar
  fractions of every other long-lived radionuclide) from the entire
  AGB matrix and routes consumers to ecoinvent versions whose
  pass-through chains don't reproduce the same totals.

We therefore *skip* the deletion when the AGB activity has zero
``technosphere`` exchanges and at least one ``biosphere`` exchange. The
deduplicator then sees both AGB-system-process and ecoinvent-namesake
producers for the same product and resolves to one of them; AGB
consumers (every food activity in the system) reach the AGB system
process via the pre-existing technosphere link.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.logging import Logging
from registry import MappingRegistry


@dataclass(frozen=True)
class AggregateDeleter:
    """Apply the registry's deletions to a SimaPro importer.

    Skips deletions on AGB system processes — see module docstring.
    """

    registry: MappingRegistry

    @property
    def _log(self):
        return Logging.get(__name__)

    def apply(self, sp) -> dict[str, int]:
        df = self.registry.deletions
        if df.empty:
            return {"deleted": 0, "kept_system_processes": 0, "kept_paired_products": 0}

        process_codes: set[str] = set()
        process_names: set[str] = set()
        product_codes: set[str] = set()
        product_names: set[str] = set()
        for r in df.itertuples(index=False):
            code = self._coerce_optional_str(getattr(r, "code", None))
            name = self._coerce_optional_str(getattr(r, "name", None))
            # Name-based matching is a fallback for rows that carry no code.
            # When the row identifies a specific process by code, registering
            # the name too would also match same-named siblings (e.g. AGB's
            # ``fish canning, small fish RoW`` twins — the S-Copied system
            # process and the U-Adapted unit process share a display name but
            # only the S code is on the deletions list; matching the U-twin
            # by name silently dropped it, collapsing canned-seafood chains
            # onto flat-aggregated biosphere).
            if r.kind == "process":
                if code:
                    process_codes.add(code)
                elif name:
                    process_names.add(name)
            elif r.kind == "product":
                if code:
                    product_codes.add(code)
                elif name:
                    product_names.add(name)

        # Two-pass deletion:
        #   Pass 1 — identify the AGB system processes we are *keeping*
        #            (matched the deletions list but have zero technosphere
        #            inputs and ≥1 biosphere edge), and collect the product
        #            names they produce. Without keeping those paired
        #            products, every kept system process becomes an orphan
        #            producer that ``DanglingEdgePruner`` strips, undoing
        #            the whole point of the system-process safeguard.
        #   Pass 2 — drop only matched activities that are NOT a kept
        #            system process AND NOT a paired product of one.
        kept_system_codes: set[str] = set()
        kept_system_names: set[str] = set()
        paired_product_codes: set[str] = set()
        paired_product_names: set[str] = set()
        for ds in sp.data:
            if not self._matches(ds, process_codes, process_names, product_codes, product_names):
                continue
            if not self._is_system_process(ds):
                continue
            kept_system_codes.add(ds.get("code", "") or "")
            kept_system_names.add(ds.get("name", "") or "")
            for product_name, product_code in self._produces(ds):
                if product_name:
                    paired_product_names.add(product_name)
                if product_code:
                    paired_product_codes.add(product_code)

        kept_data: list[dict] = []
        kept_system = 0
        kept_paired = 0
        for ds in sp.data:
            if not self._matches(ds, process_codes, process_names, product_codes, product_names):
                kept_data.append(ds)
                continue
            ds_code = ds.get("code") or ds.get("simapro_identifier") or ""
            ds_name = ds.get("name", "")
            if (ds.get("type") or "process") == "process":
                if self._is_system_process(ds):
                    kept_data.append(ds)
                    kept_system += 1
                    continue
            else:
                # Product side — keep when paired with a kept system process.
                if ds_code in paired_product_codes or ds_name in paired_product_names:
                    kept_data.append(ds)
                    kept_paired += 1
                    continue
            # Falls through: drop.

        deleted = len(sp.data) - len(kept_data)
        sp.data = kept_data
        self._log.info(
            "transforms.delete_aggregated",
            deleted=deleted,
            kept_system_processes=kept_system,
            kept_paired_products=kept_paired,
        )
        return {
            "deleted": deleted,
            "kept_system_processes": kept_system,
            "kept_paired_products": kept_paired,
        }

    @staticmethod
    def _coerce_optional_str(value: object) -> str:
        """Treat ``None`` / pandas ``NaN`` / non-strings as empty.

        ``itertuples`` returns ``float('nan')`` for missing object cells
        because pandas promotes mixed object/None columns. ``bool(NaN)``
        is ``True``, so the historic ``r.code or ""`` short-circuit
        silently kept the NaN value and the rest of the matcher misread
        it as a real code.
        """
        if value is None or not isinstance(value, str):
            try:
                if pd.isna(value):
                    return ""
            except (TypeError, ValueError):
                pass
        return str(value) if isinstance(value, str) else ""

    @staticmethod
    def _produces(ds: dict) -> list[tuple[str, str]]:
        """Return ``(product_name, product_code)`` tuples for every
        production edge on ``ds``. Most processes have exactly one; a
        handful in AGB are multifunctional (production + substitution +
        co-product) and surface multiple."""
        out: list[tuple[str, str]] = []
        for exc in ds.get("exchanges", ()) or ():
            if exc.get("type") not in ("production", "generic production"):
                continue
            name = (exc.get("name") or "").strip()
            inp = exc.get("input") or ()
            code = inp[1] if isinstance(inp, (list, tuple)) and len(inp) >= 2 else ""
            out.append((name, code))
        return out

    @staticmethod
    def _matches(
        ds: dict,
        process_codes: set[str],
        process_names: set[str],
        product_codes: set[str],
        product_names: set[str],
    ) -> bool:
        ds_type = ds.get("type") or "process"
        ds_name = ds.get("name", "")
        ds_code = ds.get("code") or ds.get("simapro_identifier") or ""
        if ds_type == "product":
            return ds_code in product_codes or ds_name in product_names
        return ds_code in process_codes or ds_name in process_names

    @staticmethod
    def _is_system_process(ds: dict) -> bool:
        """An AGB activity is a *system process* iff it carries pre-aggregated
        cradle-to-gate biosphere edges and zero technosphere inputs. See
        module docstring for why we keep these instead of deleting them.

        We keep ``market for X`` SYS processes too even though their
        baked-in chloride / heavy-metal load over-counts freshwater
        ecotoxicity by ~3× on fish products. Without them the radon
        chain breaks: AGB ``construction`` SYS processes carry the radon
        but reach consumer activities only through ``market for`` AGB
        intermediaries; trimming the latter routes consumers to ecoinvent
        markets which never see the AGB construction radon, dropping
        ionising radiation back from 0.996× → 0.044× (verified). The
        ecotoxicity over-count is the hybrid-matrix tax for that —
        documented in FIX_DATA.md § 1 as an architectural residual.
        """
        if (ds.get("type") or "process") != "process":
            return False
        n_tech = 0
        n_bio = 0
        for exc in ds.get("exchanges", ()) or ():
            t = exc.get("type")
            if t == "technosphere":
                n_tech += 1
            elif t == "biosphere":
                n_bio += 1
        return n_tech == 0 and n_bio > 0
