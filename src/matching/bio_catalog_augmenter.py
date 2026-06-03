"""``BiosphereCatalogAugmenter`` — extend the catalog with regional variants.

The runtime biosphere catalog (``registry/biosphere_catalog.parquet``) is
built from snapshots of biosphere3 and ecoinvent's biosphere database
plus the EF flow registry. It has no notion of *country of extraction*:
``Water, lake`` is a single row, not 248 country-coded variants.

After :class:`matching.biosphere.BiosphereMatcher` rewrites the
country-coded source flows (``Water, lake, FR``) onto synthetic codes
(``<base_uuid>@FR``), the catalog has a row for the base UUID but
nothing for any of the synthetic ``@region``-suffixed codes. This
augmenter scans the linked SimaPro inventory, collects the set of
synthetic codes actually used, and rewrites the catalog parquet with
matching synthetic rows appended.

Each synthetic row inherits the name / categories / unit / cas /
synonyms of its base row but carries the synthetic code in the ``code``
column, plus two new columns:

* ``base_code`` — the original (non-suffixed) UUID.
* ``region`` — the country / aggregate code (``"FR"``, ``"RoW"``, …).

For non-regional rows the new columns default to ``(code, "")`` so the
schema remains stable.

This is the single source of truth for "which regional codes the
matrix knows about". Downstream consumers (``MethodCfRegistryBuilder``,
inspection tools) read the augmented parquet and don't need access to
the linked SimaPro data themselves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from core.logging import Logging
from core.parquet_io import ParquetAtomicWriter


@dataclass(frozen=True)
class BiosphereCatalogAugmenter:
    """Rewrite ``biosphere_catalog.parquet`` with synthetic regional rows.

    The augmenter is idempotent: running it twice over the same linked
    inventory yields the same catalog (the base rows are detected and
    re-used as the canonical source, never duplicated).
    """

    catalog_path: Path

    # Databases whose codes are allowed to carry a synthetic
    # ``@<region>`` suffix. Mirrors
    # ``BiosphereMatcher.REGIONAL_DB_ALLOWLIST``.
    REGIONAL_DB_ALLOWLIST: ClassVar[frozenset[str]] = frozenset(
        {"ecoinvent-3.9.1-biosphere", "biosphere3"}
    )

    SEPARATOR: ClassVar[str] = "@"

    @property
    def _log(self) -> Any:
        return Logging.get(__name__)

    # ------------------------------------------------------------------
    # Public entrypoint.

    def augment_from_sp_data(self, sp_data: list[dict]) -> int:
        """Collect synthetic codes from ``sp_data`` and rewrite the catalog.

        Returns the number of synthetic rows added (zero on idempotent
        re-runs after the catalog already contains them).
        """
        used = self._collect_synthetic_codes(sp_data)
        return self.augment(used)

    def augment(self, used: Iterable[tuple[str, str, str]]) -> int:
        """``used`` is an iterable of ``(target_db, base_code, region)``
        triples.

        Returns the number of synthetic rows newly inserted.
        """
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"biosphere catalog parquet missing: {self.catalog_path}. "
                f"Run `dds-build-biosphere-catalog` first."
            )
        df = pd.read_parquet(self.catalog_path)

        # Ensure new columns exist on the base frame so the schema is
        # stable regardless of whether augmentation has happened before.
        if "base_code" not in df.columns:
            df["base_code"] = df["code"]
        if "region" not in df.columns:
            df["region"] = ""

        existing = {(row.database, row.code) for row in df.itertuples(index=False)}
        # Index base rows by (database, base_code, name) so we can pick a
        # canonical row to clone for each synthetic.
        base_by_key = (
            df[df["region"] == ""]
            .drop_duplicates(subset=["database", "code"], keep="first")
            .set_index(["database", "code"])
        )

        synthetic_rows: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for db, base_code, region in used:
            if not region:
                continue
            if db not in self.REGIONAL_DB_ALLOWLIST:
                continue
            synth_code = f"{base_code}{self.SEPARATOR}{region}"
            key = (db, synth_code)
            if key in existing or key in seen:
                continue
            try:
                base_row = base_by_key.loc[(db, base_code)]
            except KeyError:
                # No base entry — skip rather than fabricate one.
                self._log.warning(
                    "biosphere.catalog.augment.missing_base",
                    database=db,
                    base_code=base_code,
                    region=region,
                )
                continue
            if isinstance(base_row, pd.DataFrame):
                base_row = base_row.iloc[0]
            cats = base_row["categories"]
            cats_list = list(cats) if cats is not None else []
            syns = base_row["synonyms"]
            syns_list = list(syns) if syns is not None else []
            cas_v = base_row["cas"]
            # ``pd.notna`` on a scalar string returns True; on a None
            # returns False. Arrays are not expected here because we
            # already dedup'd to a single row.
            try:
                cas_keep = bool(pd.notna(cas_v))
            except (TypeError, ValueError):
                cas_keep = cas_v is not None
            synthetic_rows.append(
                {
                    "database": db,
                    "code": synth_code,
                    "name": str(base_row["name"]),
                    "categories": cats_list,
                    "unit": str(base_row["unit"]),
                    "cas": str(cas_v) if cas_keep else None,
                    "synonyms": syns_list,
                    "base_code": base_code,
                    "region": region,
                }
            )
            seen.add(key)

        if not synthetic_rows:
            self._log.info(
                "biosphere.catalog.augment.no_change",
                path=str(self.catalog_path),
                n_existing_synthetic=int((df["region"] != "").sum()),
            )
            return 0

        synth_df = pd.DataFrame(synthetic_rows, columns=list(df.columns))
        out = pd.concat([df, synth_df], ignore_index=True)
        out = out.astype(
            {
                "database": "string",
                "code": "string",
                "name": "string",
                "unit": "string",
                "base_code": "string",
                "region": "string",
            }
        )
        out = out.sort_values(by=["database", "code"], kind="mergesort").reset_index(drop=True)
        ParquetAtomicWriter.write(out, self.catalog_path)
        self._log.info(
            "biosphere.catalog.augment.written",
            path=str(self.catalog_path),
            n_added=len(synthetic_rows),
            n_total=len(out),
        )
        return len(synthetic_rows)

    # ------------------------------------------------------------------
    # Helpers.

    @classmethod
    def _collect_synthetic_codes(
        cls,
        sp_data: list[dict],
    ) -> list[tuple[str, str, str]]:
        """Walk SimaPro datasets, return unique synthetic codes used."""
        seen: set[tuple[str, str, str]] = set()
        for ds in sp_data:
            for exc in ds.get("exchanges", ()):
                if exc.get("type") != "biosphere":
                    continue
                input_ = exc.get("input")
                if not (isinstance(input_, tuple) and len(input_) == 2):
                    continue
                db, code = input_
                if cls.SEPARATOR not in code:
                    continue
                base_code, _, region = code.partition(cls.SEPARATOR)
                if not region:
                    continue
                seen.add((str(db), base_code, region))
        return sorted(seen)
