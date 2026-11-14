"""``SkeletonExtractor`` — produce the AGB-only scoring-package skeleton
shipped in ``sentier_agribalyse-bundle``.

BUNDLE.md §7 calls this out as the riskiest part of the bundling
pipeline. The extractor:

1. Reads an existing content-addressed scoring package
   (``cache/scoring_packages/<hash>/``), and refuses one built from
   ADEME's full export: there the copied ecoinvent blocks are Agribalyse
   processes keyed by their SimaPro process identifier, which the manifest
   of the stripped CSV lists. A skeleton is made from a package built on
   the ecoinvent-free CSV.
2. Identifies which technosphere columns correspond to ecoinvent
   activities by hashing every (database, code) row of
   ``registry/ecoinvent_catalog.parquet`` through
   :meth:`ExchangeFrameBuilder.flow_id_for` and matching against the
   ``technosphere_col_id_to_idx`` map in ``ids.json``.
3. Identifies the processes :class:`UnitCopyRebuilder` rebuilt from the
   user's ecoinvent snapshot: the 125 stripped unit blocks, two of them
   combustion extracts. They sit under the Agribalyse database, so step 2 does not
   see them, but their exchanges are ecoinvent's amounts. Their column ids
   are the hashes of ``(agribalyse database, prefix + ecoinvent code)``
   for every code of the catalog.
4. Zeros the columns of steps 2 and 3 in both the technosphere and the
   biosphere CSR matrices. Rows are kept: what an Agribalyse process draws
   from those columns is ADEME's amount.
5. Leaves the characterization vectors and the corrections rows
   untouched — CFs are per-biosphere-flow (not per-activity), and the
   correction rows for ecoinvent cols are still licence-clean because
   they encode our derived per-activity correction, not ecoinvent's
   own exchange values.
6. Writes the AGB skeleton to ``<output_root>/<hash>/`` and emits
   ``ecoinvent_slot_index.parquet`` so the customer-side filler knows
   which col_idx ↔ ecoinvent activity to refill, and
   ``rebuilt_slot_index.parquet`` for the rebuilt processes. Those two
   are apart because they are refilled differently: an ecoinvent slot
   takes the ecoinvent column as it is; a rebuilt slot takes the rebuild,
   whose inputs go to ADEME's adapted datasets where there is one.

The output directory mirrors :class:`ScoringPackageStore`'s on-disk
layout — the runtime never has to know it loaded a skeleton vs a
fully-populated package.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy import sparse as sp

from core.logging import Logging
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from transforms.unit_copy_rebuilder import UnitCopyRebuilder


@dataclass(frozen=True)
class SkeletonExtractor:
    """Strip ecoinvent IP out of a built scoring package, leaving an
    AGB-only skeleton plus a slot-index for the customer-side filler."""

    ECOINVENT_DATABASE: ClassVar[str] = "ecoinvent-3.9.1-cutoff"

    ecoinvent_catalog_path: Path
    """``registry/ecoinvent_catalog.parquet`` — source of truth for the
    set of ecoinvent activities and their (code, location, ref-product)."""

    agribalyse_database: str = "agribalyse-3.2"
    """The database the rebuilt processes were written under
    (``Settings.resolved_agribalyse_db_name``). A wrong name finds no
    rebuilt column, so the CLI passes the setting."""

    stripped_processes_path: Path | None = None
    """``source/agribalyse-3.2-stripped-ecoinvent-processes.json``: the
    copied ecoinvent blocks by SimaPro process identifier. When given, a
    package that holds one of them is refused. The CLI passes it."""

    def extract(self, source_dir: Path, output_dir: Path) -> Path:
        log = Logging.get("skeleton.extractor")
        log.info("skeleton.extract.start", source=str(source_dir), output=str(output_dir))
        self._validate_source(source_dir)
        ids = json.loads((source_dir / "ids.json").read_text())
        col_id_to_idx: dict[int, int] = {
            int(k): int(v) for k, v in ids["technosphere_col_id_to_idx"].items()
        }

        self._refuse_copied_blocks(col_id_to_idx)
        slot_df = self._build_slot_index(col_id_to_idx)
        rebuilt_df = self._build_rebuilt_index(col_id_to_idx)
        ecoinvent_cols = np.concatenate(
            [
                np.asarray(slot_df["col_idx"].to_numpy(), dtype="int64"),
                np.asarray(rebuilt_df["col_idx"].to_numpy(), dtype="int64"),
            ]
        )
        log.info(
            "skeleton.extract.cols_identified",
            n_ecoinvent_cols=len(slot_df),
            n_rebuilt_cols=len(rebuilt_df),
            n_total_cols=len(col_id_to_idx),
        )

        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing skeleton at {output_dir}. "
                f"Delete it first if you really mean to re-extract."
            )
        partial = output_dir.with_name(output_dir.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)

        n_cols = len(col_id_to_idx)
        mask = self._column_mask(n_cols, ecoinvent_cols)

        self._zero_csr(source_dir / "technosphere.csr.npz", partial / "technosphere.csr.npz", mask)
        self._zero_csr(source_dir / "biosphere.csr.npz", partial / "biosphere.csr.npz", mask)
        self._copy_dir(source_dir / "characterization", partial / "characterization")
        if (source_dir / "corrections").exists():
            self._copy_dir(source_dir / "corrections", partial / "corrections")
        shutil.copy2(source_dir / "ids.json", partial / "ids.json")
        slot_df.to_parquet(
            partial / "ecoinvent_slot_index.parquet", engine="pyarrow", compression="snappy"
        )
        rebuilt_df.to_parquet(
            partial / "rebuilt_slot_index.parquet", engine="pyarrow", compression="snappy"
        )

        partial.rename(output_dir)
        log.info("skeleton.extract.done", output=str(output_dir))
        return output_dir

    # ------------------------------------------------------------------

    @classmethod
    def _validate_source(cls, source_dir: Path) -> None:
        required = ["technosphere.csr.npz", "biosphere.csr.npz", "ids.json", "characterization"]
        for name in required:
            if not (source_dir / name).exists():
                raise FileNotFoundError(
                    f"Scoring package at {source_dir} is missing {name}. Cannot extract skeleton."
                )

    def _refuse_copied_blocks(self, col_id_to_idx: dict[int, int]) -> None:
        if self.stripped_processes_path is None:
            return
        removed = json.loads(self.stripped_processes_path.read_text())["removed"]
        found: dict[str, int] = {}
        for block in removed:
            key = (self.agribalyse_database, str(block["identifier"]))
            if ExchangeFrameBuilder.flow_id_for(key) in col_id_to_idx:
                found[block["type"]] = found.get(block["type"], 0) + 1
        if found:
            counts = ", ".join(f"{n} {kind}" for kind, n in sorted(found.items()))
            raise ValueError(
                "This scoring package was built from ADEME's full export: it holds "
                f"copied ecoinvent blocks as Agribalyse processes ({counts}). Their "
                "amounts are ecoinvent's. Build the package from the ecoinvent-free "
                "CSV and extract that one: move source/AGB32_final.CSV out of source/ "
                "(the full export is read whenever it is there), run dds-reset (the "
                "importer cache does not record which CSV it holds), then dds-link-all."
            )

    def _ecoinvent_rows(self) -> pd.DataFrame:
        cat = pd.read_parquet(self.ecoinvent_catalog_path)
        ecoinvent_rows = cat.loc[cat["database"] == self.ECOINVENT_DATABASE].copy()
        if ecoinvent_rows.empty:
            raise RuntimeError(
                f"ecoinvent_catalog has no rows for database={self.ECOINVENT_DATABASE!r}."
            )
        return ecoinvent_rows

    def _build_slot_index(self, col_id_to_idx: dict[int, int]) -> pd.DataFrame:
        ecoinvent_rows = self._ecoinvent_rows()
        ecoinvent_rows["flow_id"] = [
            ExchangeFrameBuilder.flow_id_for((db, code))
            for db, code in zip(
                ecoinvent_rows["database"].astype(str),
                ecoinvent_rows["code"].astype(str),
                strict=True,
            )
        ]
        ecoinvent_rows["col_idx"] = ecoinvent_rows["flow_id"].map(col_id_to_idx)
        present = ecoinvent_rows.loc[ecoinvent_rows["col_idx"].notna()].copy()
        present["col_idx"] = present["col_idx"].astype("int64")
        columns = ["col_idx", "code", "location", "reference_product", "unit", "name", "flow_id"]
        return present[columns].sort_values("col_idx").reset_index(drop=True)

    def _build_rebuilt_index(self, col_id_to_idx: dict[int, int]) -> pd.DataFrame:
        """The columns of processes rebuilt from ecoinvent under the
        Agribalyse database: ``kind`` says which rebuild, ``rebuilt_from``
        the ecoinvent code it was rebuilt from."""
        rebuilds = (
            ("unit_copy", UnitCopyRebuilder.PROCESS_PREFIX),
            ("combustion_extract", UnitCopyRebuilder.EXTRACT_PREFIX),
        )
        codes = self._ecoinvent_rows()
        frames = []
        for kind, prefix in rebuilds:
            rows = codes.copy()
            rows["kind"] = kind
            rows["rebuilt_from"] = rows["code"].astype(str)
            rows["flow_id"] = [
                ExchangeFrameBuilder.flow_id_for((self.agribalyse_database, prefix + code))
                for code in rows["rebuilt_from"]
            ]
            rows["col_idx"] = rows["flow_id"].map(col_id_to_idx)
            frames.append(rows.loc[rows["col_idx"].notna()])
        present = pd.concat(frames, ignore_index=True)
        present["col_idx"] = present["col_idx"].astype("int64")
        columns = [
            "col_idx",
            "kind",
            "rebuilt_from",
            "location",
            "reference_product",
            "unit",
            "name",
            "flow_id",
        ]
        return present[columns].sort_values("col_idx").reset_index(drop=True)

    @staticmethod
    def _column_mask(n_cols: int, zero_cols: np.ndarray) -> sp.dia_matrix:
        """Diagonal mask with 0 on zeroed cols, 1 elsewhere. ``M @ diag(mask)``
        zeroes the targeted columns of ``M``."""
        diag = np.ones(n_cols, dtype="float64")
        diag[zero_cols] = 0.0
        return sp.diags(diag, format="csr")

    @classmethod
    def _zero_csr(cls, src: Path, dst: Path, mask: sp.spmatrix) -> None:
        with np.load(src) as f:
            csr = sp.csr_matrix(
                (f["data"], f["indices"], f["indptr"]),
                shape=tuple(f["shape"]),
            )
        zeroed = (csr @ mask).tocsr()
        zeroed.eliminate_zeros()
        np.savez(
            dst,
            data=zeroed.data,
            indices=zeroed.indices,
            indptr=zeroed.indptr,
            shape=np.asarray(zeroed.shape, dtype="int64"),
        )

    @staticmethod
    def _copy_dir(src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
