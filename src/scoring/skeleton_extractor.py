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
   for every code of the catalog. A package linked on the ecoinvent-free
   CSV always holds some, so finding none (a wrong database name, or a
   package linked before the rebuild) refuses the package: their columns
   would keep ecoinvent's amounts.
4. Checks the recipe of the rebuilt columns before writing anything: the
   package's ``rebuilt_recipe.parquet`` (:class:`RebuiltRecipe`) names,
   for every entry of every rebuilt column, the ecoinvent exchange it came
   from, its unit multiplier and its sign. Applied to the licensed
   snapshot, normalised as the rebuilder read it, the recipe must
   reproduce each rebuilt column of the package; any difference, a missing
   recipe, or a recipe entry outside the rebuilt columns refuses the
   package. The recipe holds no amount.
5. Zeros the columns of steps 2 and 3 in both the technosphere and the
   biosphere CSR matrices. Rows are kept: what an Agribalyse process draws
   from those columns is ADEME's amount.
6. Leaves the characterization vectors and the corrections rows
   untouched — CFs are per-biosphere-flow (not per-activity), and the
   correction rows for ecoinvent cols are still licence-clean because
   they encode our derived per-activity correction, not ecoinvent's
   own exchange values.
7. Writes the AGB skeleton to ``<output_root>/<hash>/`` and emits
   ``ecoinvent_slot_index.parquet`` so the customer-side filler knows
   which col_idx ↔ ecoinvent activity to refill,
   ``rebuilt_slot_index.parquet`` for the rebuilt processes (their column
   and the kind of rebuild, nothing of ecoinvent's metadata), the checked
   recipe as ``rebuilt_recipe.parquet``, keyed by matrix row and column
   index, and ``ecoinvent_slot_cols.json``: every column the skeleton
   zeroed and a filler must fill, ecoinvent's and the rebuilt ones. That
   is the list the bundle's filler checks for completeness, so a filler
   that knows nothing of the recipe fails its check instead of writing
   empty rebuilt columns. The two kinds of slot are apart because they are
   refilled differently: an ecoinvent slot takes the ecoinvent column as
   it is; a rebuilt slot takes the recipe, whose inputs go to ADEME's
   adapted datasets where there is one.

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
from scoring.rebuilt_recipe import RebuiltRecipe
from transforms.unit_copy_rebuilder import UnitCopyRebuilder


@dataclass(frozen=True)
class SkeletonExtractor:
    """Strip ecoinvent IP out of a built scoring package, leaving an
    AGB-only skeleton plus a slot-index for the customer-side filler."""

    ECOINVENT_DATABASE: ClassVar[str] = "ecoinvent-3.9.1-cutoff"
    SKELETON_RECIPE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "matrix",
        "col_idx",
        "row_idx",
        "edge_type",
        "sign",
        "multiplier",
        *RebuiltRecipe.SOURCE_KEY,
    )
    REBUILT_SLOT_COLUMNS: ClassVar[tuple[str, ...]] = ("col_idx", "kind")
    """What ``rebuilt_slot_index.parquet`` ships: no name, location, unit,
    reference product or code of the ecoinvent activity behind a column."""
    RTOL: ClassVar[float] = 1e-9
    """Relative difference a recomputed entry may show against the package:
    the recipe sums an entry's sources in another order than the matrix
    builder, nothing else."""

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

    ecoinvent_exchanges_path: Path | None = None
    """``source/ecoinvent-3.9.1-cutoff-exchanges.parquet``: the licensed
    snapshot the package was built from. A package with rebuilt processes
    is checked against it through the recipe, and refused without it. The
    CLI passes it."""

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
        recipe = self._checked_recipe(source_dir, ids, rebuilt_df)
        zeroed_cols = np.concatenate(
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
        mask = self._column_mask(n_cols, zeroed_cols)

        self._zero_csr(source_dir / "technosphere.csr.npz", partial / "technosphere.csr.npz", mask)
        self._zero_csr(source_dir / "biosphere.csr.npz", partial / "biosphere.csr.npz", mask)
        self._copy_dir(source_dir / "characterization", partial / "characterization")
        if (source_dir / "corrections").exists():
            self._copy_dir(source_dir / "corrections", partial / "corrections")
        shutil.copy2(source_dir / "ids.json", partial / "ids.json")
        slot_df.to_parquet(
            partial / "ecoinvent_slot_index.parquet", engine="pyarrow", compression="snappy"
        )
        # Every slot a filler must fill: a filler that does not know the
        # recipe then fails its completeness check on the rebuilt columns.
        (partial / "ecoinvent_slot_cols.json").write_text(
            json.dumps(sorted(int(c) for c in zeroed_cols))
        )
        # What ships about the rebuilt processes is which columns they are and
        # which rebuild made them. The filler reads the column numbers and takes
        # everything else from the recipe; ecoinvent's names, locations, units
        # and codes stay out of the skeleton.
        rebuilt_df[list(self.REBUILT_SLOT_COLUMNS)].to_parquet(
            partial / "rebuilt_slot_index.parquet", engine="pyarrow", compression="snappy"
        )
        recipe.to_parquet(
            partial / RebuiltRecipe.FILENAME, engine="pyarrow", compression="snappy", index=False
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
                "CSV and extract that one: run dds-link-all --csv ecoinvent-free (the "
                "full export can stay in source/), then extract again."
            )

    def _checked_recipe(
        self, source_dir: Path, ids: dict, rebuilt_df: pd.DataFrame
    ) -> pd.DataFrame:
        """The recipe of the rebuilt columns, by row and column index, once it
        has reproduced every one of them from the snapshot.

        Fails closed: a missing recipe is refused, and so is a package in
        which no rebuilt column hashes under :attr:`agribalyse_database`.
        A recipe entry outside those columns is refused too."""
        recipe = RebuiltRecipe.read(source_dir)
        if recipe is None:
            raise ValueError(
                f"This scoring package has no {RebuiltRecipe.FILENAME}, so the columns of "
                "its processes rebuilt from ecoinvent could not be refilled. It was linked "
                "before the recipe existed: run dds-link-all again, then extract."
            )
        if rebuilt_df.empty:
            raise ValueError(
                "No process rebuilt from ecoinvent was found in this scoring package under "
                f"the Agribalyse database {self.agribalyse_database!r}. A package linked on "
                "the ecoinvent-free CSV holds them, so the name differs from the one the "
                "package was built with, or the package was linked before the rebuild. "
                "Their columns would keep ecoinvent's amounts; nothing was written."
            )
        n_rebuilt = len(rebuilt_df)
        snapshot = self.ecoinvent_exchanges_path
        if snapshot is None or not snapshot.exists():
            raise FileNotFoundError(
                f"This scoring package holds {n_rebuilt} processes rebuilt from ecoinvent. "
                "Checking their recipe needs the ecoinvent snapshot the package was built "
                f"from, and none was found at {snapshot}."
            )
        located = self._locate(recipe.frame, ids)
        rebuilt_cols = np.asarray(rebuilt_df["col_idx"].to_numpy(), dtype="int64")
        outside = set(located["col_idx"].tolist()) - set(rebuilt_cols.tolist())
        if outside:
            raise ValueError(
                f"The recipe has entries in {len(outside)} columns that are not rebuilt "
                f"processes under {self.agribalyse_database!r} (sample col_idx "
                f"{sorted(outside)[:5]}). Either the package was built under another "
                "Agribalyse database name, or the recipe belongs to another package: check "
                "the setting, or run dds-link-all again, then extract."
            )
        values = RebuiltRecipe.entry_values(located, pd.read_parquet(snapshot))
        technosphere = self._load_csr(source_dir / "technosphere.csr.npz")
        biosphere = self._load_csr(source_dir / "biosphere.csr.npz")
        indicator = np.zeros(technosphere.shape[1], dtype="float64")
        indicator[rebuilt_cols] = 1.0
        keep = sp.diags(indicator, format="csr")
        mismatches = 0
        sample: list[tuple[str, int, int]] = []
        worst = 0.0
        for matrix, actual in (("technosphere", technosphere), ("biosphere", biosphere)):
            part = located["matrix"] == matrix
            expected = sp.coo_matrix(
                (
                    values[part.to_numpy()],
                    (
                        located.loc[part, "row_idx"].to_numpy(),
                        located.loc[part, "col_idx"].to_numpy(),
                    ),
                ),
                shape=actual.shape,
            ).tocsr()
            n, positions, rel = self._differences(expected, (actual @ keep).tocsr())
            mismatches += n
            sample += [(matrix, r, c) for r, c in positions]
            worst = max(worst, rel)
        if mismatches:
            raise ValueError(
                f"The recipe does not reproduce the rebuilt columns of this package: "
                f"{mismatches} entries differ (sample (matrix, row_idx, col_idx): "
                f"{sample[:5]}). The package and the snapshot disagree, or a transform "
                "changed a rebuilt exchange after the rebuild. Nothing was written."
            )
        Logging.get("skeleton.extractor").info(
            "skeleton.extract.recipe_checked",
            n_rebuilt_cols=n_rebuilt,
            n_recipe_rows=len(located),
            max_rel_diff=worst,
        )
        return located[list(self.SKELETON_RECIPE_COLUMNS)].reset_index(drop=True)

    @classmethod
    def _locate(cls, recipe: pd.DataFrame, ids: dict) -> pd.DataFrame:
        """Add ``col_idx`` and ``row_idx`` from ``ids.json``; refuse an id it lacks."""
        cols = {int(k): int(v) for k, v in ids["technosphere_col_id_to_idx"].items()}
        rows = {
            "technosphere": {int(k): int(v) for k, v in ids["technosphere_row_id_to_idx"].items()},
            "biosphere": {int(k): int(v) for k, v in ids["biosphere_row_id_to_idx"].items()},
        }
        out = recipe.copy()
        out["col_idx"] = out["output_id"].map(cols)
        out["row_idx"] = [
            rows.get(str(m), {}).get(int(i))
            for m, i in zip(out["matrix"], out["input_id"], strict=True)
        ]
        unknown = out["col_idx"].isna() | out["row_idx"].isna()
        if unknown.any():
            raise ValueError(
                f"{int(unknown.sum())} recipe entries name a row or column the package does "
                "not have. The recipe belongs to another package: run dds-link-all again, "
                "then extract."
            )
        out["col_idx"] = out["col_idx"].astype("int64")
        out["row_idx"] = out["row_idx"].astype("int64")
        return out

    @classmethod
    def _differences(
        cls, expected: sp.csr_matrix, actual: sp.csr_matrix
    ) -> tuple[int, list[tuple[int, int]], float]:
        """Entries where ``expected`` and ``actual`` differ beyond :attr:`RTOL`:
        count, a few positions, and the largest relative difference seen."""
        diff = (expected - actual).tocoo()
        if diff.nnz == 0:
            return 0, [], 0.0
        r, c = diff.row, diff.col
        e = np.abs(np.asarray(expected[r, c]).ravel())
        a = np.abs(np.asarray(actual[r, c]).ravel())
        scale = np.maximum(e, a)
        gap = np.abs(diff.data)
        rel = np.divide(gap, scale, out=np.zeros_like(gap), where=scale > 0)
        # A value that is not finite never passes: NaN compares false to everything,
        # and an infinite gap is not beyond an infinite scale.
        bad = ~(gap <= cls.RTOL * scale) | ~np.isfinite(gap) | ~np.isfinite(scale)
        positions = list(zip(r[bad].tolist(), c[bad].tolist(), strict=True))
        return int(bad.sum()), positions[:5], float(rel.max(initial=0.0))

    @staticmethod
    def _load_csr(path: Path) -> sp.csr_matrix:
        with np.load(path) as f:
            return sp.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"]))

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
