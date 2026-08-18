"""``DatapackageWriter`` — EmbeddedInventory → bw_processing datapackages.

Writes one inventory datapackage (technosphere + biosphere) and one
characterization datapackage per method, each as a directory-filesystem
bw_processing datapackage. Matrix values are stored pre-signed with
``flip_array`` all-False so the matrices bw2calc assembles equal ``A``
and ``B`` exactly. Characterization is a diagonal on the biosphere flow
id. Demand is keyed by the product (row) id at calculation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bw_processing as bwp
import numpy as np

from bw_export.correction_embedder import EmbeddedInventory
from scoring.method_slug import MethodSlug


@dataclass(frozen=True)
class WriteResult:
    inventory_path: Path
    method_paths: dict[tuple[str, ...], Path]
    product_ids: list[int]


@dataclass(frozen=True)
class DatapackageWriter:
    """Stateless. ``write(embedded, out_root)`` lays out the datapackages."""

    def write(self, embedded: EmbeddedInventory, out_root: Path) -> WriteResult:
        out_root = Path(out_root)
        inv_dir = out_root / "inventory"
        inv_dir.mkdir(parents=True, exist_ok=True)

        inventory = bwp.create_datapackage(fs=bwp.generic_directory_filesystem(dirpath=inv_dir))
        t_coo = embedded.technosphere.tocoo()
        t_row_id = {v: k for k, v in embedded.technosphere_row_id_to_idx.items()}
        t_col_id = {v: k for k, v in embedded.technosphere_col_id_to_idx.items()}
        inventory.add_persistent_vector(
            matrix="technosphere_matrix",
            name="technosphere",
            data_array=t_coo.data.astype("float64"),
            indices_array=self._indices(
                [
                    (t_row_id[int(r)], t_col_id[int(c)])
                    for r, c in zip(t_coo.row, t_coo.col, strict=True)
                ]
            ),
            flip_array=np.zeros(t_coo.nnz, dtype=bool),
        )
        b_coo = embedded.biosphere.tocoo()
        b_row_id = {v: k for k, v in embedded.biosphere_row_id_to_idx.items()}
        inventory.add_persistent_vector(
            matrix="biosphere_matrix",
            name="biosphere",
            data_array=b_coo.data.astype("float64"),
            indices_array=self._indices(
                [
                    (b_row_id[int(r)], t_col_id[int(c)])
                    for r, c in zip(b_coo.row, b_coo.col, strict=True)
                ]
            ),
            flip_array=np.zeros(b_coo.nnz, dtype=bool),
        )
        inventory.finalize_serialization()

        method_paths: dict[tuple[str, ...], Path] = {}
        methods_root = out_root / "methods"
        methods_root.mkdir(parents=True, exist_ok=True)
        for method, cfs in embedded.method_cfs.items():
            slug = MethodSlug.encode(method)
            mdir = methods_root / slug
            mdir.mkdir(parents=True, exist_ok=True)
            method_dp = bwp.create_datapackage(fs=bwp.generic_directory_filesystem(dirpath=mdir))
            flow_ids = sorted(cfs)
            method_dp.add_persistent_vector(
                matrix="characterization_matrix",
                name="characterization",
                data_array=np.array([cfs[f] for f in flow_ids], dtype="float64"),
                indices_array=self._indices([(f, f) for f in flow_ids]),
            )
            method_dp.finalize_serialization()
            method_paths[method] = mdir

        return WriteResult(
            inventory_path=inv_dir,
            method_paths=method_paths,
            product_ids=sorted(embedded.technosphere_row_id_to_idx),
        )

    @staticmethod
    def _indices(pairs: list[tuple[int, int]]) -> np.ndarray:
        # Vectorised column assignment — the per-method characterization can
        # carry 80k+ CFs, so a Python-level loop here is needlessly slow.
        arr = np.empty(len(pairs), dtype=bwp.INDICES_DTYPE)
        if pairs:
            rows, cols = zip(*pairs, strict=True)
            arr["row"] = np.asarray(rows, dtype="int64")
            arr["col"] = np.asarray(cols, dtype="int64")
        return arr
