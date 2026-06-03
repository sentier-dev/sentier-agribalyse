"""Unit tests for :class:`SkeletonExtractor`.

These are deliberately fixture-driven — we build a tiny content-
addressed scoring package on disk, run the extractor, and verify:

* ecoinvent columns are zero in both technosphere + biosphere.
* AGB columns are untouched (full numerical equality).
* characterization / corrections / ids.json are byte-equivalent copies.
* ``ecoinvent_slot_index.parquet`` carries exactly the ecoinvent cols.

The fixture mimics the on-disk layout :class:`ScoringPackageStore`
writes, with a 4×4 technosphere and a 3×4 biosphere — two AGB
activities, two ecoinvent activities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.skeleton_extractor import SkeletonExtractor


class TestSkeletonExtractor:
    @pytest.fixture
    def setup_package(self, tmp_path: Path):
        """Build a 4-col synthetic package with 2 AGB + 2 ecoinvent activities."""
        agb_codes = ["agb_a", "agb_b"]
        ei_codes = ["ei_x", "ei_y"]
        all_codes = agb_codes + ei_codes
        agb_db = "agribalyse-3.2"
        ei_db = "ecoinvent-3.9.1-cutoff"
        flow_ids = {
            code: ExchangeFrameBuilder.flow_id_for((agb_db if code in agb_codes else ei_db, code))
            for code in all_codes
        }
        col_id_to_idx = {flow_ids[c]: i for i, c in enumerate(all_codes)}

        # 4x4 technosphere — every activity produces its own product on
        # the diagonal; ecoinvent cols also have one consumption edge.
        tech = np.array(
            [
                [1.0, 0.0, -0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, -0.3],
                [0.0, 0.0, 0.0, 3.0],
            ],
            dtype="float64",
        )
        # 3x4 biosphere — one elementary flow each from ei_x and ei_y;
        # AGB cols emit a flow each as well.
        bio = np.array(
            [
                [0.1, 0.0, 0.7, 0.0],
                [0.0, 0.2, 0.0, 0.8],
                [0.0, 0.0, 0.6, 0.9],
            ],
            dtype="float64",
        )

        pkg_dir = tmp_path / "src_package"
        pkg_dir.mkdir()
        self._save_csr(pkg_dir / "technosphere.csr.npz", sp.csr_matrix(tech))
        self._save_csr(pkg_dir / "biosphere.csr.npz", sp.csr_matrix(bio))
        char_dir = pkg_dir / "characterization"
        char_dir.mkdir()
        char = sp.csr_matrix(np.array([[1.0, 2.0, 3.0]]))
        self._save_csr(char_dir / "test-method.csr.npz", char)
        ids = {
            "technosphere_row_id_to_idx": col_id_to_idx,
            "technosphere_col_id_to_idx": col_id_to_idx,
            "biosphere_row_id_to_idx": {"100": 0, "200": 1, "300": 2},
            "method_keys": [["test", "method"]],
            "correction_keys": [],
        }
        (pkg_dir / "ids.json").write_text(json.dumps(ids, sort_keys=True))

        # Synthetic ecoinvent_catalog covering the two ecoinvent codes.
        catalog = pd.DataFrame(
            [
                {
                    "database": ei_db,
                    "code": "ei_x",
                    "name": "ei_x_name",
                    "unit": "kg",
                    "location": "GLO",
                    "reference_product": "x",
                },
                {
                    "database": ei_db,
                    "code": "ei_y",
                    "name": "ei_y_name",
                    "unit": "kg",
                    "location": "RoW",
                    "reference_product": "y",
                },
                # An ecoinvent row that's NOT in the technosphere — should be skipped.
                {
                    "database": ei_db,
                    "code": "ei_z_extra",
                    "name": "ei_z_name",
                    "unit": "kg",
                    "location": "GLO",
                    "reference_product": "z",
                },
            ]
        )
        catalog_path = tmp_path / "ecoinvent_catalog.parquet"
        catalog.to_parquet(catalog_path)

        return pkg_dir, catalog_path, all_codes, col_id_to_idx, tech, bio, char

    @staticmethod
    def _save_csr(path: Path, csr: sp.csr_matrix) -> None:
        np.savez(
            path,
            data=csr.data,
            indices=csr.indices,
            indptr=csr.indptr,
            shape=np.asarray(csr.shape, dtype="int64"),
        )

    @staticmethod
    def _load_csr(path: Path) -> sp.csr_matrix:
        with np.load(path) as f:
            return sp.csr_matrix(
                (f["data"], f["indices"], f["indptr"]),
                shape=tuple(f["shape"]),
            )

    def test_ecoinvent_columns_are_zeroed(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _all_codes, _col_map, _, _, _ = setup_package
        out = tmp_path / "skel"
        SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg_dir, out)
        tech = self._load_csr(out / "technosphere.csr.npz").toarray()
        bio = self._load_csr(out / "biosphere.csr.npz").toarray()
        # ecoinvent activities sit at col indices 2, 3 in the fixture.
        assert np.all(tech[:, 2:] == 0)
        assert np.all(bio[:, 2:] == 0)

    def test_agb_columns_are_preserved(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _all_codes, _col_map, src_tech, src_bio, _ = setup_package
        out = tmp_path / "skel"
        SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg_dir, out)
        tech = self._load_csr(out / "technosphere.csr.npz").toarray()
        bio = self._load_csr(out / "biosphere.csr.npz").toarray()
        np.testing.assert_array_equal(tech[:, :2], src_tech[:, :2])
        np.testing.assert_array_equal(bio[:, :2], src_bio[:, :2])

    def test_characterization_copied_verbatim(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _all_codes, _col_map, _, _, src_char = setup_package
        out = tmp_path / "skel"
        SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg_dir, out)
        out_char = self._load_csr(out / "characterization" / "test-method.csr.npz")
        np.testing.assert_array_equal(out_char.toarray(), src_char.toarray())

    def test_ids_json_unchanged(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _, _, _, _, _ = setup_package
        out = tmp_path / "skel"
        SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg_dir, out)
        assert (pkg_dir / "ids.json").read_bytes() == (out / "ids.json").read_bytes()

    def test_slot_index_lists_only_present_ecoinvent_cols(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _all_codes, col_map, _, _, _ = setup_package
        out = tmp_path / "skel"
        SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg_dir, out)
        slot = pd.read_parquet(out / "ecoinvent_slot_index.parquet")
        # ei_z_extra is in the catalog but not in the technosphere → skipped.
        assert sorted(slot["code"].tolist()) == ["ei_x", "ei_y"]
        # col_idx must point at the actual matrix columns.
        for _, row in slot.iterrows():
            flow_id = ExchangeFrameBuilder.flow_id_for(("ecoinvent-3.9.1-cutoff", row["code"]))
            assert col_map[flow_id] == row["col_idx"]

    def test_refuses_to_overwrite_existing_output(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _, _, _, _, _ = setup_package
        out = tmp_path / "skel"
        out.mkdir()
        with pytest.raises(FileExistsError):
            SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg_dir, out)
