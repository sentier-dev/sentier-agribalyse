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


class TestRebuiltEcoinventProcesses:
    """A package built from the ecoinvent-free CSV holds processes that
    :class:`UnitCopyRebuilder` rebuilt from the user's ecoinvent snapshot:
    the 125 stripped unit blocks, two of them combustion extracts. They sit
    under the Agribalyse database, so they are not ecoinvent columns by
    database, but their exchanges are ecoinvent's amounts. The skeleton must
    zero them as it zeroes ecoinvent's own columns, and must name them apart
    from the ecoinvent slots: refilling one means rebuilding it (its inputs
    go to ADEME's adapted datasets), not copying an ecoinvent column.

    Fixture: six columns: two Agribalyse processes, two ecoinvent
    activities, the rebuilt unit copy of ``ei_x`` and the combustion
    extract of ``ei_y``.
    """

    AGB_DB = "agribalyse-3.2"
    EI_DB = "ecoinvent-3.9.1-cutoff"

    @pytest.fixture
    def package(self, tmp_path: Path):
        from transforms.unit_copy_rebuilder import UnitCopyRebuilder

        keys = [
            (self.AGB_DB, "agb_a"),
            (self.AGB_DB, "agb_b"),
            (self.EI_DB, "ei_x"),
            (self.EI_DB, "ei_y"),
            (self.AGB_DB, UnitCopyRebuilder.PROCESS_PREFIX + "ei_x"),
            (self.AGB_DB, UnitCopyRebuilder.EXTRACT_PREFIX + "ei_y"),
        ]
        col_map = {ExchangeFrameBuilder.flow_id_for(k): i for i, k in enumerate(keys)}
        # Column 4 (the rebuilt unit copy) draws agb_b and both ecoinvent
        # activities and emits all three flows; column 5 (the combustion
        # extract) emits two. agb_a draws the rebuilt unit copy.
        tech = np.array(
            [
                [1.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, -0.11, 0.0],
                [0.0, 0.0, 2.0, -0.3, -0.22, 0.0],
                [0.0, 0.0, 0.0, 3.0, -0.33, 0.0],
                [-0.4, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, -0.6, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype="float64",
        )
        bio = np.array(
            [
                [0.1, 0.0, 0.7, 0.0, 0.44, 0.0],
                [0.0, 0.2, 0.0, 0.8, 0.55, 0.77],
                [0.0, 0.0, 0.6, 0.9, 0.66, 0.88],
            ],
            dtype="float64",
        )
        pkg = tmp_path / "src_package"
        (pkg / "characterization").mkdir(parents=True)
        TestSkeletonExtractor._save_csr(pkg / "technosphere.csr.npz", sp.csr_matrix(tech))
        TestSkeletonExtractor._save_csr(pkg / "biosphere.csr.npz", sp.csr_matrix(bio))
        TestSkeletonExtractor._save_csr(
            pkg / "characterization" / "test-method.csr.npz", sp.csr_matrix(np.ones((1, 3)))
        )
        ids = {
            "technosphere_row_id_to_idx": col_map,
            "technosphere_col_id_to_idx": col_map,
            "biosphere_row_id_to_idx": {"100": 0, "200": 1, "300": 2},
            "method_keys": [["test", "method"]],
            "correction_keys": [],
        }
        (pkg / "ids.json").write_text(json.dumps(ids, sort_keys=True))
        catalog = pd.DataFrame(
            [
                {
                    "database": self.EI_DB,
                    "code": code,
                    "name": f"{code}_name",
                    "unit": "kg",
                    "location": "GLO",
                    "reference_product": code,
                }
                for code in ("ei_x", "ei_y", "ei_z_extra")
            ]
        )
        catalog_path = tmp_path / "ecoinvent_catalog.parquet"
        catalog.to_parquet(catalog_path)
        return pkg, catalog_path, col_map, keys, tech, bio

    def _extract(self, package, tmp_path: Path) -> Path:
        pkg, catalog_path, *_ = package
        out = tmp_path / "skel"
        SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg, out)
        return out

    def test_no_amount_of_a_rebuilt_process_survives(self, package, tmp_path: Path):
        out = self._extract(package, tmp_path)
        tech = TestSkeletonExtractor._load_csr(out / "technosphere.csr.npz").toarray()
        bio = TestSkeletonExtractor._load_csr(out / "biosphere.csr.npz").toarray()
        assert np.all(tech[:, 2:] == 0), "ecoinvent or rebuilt columns kept an amount"
        assert np.all(bio[:, 2:] == 0), "ecoinvent or rebuilt columns kept an emission"

    def test_agribalyse_columns_and_rows_are_kept(self, package, tmp_path: Path):
        *_, src_tech, src_bio = package
        out = self._extract(package, tmp_path)
        tech = TestSkeletonExtractor._load_csr(out / "technosphere.csr.npz").toarray()
        bio = TestSkeletonExtractor._load_csr(out / "biosphere.csr.npz").toarray()
        # agb_a's draw on the rebuilt unit copy (row 4) is ADEME's amount: it stays.
        np.testing.assert_array_equal(tech[:, :2], src_tech[:, :2])
        np.testing.assert_array_equal(bio[:, :2], src_bio[:, :2])

    def test_rebuilt_processes_are_named_apart_from_the_ecoinvent_slots(
        self, package, tmp_path: Path
    ):
        _, _, col_map, keys, *_ = package
        out = self._extract(package, tmp_path)
        slots = pd.read_parquet(out / "ecoinvent_slot_index.parquet")
        assert sorted(slots["code"]) == ["ei_x", "ei_y"]
        rebuilt = pd.read_parquet(out / "rebuilt_slot_index.parquet")
        assert sorted(zip(rebuilt["kind"], rebuilt["rebuilt_from"], strict=True)) == [
            ("combustion_extract", "ei_y"),
            ("unit_copy", "ei_x"),
        ]
        for row in rebuilt.itertuples():
            assert col_map[int(row.flow_id)] == row.col_idx
            assert keys[row.col_idx][0] == self.AGB_DB


class TestPackageBuiltFromTheFullExport:
    """Built from ADEME's full export, a package holds the copied ecoinvent
    blocks themselves as Agribalyse processes, keyed by their SimaPro process
    identifier: the 125 unit blocks, and the aggregated copies that survive
    the build (:class:`AggregateDeleter` keeps those that are system
    processes). Their amounts are
    ecoinvent's. The manifest of the stripped CSV lists every one of them by
    identifier, so the extractor can tell such a package, and refuses it:
    a skeleton is made from a package built on the ecoinvent-free CSV.
    """

    AGB_DB = "agribalyse-3.2"

    @pytest.fixture
    def manifest(self, tmp_path: Path) -> Path:
        path = tmp_path / "stripped.json"
        path.write_text(
            json.dumps(
                {
                    "removed": [
                        {"identifier": "COPIEDSYSTEM0000000001", "type": "System", "sha256": "0"},
                        {
                            "identifier": "COPIEDUNIT000000000002",
                            "type": "Unit process",
                            "sha256": "0",
                        },
                    ]
                }
            )
        )
        return path

    def _package(self, tmp_path: Path, codes: list[str]) -> tuple[Path, Path]:
        keys = [(self.AGB_DB, code) for code in codes] + [("ecoinvent-3.9.1-cutoff", "ei_x")]
        col_map = {ExchangeFrameBuilder.flow_id_for(k): i for i, k in enumerate(keys)}
        n = len(keys)
        pkg = tmp_path / "src_package"
        (pkg / "characterization").mkdir(parents=True)
        TestSkeletonExtractor._save_csr(pkg / "technosphere.csr.npz", sp.identity(n, format="csr"))
        TestSkeletonExtractor._save_csr(
            pkg / "biosphere.csr.npz", sp.csr_matrix(np.full((1, n), 0.5))
        )
        TestSkeletonExtractor._save_csr(
            pkg / "characterization" / "m.csr.npz", sp.csr_matrix(np.ones((1, 1)))
        )
        ids = {
            "technosphere_row_id_to_idx": col_map,
            "technosphere_col_id_to_idx": col_map,
            "biosphere_row_id_to_idx": {"100": 0},
            "method_keys": [["m"]],
            "correction_keys": [],
        }
        (pkg / "ids.json").write_text(json.dumps(ids, sort_keys=True))
        catalog = tmp_path / "ecoinvent_catalog.parquet"
        pd.DataFrame(
            [
                {
                    "database": "ecoinvent-3.9.1-cutoff",
                    "code": "ei_x",
                    "name": "x",
                    "unit": "kg",
                    "location": "GLO",
                    "reference_product": "x",
                }
            ]
        ).to_parquet(catalog)
        return pkg, catalog

    def test_a_package_holding_a_copied_block_is_refused(self, manifest, tmp_path: Path):
        pkg, catalog = self._package(tmp_path, ["agb_a", "COPIEDSYSTEM0000000001"])
        out = tmp_path / "skel"
        extractor = SkeletonExtractor(
            ecoinvent_catalog_path=catalog, stripped_processes_path=manifest
        )
        with pytest.raises(ValueError, match="full export"):
            extractor.extract(pkg, out)
        assert not out.exists()
        assert not out.with_name(out.name + ".partial").exists()

    def test_a_package_built_on_the_ecoinvent_free_csv_is_extracted(self, manifest, tmp_path: Path):
        pkg, catalog = self._package(tmp_path, ["agb_a", "agb_b"])
        out = tmp_path / "skel"
        SkeletonExtractor(ecoinvent_catalog_path=catalog, stripped_processes_path=manifest).extract(
            pkg, out
        )
        assert (out / "ecoinvent_slot_index.parquet").exists()


class TestBuildSkeletonCli:
    """``dds-build-skeleton`` hands the extractor the Agribalyse database name
    and the manifest from the settings. Without them the extractor would find
    no rebuilt column under a configured database name, and would not refuse
    a package built from the full export."""

    AGB_DB = "agribalyse-configured"
    EI_DB = "ecoinvent-3.9.1-cutoff"
    HASH = "0" * 64

    def _settings(self, tmp_path: Path, codes: list[str], removed: list[dict]):
        from config import Settings
        from config.paths import Paths

        settings = Settings(paths=Paths(package_root=tmp_path), agribalyse_db_name=self.AGB_DB)
        paths = settings.paths
        keys = [(self.AGB_DB, code) for code in codes] + [(self.EI_DB, "ei_x")]
        col_map = {ExchangeFrameBuilder.flow_id_for(k): i for i, k in enumerate(keys)}
        n = len(keys)
        pkg = paths.scoring_packages_root / self.HASH
        (pkg / "characterization").mkdir(parents=True)
        TestSkeletonExtractor._save_csr(pkg / "technosphere.csr.npz", sp.identity(n, format="csr"))
        TestSkeletonExtractor._save_csr(
            pkg / "biosphere.csr.npz", sp.csr_matrix(np.full((1, n), 0.5))
        )
        TestSkeletonExtractor._save_csr(
            pkg / "characterization" / "m.csr.npz", sp.csr_matrix(np.ones((1, 1)))
        )
        ids = {
            "technosphere_row_id_to_idx": col_map,
            "technosphere_col_id_to_idx": col_map,
            "biosphere_row_id_to_idx": {"100": 0},
            "method_keys": [["m"]],
            "correction_keys": [],
        }
        (pkg / "ids.json").write_text(json.dumps(ids, sort_keys=True))
        paths.registry_ecoinvent_catalog.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "database": self.EI_DB,
                    "code": "ei_x",
                    "name": "x",
                    "unit": "kg",
                    "location": "GLO",
                    "reference_product": "x",
                }
            ]
        ).to_parquet(paths.registry_ecoinvent_catalog)
        paths.stripped_processes_json.parent.mkdir(parents=True, exist_ok=True)
        paths.stripped_processes_json.write_text(json.dumps({"removed": removed}))
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        return settings, bundle

    def _run(self, settings, bundle: Path) -> None:
        from cli.build_skeleton import BuildSkeletonCli

        cli = BuildSkeletonCli(settings=settings)
        cli.execute(
            cli.parser().parse_args(["--content-hash", self.HASH, "--bundle-root", str(bundle)])
        )

    def test_it_finds_rebuilt_processes_under_the_configured_database(self, tmp_path: Path):
        from transforms.unit_copy_rebuilder import UnitCopyRebuilder

        settings, bundle = self._settings(
            tmp_path, ["agb_a", UnitCopyRebuilder.PROCESS_PREFIX + "ei_x"], removed=[]
        )
        self._run(settings, bundle)
        rebuilt = pd.read_parquet(bundle / "skeleton" / self.HASH / "rebuilt_slot_index.parquet")
        assert rebuilt["rebuilt_from"].tolist() == ["ei_x"]

    def test_it_refuses_a_package_built_from_the_full_export(self, tmp_path: Path):
        removed = [{"identifier": "COPIEDSYSTEM0000000001", "type": "System", "sha256": "0"}]
        settings, bundle = self._settings(tmp_path, ["agb_a", "COPIEDSYSTEM0000000001"], removed)
        with pytest.raises(ValueError, match="full export"):
            self._run(settings, bundle)
        assert not (bundle / "skeleton" / self.HASH).exists()
