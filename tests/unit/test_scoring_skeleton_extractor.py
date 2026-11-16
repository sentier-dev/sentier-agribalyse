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

from scoring.ecoinvent_exchanges import EcoinventExchangesIngester
from scoring.exchange_frame_builder import ExchangeFrameBuilder
from scoring.rebuilt_recipe import RebuiltRecipe
from scoring.skeleton_extractor import SkeletonExtractor

BIO_DB = "ecoinvent-3.9.1-biosphere"


def _snapshot(path: Path, rows: list[tuple]) -> Path:
    """An ecoinvent exchanges snapshot: ``(output, input db, input, amount, type)``."""
    pd.DataFrame(
        [
            {
                "output_database": "ecoinvent-3.9.1-cutoff",
                "output_code": out,
                "input_database": db,
                "input_code": code,
                "amount": amount,
                "type": kind,
            }
            for out, db, code, amount, kind in rows
        ]
    ).to_parquet(path)
    return path


def _recipe(pkg: Path, rows: list[tuple]) -> None:
    """Write a package recipe: ``(matrix, output_id, input_id, edge_type,
    multiplier, (source output, source input db, source input, source type))``."""
    RebuiltRecipe(
        frame=RebuiltRecipe._typed(
            pd.DataFrame(
                [
                    {
                        "matrix": matrix,
                        "output_id": out,
                        "input_id": inp,
                        "edge_type": edge,
                        "sign": RebuiltRecipe.sign_for(edge),
                        "multiplier": multiplier,
                        **dict(zip(RebuiltRecipe.SOURCE_KEY, source, strict=True)),
                    }
                    for matrix, out, inp, edge, multiplier, source in rows
                ],
                columns=list(RebuiltRecipe.COLUMNS),
            )
        )
    ).write(pkg)


def _seed_unit_copy(pkg: Path, snapshot: Path, rebuilt_id: int, bio_row_id: int, emission: float):
    """Recipe and snapshot for a rebuilt unit copy of ``ei_x`` whose column
    holds a production of 1.0 on its own row and one emission."""
    ei_db = "ecoinvent-3.9.1-cutoff"
    production = ("ei_x", ei_db, "ei_x", "production")
    flow = ("ei_x", BIO_DB, "f100", "biosphere")
    _recipe(
        pkg,
        [
            ("technosphere", rebuilt_id, rebuilt_id, "production", 1.0, production),
            ("biosphere", rebuilt_id, bio_row_id, "biosphere", 1.0, flow),
        ],
    )
    _snapshot(
        snapshot,
        [
            ("ei_x", ei_db, "ei_x", 1.0, "production"),
            ("ei_x", BIO_DB, "f100", emission, "biosphere"),
        ],
    )


class TestSkeletonExtractor:
    """A package linked on the ecoinvent-free CSV always holds rebuilt
    processes, so the fixture has one: the unit copy of ``ei_x`` at column 4,
    with its recipe and the snapshot it was rebuilt from."""

    @pytest.fixture
    def setup_package(self, tmp_path: Path):
        """Build a 5-col synthetic package: 2 AGB, 2 ecoinvent, 1 rebuilt."""
        from transforms.unit_copy_rebuilder import UnitCopyRebuilder

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
        rebuilt_id = ExchangeFrameBuilder.flow_id_for(
            (agb_db, UnitCopyRebuilder.PROCESS_PREFIX + "ei_x")
        )
        col_id_to_idx[rebuilt_id] = 4

        # 5x5 technosphere — every activity produces its own product on
        # the diagonal; ecoinvent cols also have one consumption edge, and
        # agb_b draws the rebuilt unit copy.
        tech = np.array(
            [
                [1.0, 0.0, -0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, -0.3, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, -0.4, 0.0, 0.0, 1.0],
            ],
            dtype="float64",
        )
        # 3x5 biosphere — one elementary flow each from ei_x and ei_y;
        # AGB cols emit a flow each as well, the rebuilt column one.
        bio = np.array(
            [
                [0.1, 0.0, 0.7, 0.0, 0.5],
                [0.0, 0.2, 0.0, 0.8, 0.0],
                [0.0, 0.0, 0.6, 0.9, 0.0],
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
        _seed_unit_copy(pkg_dir, self._snapshot_path(catalog_path), rebuilt_id, 100, 0.5)

        return pkg_dir, catalog_path, all_codes, col_id_to_idx, tech, bio, char

    @staticmethod
    def _snapshot_path(catalog_path: Path) -> Path:
        return catalog_path.with_name("exchanges.parquet")

    @classmethod
    def _extractor(cls, catalog_path: Path) -> SkeletonExtractor:
        return SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path,
            ecoinvent_exchanges_path=cls._snapshot_path(catalog_path),
        )

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
        self._extractor(catalog_path).extract(pkg_dir, out)
        tech = self._load_csr(out / "technosphere.csr.npz").toarray()
        bio = self._load_csr(out / "biosphere.csr.npz").toarray()
        # ecoinvent activities sit at col indices 2, 3, the rebuilt one at 4.
        assert np.all(tech[:, 2:] == 0)
        assert np.all(bio[:, 2:] == 0)

    def test_agb_columns_are_preserved(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _all_codes, _col_map, src_tech, src_bio, _ = setup_package
        out = tmp_path / "skel"
        self._extractor(catalog_path).extract(pkg_dir, out)
        tech = self._load_csr(out / "technosphere.csr.npz").toarray()
        bio = self._load_csr(out / "biosphere.csr.npz").toarray()
        np.testing.assert_array_equal(tech[:, :2], src_tech[:, :2])
        np.testing.assert_array_equal(bio[:, :2], src_bio[:, :2])

    def test_characterization_copied_verbatim(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _all_codes, _col_map, _, _, src_char = setup_package
        out = tmp_path / "skel"
        self._extractor(catalog_path).extract(pkg_dir, out)
        out_char = self._load_csr(out / "characterization" / "test-method.csr.npz")
        np.testing.assert_array_equal(out_char.toarray(), src_char.toarray())

    def test_ids_json_unchanged(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _, _, _, _, _ = setup_package
        out = tmp_path / "skel"
        self._extractor(catalog_path).extract(pkg_dir, out)
        assert (pkg_dir / "ids.json").read_bytes() == (out / "ids.json").read_bytes()

    def test_slot_cols_json_lists_every_column_a_filler_must_fill(
        self, setup_package, tmp_path: Path
    ):
        """``ecoinvent_slot_cols.json`` is the slot list the bundle's filler
        checks for completeness. It lists the rebuilt columns too, so a filler
        that does not know the recipe fails its check instead of leaving them
        empty."""
        pkg_dir, catalog_path, *_ = setup_package
        out = tmp_path / "skel"
        self._extractor(catalog_path).extract(pkg_dir, out)
        slot = pd.read_parquet(out / "ecoinvent_slot_index.parquet")
        rebuilt = pd.read_parquet(out / "rebuilt_slot_index.parquet")
        assert rebuilt["col_idx"].tolist() == [4]
        assert json.loads((out / "ecoinvent_slot_cols.json").read_text()) == sorted(
            [*slot["col_idx"].tolist(), *rebuilt["col_idx"].tolist()]
        )

    def test_slot_index_lists_only_present_ecoinvent_cols(self, setup_package, tmp_path: Path):
        pkg_dir, catalog_path, _all_codes, col_map, _, _, _ = setup_package
        out = tmp_path / "skel"
        self._extractor(catalog_path).extract(pkg_dir, out)
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
            self._extractor(catalog_path).extract(pkg_dir, out)


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
    extract of ``ei_y``, the package's recipe for both rebuilt columns, and
    the snapshot they were rebuilt from. ``ei_y`` is a waste treatment in
    ecoinvent's sign convention, so the unit copy's draw on it is read
    flipped, as the rebuilder reads it.
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
        fid = [ExchangeFrameBuilder.flow_id_for(k) for k in keys]
        tech_rows = "technosphere"
        _recipe(
            pkg,
            [
                (
                    tech_rows,
                    fid[4],
                    fid[4],
                    "production",
                    1.0,
                    ("ei_x", self.EI_DB, "ei_x", "production"),
                ),
                (
                    tech_rows,
                    fid[4],
                    fid[1],
                    "technosphere",
                    0.001,
                    ("ei_x", self.EI_DB, "soy", "technosphere"),
                ),
                (
                    tech_rows,
                    fid[4],
                    fid[2],
                    "technosphere",
                    1.0,
                    ("ei_x", self.EI_DB, "ei_x", "technosphere"),
                ),
                (
                    tech_rows,
                    fid[4],
                    fid[3],
                    "technosphere",
                    1.0,
                    ("ei_x", self.EI_DB, "ei_y", "technosphere"),
                ),
                ("biosphere", fid[4], 100, "biosphere", 1.0, ("ei_x", BIO_DB, "f100", "biosphere")),
                ("biosphere", fid[4], 200, "biosphere", 1.0, ("ei_x", BIO_DB, "f200", "biosphere")),
                ("biosphere", fid[4], 300, "biosphere", 1.0, ("ei_x", BIO_DB, "f300", "biosphere")),
                (
                    tech_rows,
                    fid[5],
                    fid[5],
                    "production",
                    1.0,
                    ("ei_y", self.EI_DB, "fuel", "technosphere"),
                ),
                ("biosphere", fid[5], 200, "biosphere", 1.0, ("ei_y", BIO_DB, "f200", "biosphere")),
                ("biosphere", fid[5], 300, "biosphere", 1.0, ("ei_y", BIO_DB, "f300", "biosphere")),
            ],
        )
        snapshot = _snapshot(
            tmp_path / "exchanges.parquet",
            [
                ("ei_x", self.EI_DB, "ei_x", 1.0, "production"),
                ("ei_x", self.EI_DB, "soy", 110.0, "technosphere"),
                ("ei_x", self.EI_DB, "ei_x", 0.22, "technosphere"),
                ("ei_x", self.EI_DB, "ei_y", -0.33, "technosphere"),
                ("ei_x", BIO_DB, "f100", 0.44, "biosphere"),
                ("ei_x", BIO_DB, "f200", 0.55, "biosphere"),
                ("ei_x", BIO_DB, "f300", 0.66, "biosphere"),
                ("ei_y", self.EI_DB, "ei_y", -3.0, "production"),
                ("ei_y", self.EI_DB, "fuel", 1.0, "technosphere"),
                ("ei_y", BIO_DB, "f200", 0.77, "biosphere"),
                ("ei_y", BIO_DB, "f300", 0.88, "biosphere"),
            ],
        )
        return pkg, catalog_path, col_map, keys, tech, bio, snapshot

    def _extract(self, package, tmp_path: Path) -> Path:
        pkg, catalog_path, *_, snapshot = package
        out = tmp_path / "skel"
        SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path, ecoinvent_exchanges_path=snapshot
        ).extract(pkg, out)
        return out

    def test_no_amount_of_a_rebuilt_process_survives(self, package, tmp_path: Path):
        out = self._extract(package, tmp_path)
        tech = TestSkeletonExtractor._load_csr(out / "technosphere.csr.npz").toarray()
        bio = TestSkeletonExtractor._load_csr(out / "biosphere.csr.npz").toarray()
        assert np.all(tech[:, 2:] == 0), "ecoinvent or rebuilt columns kept an amount"
        assert np.all(bio[:, 2:] == 0), "ecoinvent or rebuilt columns kept an emission"

    def test_agribalyse_columns_and_rows_are_kept(self, package, tmp_path: Path):
        *_, src_tech, src_bio, _snapshot_path = package
        out = self._extract(package, tmp_path)
        tech = TestSkeletonExtractor._load_csr(out / "technosphere.csr.npz").toarray()
        bio = TestSkeletonExtractor._load_csr(out / "biosphere.csr.npz").toarray()
        # agb_a's draw on the rebuilt unit copy (row 4) is ADEME's amount: it stays.
        np.testing.assert_array_equal(tech[:, :2], src_tech[:, :2])
        np.testing.assert_array_equal(bio[:, :2], src_bio[:, :2])

    def test_rebuilt_processes_are_named_apart_from_the_ecoinvent_slots(
        self, package, tmp_path: Path
    ):
        from transforms.unit_copy_rebuilder import UnitCopyRebuilder

        _, _, _col_map, keys, *_ = package
        out = self._extract(package, tmp_path)
        slots = pd.read_parquet(out / "ecoinvent_slot_index.parquet")
        assert sorted(slots["code"]) == ["ei_x", "ei_y"]
        rebuilt = pd.read_parquet(out / "rebuilt_slot_index.parquet")
        # Column and kind of rebuild only: no name, location, unit or code of
        # the ecoinvent activity behind a column ships in the skeleton.
        assert list(rebuilt.columns) == ["col_idx", "kind"]
        assert sorted((row.kind, keys[row.col_idx]) for row in rebuilt.itertuples()) == [
            ("combustion_extract", (self.AGB_DB, UnitCopyRebuilder.EXTRACT_PREFIX + "ei_y")),
            ("unit_copy", (self.AGB_DB, UnitCopyRebuilder.PROCESS_PREFIX + "ei_x")),
        ]

    def test_the_skeleton_recipe_refills_every_rebuilt_column(self, package, tmp_path: Path):
        """Applied to the snapshot (waste signs normalised, as the rebuilder
        read it), the skeleton's recipe gives back the package's rebuilt
        columns, so a licensed user can refill them."""
        *_, src_tech, src_bio, snapshot = package
        out = self._extract(package, tmp_path)
        recipe = pd.read_parquet(out / RebuiltRecipe.FILENAME)
        exchanges = EcoinventExchangesIngester._normalise_waste_signs(pd.read_parquet(snapshot))
        amount = {
            (r.output_code, r.input_database, r.input_code, r.type): r.amount
            for r in exchanges.itertuples()
        }
        tech = TestSkeletonExtractor._load_csr(out / "technosphere.csr.npz").toarray()
        bio = TestSkeletonExtractor._load_csr(out / "biosphere.csr.npz").toarray()
        for r in recipe.itertuples():
            source = (r.source_output_code, r.source_input_database, r.source_input_code)
            value = amount[(*source, r.source_type)] * r.multiplier * r.sign
            target = tech if r.matrix == "technosphere" else bio
            target[r.row_idx, r.col_idx] += value
        # Columns 2 and 3 are ecoinvent's own: the filler copies those, not the recipe.
        kept = [0, 1, 4, 5]
        np.testing.assert_allclose(tech[:, kept], src_tech[:, kept], rtol=1e-12, atol=0)
        np.testing.assert_allclose(bio[:, kept], src_bio[:, kept], rtol=1e-12, atol=0)

    def test_the_skeleton_recipe_holds_no_amount(self, package, tmp_path: Path):
        out = self._extract(package, tmp_path)
        recipe = pd.read_parquet(out / RebuiltRecipe.FILENAME)
        assert tuple(recipe.columns) == SkeletonExtractor.SKELETON_RECIPE_COLUMNS
        assert set(recipe["sign"]) <= {1.0, -1.0}
        assert sorted(recipe["col_idx"].unique()) == [4, 5]

    @pytest.mark.parametrize(
        ("change", "value"),
        [("soy", 111.0), ("f300", 0.67)],
    )
    def test_a_package_the_recipe_does_not_reproduce_is_refused(
        self, package, tmp_path: Path, change, value
    ):
        """One source amount differs between the snapshot and what the package
        holds: the extractor refuses and writes nothing."""
        pkg, catalog_path, *_, snapshot = package
        exchanges = pd.read_parquet(snapshot)
        exchanges.loc[exchanges["input_code"] == change, "amount"] = value
        exchanges.to_parquet(snapshot)
        out = tmp_path / "skel"
        extractor = SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path, ecoinvent_exchanges_path=snapshot
        )
        with pytest.raises(ValueError, match="does not reproduce"):
            extractor.extract(pkg, out)
        assert not out.exists()
        assert not out.with_name(out.name + ".partial").exists()

    def test_a_package_entry_the_recipe_lacks_is_refused(self, package, tmp_path: Path):
        pkg, catalog_path, *_, snapshot = package
        recipe = RebuiltRecipe.read(pkg)
        RebuiltRecipe(frame=recipe.frame.iloc[1:].reset_index(drop=True)).write(pkg)
        extractor = SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path, ecoinvent_exchanges_path=snapshot
        )
        with pytest.raises(ValueError, match="does not reproduce"):
            extractor.extract(pkg, tmp_path / "skel")

    def test_a_package_without_a_recipe_is_refused(self, package, tmp_path: Path):
        pkg, catalog_path, *_, snapshot = package
        (pkg / RebuiltRecipe.FILENAME).unlink()
        extractor = SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path, ecoinvent_exchanges_path=snapshot
        )
        with pytest.raises(ValueError, match="dds-link-all again"):
            extractor.extract(pkg, tmp_path / "skel")
        assert not (tmp_path / "skel").exists()

    @pytest.mark.parametrize("recipe", ["kept", "empty"])
    def test_a_wrong_agribalyse_database_name_is_refused(self, package, tmp_path: Path, recipe):
        """Hashing finds no rebuilt column under another name, which would
        leave their amounts in the skeleton. Any package linked on the
        ecoinvent-free CSV holds rebuilt processes, so finding none refuses
        the package, whatever the recipe holds."""
        pkg, catalog_path, *_, snapshot = package
        if recipe == "empty":
            RebuiltRecipe(frame=RebuiltRecipe.read(pkg).frame.iloc[0:0]).write(pkg)
        extractor = SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path,
            agribalyse_database="agribalyse-other",
            ecoinvent_exchanges_path=snapshot,
        )
        with pytest.raises(ValueError, match="No process rebuilt from ecoinvent"):
            extractor.extract(pkg, tmp_path / "skel")
        assert not (tmp_path / "skel").exists()

    def test_a_missing_recipe_is_refused_under_any_database_name(self, package, tmp_path: Path):
        """A missing recipe is refused first and for what it is, also when no
        rebuilt column hashes under the given name."""
        pkg, catalog_path, *_, snapshot = package
        (pkg / RebuiltRecipe.FILENAME).unlink()
        extractor = SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path,
            agribalyse_database="agribalyse-other",
            ecoinvent_exchanges_path=snapshot,
        )
        with pytest.raises(ValueError, match=r"no rebuilt_recipe\.parquet"):
            extractor.extract(pkg, tmp_path / "skel")
        assert not (tmp_path / "skel").exists()

    def test_recipe_entries_outside_the_rebuilt_columns_are_refused(self, package, tmp_path: Path):
        pkg, catalog_path, _col_map, keys, *_, snapshot = package
        frame = RebuiltRecipe.read(pkg).frame.copy()
        # Move one entry to the Agribalyse column agb_a.
        frame.loc[0, "output_id"] = ExchangeFrameBuilder.flow_id_for(keys[0])
        RebuiltRecipe(frame=frame).write(pkg)
        extractor = SkeletonExtractor(
            ecoinvent_catalog_path=catalog_path, ecoinvent_exchanges_path=snapshot
        )
        with pytest.raises(ValueError, match="not rebuilt processes"):
            extractor.extract(pkg, tmp_path / "skel")

    def test_without_the_snapshot_it_refuses(self, package, tmp_path: Path):
        pkg, catalog_path, *_ = package
        with pytest.raises(FileNotFoundError, match="snapshot"):
            SkeletonExtractor(ecoinvent_catalog_path=catalog_path).extract(pkg, tmp_path / "skel")
        assert not (tmp_path / "skel").exists()


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
        from transforms.unit_copy_rebuilder import UnitCopyRebuilder

        rebuilt = UnitCopyRebuilder.PROCESS_PREFIX + "ei_x"
        pkg, catalog = self._package(tmp_path, ["agb_a", rebuilt])
        snapshot = tmp_path / "exchanges.parquet"
        rebuilt_id = ExchangeFrameBuilder.flow_id_for((self.AGB_DB, rebuilt))
        _seed_unit_copy(pkg, snapshot, rebuilt_id, 100, 0.5)
        out = tmp_path / "skel"
        SkeletonExtractor(
            ecoinvent_catalog_path=catalog,
            stripped_processes_path=manifest,
            ecoinvent_exchanges_path=snapshot,
        ).extract(pkg, out)
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
        from transforms.unit_copy_rebuilder import UnitCopyRebuilder

        rebuilt = (self.AGB_DB, UnitCopyRebuilder.PROCESS_PREFIX + "ei_x")
        paths.ecoinvent_exchanges_parquet.parent.mkdir(parents=True, exist_ok=True)
        if rebuilt in keys:
            _seed_unit_copy(
                pkg,
                paths.ecoinvent_exchanges_parquet,
                ExchangeFrameBuilder.flow_id_for(rebuilt),
                100,
                0.5,
            )
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
        """Also checks the snapshot is handed over: the rebuilt column's recipe
        is checked against it, and without it the extractor refuses."""
        from transforms.unit_copy_rebuilder import UnitCopyRebuilder

        settings, bundle = self._settings(
            tmp_path, ["agb_a", UnitCopyRebuilder.PROCESS_PREFIX + "ei_x"], removed=[]
        )
        self._run(settings, bundle)
        skeleton = bundle / "skeleton" / self.HASH
        rebuilt = pd.read_parquet(skeleton / "rebuilt_slot_index.parquet")
        assert rebuilt.to_dict("records") == [{"col_idx": 1, "kind": "unit_copy"}]
        recipe = pd.read_parquet(skeleton / RebuiltRecipe.FILENAME)
        assert recipe["col_idx"].tolist() == [1, 1]

    def test_it_refuses_a_package_built_from_the_full_export(self, tmp_path: Path):
        removed = [{"identifier": "COPIEDSYSTEM0000000001", "type": "System", "sha256": "0"}]
        settings, bundle = self._settings(tmp_path, ["agb_a", "COPIEDSYSTEM0000000001"], removed)
        with pytest.raises(ValueError, match="full export"):
            self._run(settings, bundle)
        assert not (bundle / "skeleton" / self.HASH).exists()


class TestDifferences:
    """The comparison behind the extractor's checks."""

    @staticmethod
    def _matrix(value: float) -> sp.csr_matrix:
        return sp.csr_matrix(np.array([[1.0, value], [0.0, 2.0]]))

    def test_equal_matrices_have_no_difference(self):
        n, positions, worst = SkeletonExtractor._differences(self._matrix(3.0), self._matrix(3.0))
        assert (n, positions, worst) == (0, [], 0.0)

    def test_a_difference_within_the_tolerance_passes(self):
        n, _, _ = SkeletonExtractor._differences(self._matrix(3.0), self._matrix(3.0 * (1 + 1e-11)))
        assert n == 0

    def test_a_difference_beyond_the_tolerance_is_counted(self):
        n, positions, _ = SkeletonExtractor._differences(self._matrix(3.0), self._matrix(3.1))
        assert (n, positions) == (1, [(0, 1)])

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("side", ["expected", "actual", "both"])
    def test_a_value_that_is_not_finite_never_passes(self, value: float, side: str):
        expected = self._matrix(value if side in ("expected", "both") else 3.0)
        actual = self._matrix(value if side in ("actual", "both") else 3.0)
        n, positions, _ = SkeletonExtractor._differences(expected, actual)
        assert (n, positions) == (1, [(0, 1)])
