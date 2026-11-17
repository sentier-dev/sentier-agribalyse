"""Files that travel with a scoring package land in the same atomic write."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from scoring.rebuilt_recipe import RebuiltRecipe
from scoring.scoring_package import ScoringPackageStore
from tests.fixtures.bw_synthetic import make_synthetic_scoring_package


@dataclass
class _Note:
    text: str = "first"
    fails: bool = False
    seen_final_directory: bool | None = None

    def write(self, package_dir: Path) -> Path:
        if self.fails:
            raise OSError("no space left")
        self.seen_final_directory = not package_dir.name.endswith(".partial")
        target = package_dir / "note.txt"
        target.write_text(self.text)
        return target


class TestCompanions:
    def test_a_companion_is_written_before_the_package_appears(self, tmp_path: Path):
        package = make_synthetic_scoring_package()
        note = _Note()

        target = ScoringPackageStore(root=tmp_path).write(package, companions=(note,))

        assert (target / "note.txt").read_text() == "first"
        assert note.seen_final_directory is False

    def test_a_companion_that_fails_leaves_no_package(self, tmp_path: Path):
        package = make_synthetic_scoring_package()
        store = ScoringPackageStore(root=tmp_path)

        with pytest.raises(OSError, match="no space left"):
            store.write(package, companions=(_Note(fails=True),))

        assert not store.path_for(package.content_hash).exists()

    def test_a_package_that_exists_gets_its_companion_again(self, tmp_path: Path):
        package = make_synthetic_scoring_package()
        store = ScoringPackageStore(root=tmp_path)
        store.write(package, companions=(_Note("first"),))
        again = _Note("second")

        target = store.write(package, companions=(again,))

        assert (target / "note.txt").read_text() == "second"
        assert again.seen_final_directory is True

    def test_a_package_without_companions_is_written_as_before(self, tmp_path: Path):
        package = make_synthetic_scoring_package()

        target = ScoringPackageStore(root=tmp_path).write(package)

        assert not (target / "note.txt").exists()
        assert (target / "ids.json").exists()

    def test_the_rebuilt_recipe_is_such_a_companion(self, tmp_path: Path):
        package = make_synthetic_scoring_package()
        recipe = RebuiltRecipe(frame=pd.DataFrame(columns=list(RebuiltRecipe.COLUMNS)))

        target = ScoringPackageStore(root=tmp_path).write(package, companions=(recipe,))

        assert (target / RebuiltRecipe.FILENAME).exists()
