"""The pre-commit guard and .gitignore around the two Agribalyse CSV variants."""

from __future__ import annotations

import lzma
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.org"],
        ["config", "user.name", "t"],
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True)
    shutil.copytree(REPO_ROOT / ".githooks", tmp_path / ".githooks")
    shutil.copy(REPO_ROOT / ".gitignore", tmp_path / ".gitignore")
    (tmp_path / "source").mkdir()
    return tmp_path


ARCHIVE = "source/AGB32_ecoinvent_free.CSV.xz"


def _archive(product: str, *, section: str = "Products", inputs: str = "") -> bytes:
    text = (
        "{SimaPro 9.6.0.1}\r\n\r\nProcess\r\n\r\n"
        "Type\r\nUnit process\r\n\r\n"
        f"{section}\r\n{product};kg;1;100;not defined;Cat;\r\n\r\n"
        f"Materials/fuels\r\n{inputs}\r\n\r\n"
        "End\r\n\r\n"
    )
    return lzma.compress(text.encode("latin-1"), format=lzma.FORMAT_XZ)


def _guard(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / ".githooks" / "pre-commit")], cwd=repo, capture_output=True, text=True
    )


def _stage(repo: Path, relative: str, *, force: bool = False, content: bytes = b"x") -> None:
    path = repo / relative
    path.write_bytes(content)
    subprocess.run(["git", "add", *(["-f"] if force else []), relative], cwd=repo, check=True)


class TestGuard:
    @pytest.mark.parametrize("name", ["AGB32_final.CSV", "another_export.CSV", "export.csv"])
    def test_blocks_a_force_staged_simapro_export(self, repo, name):
        _stage(repo, f"source/{name}", force=True)

        result = _guard(repo)

        assert result.returncode == 1
        assert name in result.stderr

    def test_allows_the_stripped_archive_and_its_manifest(self, repo):
        _stage(repo, ARCHIVE, content=_archive("Wheat flour {FR}| production | Cut-off, U"))
        _stage(repo, "source/agribalyse-3.2-stripped-ecoinvent-processes.json")

        assert _guard(repo).returncode == 0

    @pytest.mark.parametrize(
        ("section", "name"),
        [
            ("Products", "Steel {RER}| production | Cut-off, S - Copied from Ecoinvent U"),
            ("Products", "Banana {CO}| banana production | Cut-off, U - Adapted from Ecoinvent U"),
            ("Waste treatment", "Waste {CH}| treatment | Cut-off, S - adapted from ecoinvent U"),
        ],
    )
    def test_blocks_an_archive_that_still_holds_an_embedded_ecoinvent_block(
        self, repo, section, name
    ):
        _stage(repo, ARCHIVE, content=_archive(name, section=section))

        result = _guard(repo)

        assert result.returncode == 1
        assert "AGB32_ecoinvent_free.CSV.xz" in result.stderr

    def test_an_input_line_naming_an_ecoinvent_dataset_is_not_a_block(self, repo):
        content = _archive(
            "Bread {FR}| baking | Cut-off, U",
            inputs="Wheat {FR}| market | Cut-off, U - Adapted from Ecoinvent U;kg;0,7;Undefined",
        )
        _stage(repo, ARCHIVE, content=content)

        assert _guard(repo).returncode == 0

    def test_blocks_an_archive_that_is_not_xz(self, repo):
        _stage(repo, ARCHIVE)

        assert _guard(repo).returncode != 0


class TestGitignore:
    @pytest.mark.parametrize(
        "name", ["AGB32_final.CSV", "another_export.CSV", "harmonised-flows-simple.json.gz"]
    )
    def test_local_only_inputs_are_ignored(self, repo, name):
        result = subprocess.run(["git", "check-ignore", "-q", f"source/{name}"], cwd=repo)

        assert result.returncode == 0

    @pytest.mark.parametrize(
        "path",
        [
            "dashboard/backtest_pass1.csv",
            "dashboard/run_report.json",
            "dashboard/cf_comparison.csv",
            "registry/registry.meta.json",
            "registry/mappings_biosphere.parquet",
            "to_review/mappings_comparison.xlsx",
            "unlinked/biosphere_unlinked.xlsx",
            "unlinked/technosphere_unlinked.json",
            "cache/bootstrap-brightway/projects.db",
            "source/ecoinvent-3.9.1-cutoff-exchanges.parquet",
            "source/ecoinvent-3.9.1-cutoff-activities.json",
            "source/ecoinvent-3.9.1-biosphere-flows.json",
            "source/ef-v31-methods.json",
            "bw_package/README.md",
        ],
    )
    def test_everything_a_run_writes_is_ignored(self, repo, path):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo)

        assert result.returncode == 0, f"{path} would dirty the tree after a run"

    @pytest.mark.parametrize(
        "path",
        [
            "dashboard/backtest_dashboard.html",
            "dashboard/outlier_reasons.json",
            "dashboard/.gitkeep",
            "to_review/.gitkeep",
            "unlinked/.gitkeep",
        ],
    )
    def test_shipped_dashboard_files_stay_tracked(self, repo, path):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo)

        assert result.returncode == 1

    @pytest.mark.parametrize(
        "name",
        ["AGB32_ecoinvent_free.CSV.xz", "agribalyse-3.2-stripped-ecoinvent-processes.json"],
    )
    def test_shipped_files_are_not_ignored(self, repo, name):
        result = subprocess.run(["git", "check-ignore", "-q", f"source/{name}"], cwd=repo)

        assert result.returncode == 1
