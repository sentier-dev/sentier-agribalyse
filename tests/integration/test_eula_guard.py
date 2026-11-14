"""The pre-commit guard and .gitignore around the two Agribalyse CSV variants."""

from __future__ import annotations

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


def _guard(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(repo / ".githooks" / "pre-commit")], cwd=repo, capture_output=True, text=True
    )


def _stage(repo: Path, relative: str, *, force: bool = False) -> None:
    path = repo / relative
    path.write_bytes(b"x")
    subprocess.run(["git", "add", *(["-f"] if force else []), relative], cwd=repo, check=True)


class TestGuard:
    @pytest.mark.parametrize("name", ["AGB32_final.CSV", "another_export.CSV", "export.csv"])
    def test_blocks_a_force_staged_simapro_export(self, repo, name):
        _stage(repo, f"source/{name}", force=True)

        result = _guard(repo)

        assert result.returncode == 1
        assert name in result.stderr

    def test_allows_the_stripped_archive_and_its_manifest(self, repo):
        _stage(repo, "source/AGB32_ecoinvent_free.CSV.xz")
        _stage(repo, "source/agribalyse-3.2-stripped-ecoinvent-processes.json")

        assert _guard(repo).returncode == 0


class TestGitignore:
    @pytest.mark.parametrize(
        "name", ["AGB32_final.CSV", "another_export.CSV", "harmonised-flows-simple.json.gz"]
    )
    def test_local_only_inputs_are_ignored(self, repo, name):
        result = subprocess.run(["git", "check-ignore", "-q", f"source/{name}"], cwd=repo)

        assert result.returncode == 0

    @pytest.mark.parametrize(
        "name",
        ["AGB32_ecoinvent_free.CSV.xz", "agribalyse-3.2-stripped-ecoinvent-processes.json"],
    )
    def test_shipped_files_are_not_ignored(self, repo, name):
        result = subprocess.run(["git", "check-ignore", "-q", f"source/{name}"], cwd=repo)

        assert result.returncode == 1
