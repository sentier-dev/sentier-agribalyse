"""The ecoinvent EULA pre-commit guard (``.githooks/pre-commit``) and ``.gitignore``.

Each test builds a throwaway git repository, stages paths, runs the hook and
checks that it refuses or allows the commit. Covers every path class the hook
names, the paths it must leave alone (the ODbL elementary-flow snapshot, the
stripped Agribalyse archive, code whose file name contains ``bw_package``),
renames, non-ASCII names, deletions and the first commit of a repository.
The last class checks ``.gitignore`` around the two Agribalyse CSV variants.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".githooks" / "pre-commit"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("git") is None or shutil.which("bash") is None,
        reason="the guard test needs git and bash",
    ),
]

# Git with no user or system config, so core.quotePath and friends are at
# their defaults whatever the machine running the tests has set. Inherited
# GIT_* variables are dropped: git exports GIT_DIR and GIT_INDEX_FILE to a
# hook, and a test started from one would then write to the real repository.
_ENV = {
    **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "guard-test",
    "GIT_AUTHOR_EMAIL": "guard-test@example.invalid",
    "GIT_COMMITTER_NAME": "guard-test",
    "GIT_COMMITTER_EMAIL": "guard-test@example.invalid",
}

REFUSED = [
    # Licensed snapshots. A suffixed copy holds the same data.
    "source/ecoinvent-3.9.1-cutoff-exchanges.parquet",
    "source/ecoinvent-3.9.1-cutoff-exchanges.parquet.bak",
    "source/ecoinvent-3.9.1-cutoff-activities.json",
    "source/ecoinvent-3.9.1-cutoff-activities.json.bak",
    "source/ef-v31-methods.json",
    "source/ef-v31-methods.json.orig",
    # A SimaPro export: ADEME's full one embeds ecoinvent inventories. Refused
    # under source/ at any depth, compressed or suffixed, and by its own name
    # anywhere.
    "source/AGB32_final.CSV",
    "source/another_export.CSV",
    "source/export.csv",
    "source/AGB32_final.CSV.xz",
    "source/AGB32_final.CSV.gz",
    "source/another_export.csv.bak",
    "source/another_export.Csv",
    "source/another_export.CSV~",
    "source/sub/another_export.csv",
    "source/sub/AGB32_final.CSV",
    "source/AGB32_final.zip",
    "AGB32_final.CSV",
    "notes/AGB32_final.CSV.xz",
    "notes/agb32_final.zip",
    # The stripped archive is excepted by its exact path only.
    "source/AGB32_ecoinvent_free.CSV.xz.bak",
    "source/sub/AGB32_ecoinvent_free.CSV.xz",
    # Registry outputs with CFs or the dataset list.
    "registry/method_cfs/ecoinvent-3.9.1__EF%20v3.1__acidification__x/cfs.parquet",
    "registry/method_cfs/another-slug/regional_cfs.parquet",
    "registry/cf_comparison_join.parquet",
    "registry/ecoinvent_catalog.parquet",
    "registry/activity_catalog.parquet",
    # Caches.
    "cache/scoring_packages/abc123/technosphere.csr.npz",
    "cache/linked_cache.pkl",
    "cache/importer_cache.pkl",
    # Dashboard and review outputs with registry CFs.
    "dashboard/cf_comparison.csv",
    "dashboard/decomp/1000.json",
    "dashboard/product_reasons.json",
    "to_review/cf_comparison/acidification.parquet",
    # Brightway export and scoring-package matrices, wherever they land.
    "bw_package/inventory/datapackage.json",
    "export/bw_package/inventory/datapackage.json",
    "bw_package.zip",
    "out/bw_package.tar.gz",
    "agb-export/inventory/technosphere.data.npy",
    "agb-export/inventory/technosphere.flip.npy",
    "agb-export/inventory/biosphere.indices.npy",
    "agb-export/methods/x/characterization.data.npy",
    "elsewhere/abc123/biosphere.csr.npz",
    # Non-ASCII names, which git quotes in its default output.
    "bw_package/métadonnées.json",
    "cache/résultats.pkl",
]

ALLOWED = [
    # Elementary-flow lists: ODbL Data Glossary content.
    "source/ecoinvent-3.9.1-biosphere-flows.json",
    "source/biosphere3-flows.json",
    # The Agribalyse export that ships: the stripped archive and its manifest.
    "source/AGB32_ecoinvent_free.CSV.xz",
    "source/agribalyse-3.2-stripped-ecoinvent-processes.json",
    # Registry files without ecoinvent amounts or ecoinvent-LCIA CFs.
    "registry/method_cfs/_index.json",
    "registry/cf_comparison_by_code.parquet",
    "registry/mappings_biosphere.parquet",
    "registry/registry.meta.json",
    # Tracked dashboard files and product-level backtest results.
    "dashboard/backtest_pass1.csv",
    "dashboard/backtest_dashboard.html",
    "dashboard/outlier_reasons.json",
    "dashboard/backtest/scores.parquet",
    # Code and prose whose name contains bw_package, or is bw_package.
    "src/cli/build_bw_package.py",
    "tests/unit/bw_export/test_build_bw_package_cli.py",
    "src/bw_export/bw_package.py",
    "site/bw_package.md",
    # A CSV that is not under source/.
    "tests/fixtures/export.csv",
    "README.md",
    "notes/données.md",
]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, env=_ENV, check=True, capture_output=True, text=True
    )


def _stage(repo: Path, rel: str, *, force: bool = True) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"placeholder for {rel}\n", encoding="utf-8")
    _git(repo, "add", *(["-f"] if force else []), "--", rel)


def _run_hook(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HOOK)],
        cwd=repo,
        env=_ENV,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "init")
    return tmp_path


@pytest.fixture
def ignoring_repo(repo: Path) -> Path:
    """A repository that carries this repository's ``.gitignore``."""
    shutil.copy(REPO_ROOT / ".gitignore", repo / ".gitignore")
    return repo


@pytest.mark.parametrize("path", REFUSED)
def test_refuses_licensed_path(repo: Path, path: str) -> None:
    _stage(repo, path)
    result = _run_hook(repo)
    assert result.returncode == 1, result.stderr
    assert path in result.stderr


@pytest.mark.parametrize("path", ALLOWED)
def test_allows_path(repo: Path, path: str) -> None:
    _stage(repo, path)
    result = _run_hook(repo)
    assert result.returncode == 0, result.stderr


def test_refuses_a_rename_onto_a_licensed_path(repo: Path) -> None:
    _stage(repo, "notes/methods.json")
    _git(repo, "commit", "-q", "--no-verify", "-m", "add notes")
    (repo / "source").mkdir()
    _git(repo, "mv", "notes/methods.json", "source/ef-v31-methods.json")
    # Make sure git reports a rename, not an add, so this exercises R.
    status = _git(repo, "diff", "--cached", "--name-status").stdout
    assert status.startswith("R"), status
    result = _run_hook(repo)
    assert result.returncode == 1, result.stderr
    assert "source/ef-v31-methods.json" in result.stderr


def test_refuses_a_modified_licensed_file(repo: Path) -> None:
    _stage(repo, "cache/linked_cache.pkl")
    _git(repo, "commit", "-q", "--no-verify", "-m", "slipped through")
    (repo / "cache" / "linked_cache.pkl").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "-f", "--", "cache/linked_cache.pkl")
    status = _git(repo, "diff", "--cached", "--name-status").stdout
    assert status.startswith("M"), status
    result = _run_hook(repo)
    assert result.returncode == 1, result.stderr
    assert "cache/linked_cache.pkl" in result.stderr


def test_refuses_a_licensed_path_that_changes_type(repo: Path) -> None:
    _stage(repo, "cache/linked_cache.pkl")
    _git(repo, "commit", "-q", "--no-verify", "-m", "slipped through")
    path = repo / "cache" / "linked_cache.pkl"
    path.unlink()
    path.symlink_to("elsewhere.pkl")
    _git(repo, "add", "-f", "--", "cache/linked_cache.pkl")
    status = _git(repo, "diff", "--cached", "--name-status").stdout
    assert status.startswith("T"), status
    result = _run_hook(repo)
    assert result.returncode == 1, result.stderr
    assert "cache/linked_cache.pkl" in result.stderr


def test_refuses_a_path_that_is_not_utf8_whatever_the_locale(repo: Path) -> None:
    # A Latin-1 name cannot be created on every file system, so the index
    # entry is written directly. Under a UTF-8 locale a regex match on such
    # bytes fails; the hook therefore matches bytes (LC_ALL=C).
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        env=_ENV,
        input=b"x",
        check=True,
        capture_output=True,
    ).stdout.strip()
    subprocess.run(
        [
            b"git",
            b"update-index",
            b"--add",
            b"--cacheinfo",
            b"100644," + blob + b",cache/r\xe9sultats.pkl",
        ],
        cwd=repo,
        env=_ENV,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["bash", str(HOOK)],
        cwd=repo,
        env={**_ENV, "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"},
        capture_output=True,
    )
    assert result.returncode == 1, result.stderr
    assert b"cache/r\xe9sultats.pkl" in result.stderr


def test_refuses_no_file_that_is_in_git(repo: Path) -> None:
    """Every tracked path of this repository passes the guard, so the guard
    never blocks an edit of a file that is meant to be in git."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, env=_ENV, check=True, capture_output=True
    )
    tracked = [os.fsdecode(name) for name in listed.stdout.split(b"\0") if name]
    assert "README.md" in tracked
    for rel in tracked:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    _git(repo, "add", "-f", "--", ".")
    result = _run_hook(repo)
    assert result.returncode == 0, result.stderr


def test_allows_deleting_a_licensed_file(repo: Path) -> None:
    _stage(repo, "bw_package/inventory/datapackage.json")
    _git(repo, "commit", "-q", "--no-verify", "-m", "slipped through")
    _git(repo, "rm", "-q", "bw_package/inventory/datapackage.json")
    result = _run_hook(repo)
    assert result.returncode == 0, result.stderr


def test_allows_an_empty_index(repo: Path) -> None:
    result = _run_hook(repo)
    assert result.returncode == 0, result.stderr


def test_refuses_on_the_first_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _stage(tmp_path, "cache/linked_cache.pkl")
    result = _run_hook(tmp_path)
    assert result.returncode == 1, result.stderr


def test_the_stripped_archive_does_not_end_the_check(repo: Path) -> None:
    # Paths are checked in git's order: the archive sorts before the two
    # refused paths below, and the check must go on after it.
    _stage(repo, "source/AGB32_ecoinvent_free.CSV.xz")
    _stage(repo, "source/ef-v31-methods.json")
    _stage(repo, "tests/x.csr.npz")
    result = _run_hook(repo)
    assert result.returncode == 1, result.stderr
    assert "source/ef-v31-methods.json" in result.stderr
    assert "tests/x.csr.npz" in result.stderr
    assert "AGB32_ecoinvent_free" not in result.stderr


def test_lists_only_the_licensed_paths(repo: Path) -> None:
    _stage(repo, "README.md")
    _stage(repo, "dashboard/cf_comparison.csv")
    result = _run_hook(repo)
    assert result.returncode == 1, result.stderr
    assert "dashboard/cf_comparison.csv" in result.stderr
    assert "README.md" not in result.stderr


def test_git_commit_runs_the_guard(repo: Path) -> None:
    _stage(repo, "export/bw_package/inventory/datapackage.json")
    result = subprocess.run(
        ["git", "-c", f"core.hooksPath={HOOK.parent}", "commit", "-q", "-m", "export"],
        cwd=repo,
        env=_ENV,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode != 0, result.stderr
    assert "ecoinvent EULA guard" in result.stderr


def test_the_stripped_archive_and_its_manifest_need_no_force(ignoring_repo: Path) -> None:
    _stage(ignoring_repo, "source/AGB32_ecoinvent_free.CSV.xz", force=False)
    _stage(ignoring_repo, "source/agribalyse-3.2-stripped-ecoinvent-processes.json", force=False)
    result = _run_hook(ignoring_repo)
    assert result.returncode == 0, result.stderr


class TestGitignore:
    @pytest.mark.parametrize(
        "path",
        [
            "source/AGB32_final.CSV.xz",
            "source/another_export.csv.bak",
            "source/export.csv",
            "source/sub/another_export.csv",
            "source/AGB32_final.zip",
            "AGB32_final.CSV",
            "notes/agb32_final.zip",
            "source/ecoinvent-3.9.1-cutoff-exchanges.parquet",
            "source/ecoinvent-3.9.1-cutoff-activities.json",
            "source/ef-v31-methods.json",
            "registry/method_cfs/any-slug/cfs.parquet",
            "registry/activity_catalog.parquet",
            "cache/linked_cache.pkl",
            "dashboard/cf_comparison.csv",
            "to_review/cf_comparison/acidification.parquet",
            "bw_package/inventory/datapackage.json",
        ],
    )
    def test_licensed_outputs_are_ignored(self, ignoring_repo: Path, path: str) -> None:
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ignoring_repo, env=_ENV)

        assert result.returncode == 0

    @pytest.mark.parametrize(
        "name", ["AGB32_final.CSV", "another_export.CSV", "harmonised-flows-simple.json.gz"]
    )
    def test_local_only_inputs_are_ignored(self, ignoring_repo: Path, name: str) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "-q", f"source/{name}"], cwd=ignoring_repo, env=_ENV
        )

        assert result.returncode == 0

    @pytest.mark.parametrize(
        "name",
        ["AGB32_ecoinvent_free.CSV.xz", "agribalyse-3.2-stripped-ecoinvent-processes.json"],
    )
    def test_shipped_files_are_not_ignored(self, ignoring_repo: Path, name: str) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "-q", f"source/{name}"], cwd=ignoring_repo, env=_ENV
        )

        assert result.returncode == 1
