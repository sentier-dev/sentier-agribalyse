"""Unit tests for the standalone ``import_into_brightway`` script.

These exercise the pure-data reconstruction (no bw2data needed) plus config
resolution and the export-validation error paths. The dual-generation
round-trip through real bw2data lives in
``tests/integration/bw_import/test_import_integration.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from bw_import import import_into_brightway as imp

_IDX_DTYPE = np.dtype([("row", "<i8"), ("col", "<i8")])


def _indices(pairs: list[tuple[int, int]]) -> np.ndarray:
    arr = np.empty(len(pairs), dtype=_IDX_DTYPE)
    for i, (r, c) in enumerate(pairs):
        arr[i] = (r, c)
    return arr


# --------------------------------------------------------------------------- #
# The importer must never import from this package — it is copied verbatim
# into the export and runs standalone in the user's Brightway environment.
# --------------------------------------------------------------------------- #


def test_importer_imports_nothing_from_package():
    src = Path(imp.__file__).read_text()
    tree = ast.parse(src)
    banned = {
        "bw_export",
        "bw_import",
        "scoring",
        "cli",
        "config",
        "core",
        "domain",
        "pipelines",
        "reporting",
        "readers",
        "registry",
        "matching",
        "transforms",
        "exports",
        "llm",
        "ef",
        "utils",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned, alias.name
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in banned, node.module


# --------------------------------------------------------------------------- #
# .env parsing + config precedence
# --------------------------------------------------------------------------- #


def test_read_dotenv_handles_quotes_exports_comments(tmp_path):
    (tmp_path / ".env").write_text(
        "# comment\n"
        "\n"
        "BW_PROJECT=demo\n"
        'BRIGHTWAY2_DIR="/tmp/bw dir"\n'
        "export ECOINVENT_USERNAME='alice'\n"
        "NOEQUALS\n"
    )
    env = imp._read_dotenv(tmp_path / ".env")
    assert env == {
        "BW_PROJECT": "demo",
        "BRIGHTWAY2_DIR": "/tmp/bw dir",
        "ECOINVENT_USERNAME": "alice",
    }


def test_config_precedence_flag_over_env_over_dotenv_over_default(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("BW_PROJECT=from-dotenv\n")
    args = argparse.Namespace(project=None, brightway_dir=None)

    # .env only
    monkeypatch.delenv("BW_PROJECT", raising=False)
    project, _ = imp._resolve_config(args, tmp_path)
    assert project == "from-dotenv"

    # env beats .env
    monkeypatch.setenv("BW_PROJECT", "from-env")
    project, _ = imp._resolve_config(args, tmp_path)
    assert project == "from-env"

    # flag beats env
    args.project = "from-flag"
    project, _ = imp._resolve_config(args, tmp_path)
    assert project == "from-flag"


def test_config_default_project_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("BW_PROJECT", raising=False)
    args = argparse.Namespace(project=None, brightway_dir=None)
    project, bw_dir = imp._resolve_config(args, tmp_path)
    assert project == imp._DEFAULT_PROJECT
    assert bw_dir is None


# --------------------------------------------------------------------------- #
# Sign convention — the crux
# --------------------------------------------------------------------------- #


def test_build_activities_sign_convention(monkeypatch):
    # A = [[1, -0.5], [0, 1]] plus a positive off-diagonal substitution.
    # rows are product ids 101/102, cols are activity ids 201/202.
    activities = pd.DataFrame(
        {
            "col_id": [201, 202],
            "name": ["act one", "act two"],
            "unit": ["kg", "kg"],
            "location": ["GLO", "GLO"],
            "reference_product": ["p1", "p2"],
            "database": ["db", "db"],
            "code": ["a201", "a202"],
            "production_product_id": [101, 102],
        }
    )
    t_idx = _indices([(101, 201), (101, 202), (102, 202), (102, 201)])
    t_data = np.array([1.0, -0.5, 1.0, 0.4], dtype="float64")
    b_idx = _indices([(301, 201), (301, 202)])
    b_data = np.array([2.0, 3.0], dtype="float64")

    monkeypatch.setattr(imp, "_BIO_CODE", {301: "biosphere3::b301"})
    data = imp.build_activities_data(activities, t_idx, t_data, b_idx, b_data)

    key201 = (imp._ACTIVITIES_DB, "db::a201")
    key202 = (imp._ACTIVITIES_DB, "db::a202")
    bio_key = (imp._BIOSPHERE_DB, "biosphere3::b301")

    ex201 = data[key201]["exchanges"]
    ex202 = data[key202]["exchanges"]

    # 201: production of self (1.0) + biosphere 2.0
    assert {"input": key201, "amount": 1.0, "type": "production"} in ex201
    assert {"input": bio_key, "amount": 2.0, "type": "biosphere"} in ex201
    # 202: production of self (1.0), a technosphere input from 201 (-(-0.5)=0.5),
    # and biosphere 3.0
    assert {"input": key202, "amount": 1.0, "type": "production"} in ex202
    assert {"input": key201, "amount": 0.5, "type": "technosphere"} in ex202
    assert {"input": bio_key, "amount": 3.0, "type": "biosphere"} in ex202
    # positive off-diagonal (102 consumed by 201) becomes a negative-amount
    # (substitution) technosphere input from 202's producer.
    assert {"input": key202, "amount": -0.4, "type": "technosphere"} in ex201

    assert data[key201]["type"] == "process"
    assert data[key201]["reference product"] == "p1"


def test_build_activities_skips_zeros(monkeypatch):
    activities = pd.DataFrame(
        {
            "col_id": [201],
            "name": ["a"],
            "unit": ["kg"],
            "location": [""],
            "reference_product": [""],
            "database": ["db"],
            "code": ["a201"],
            "production_product_id": [101],
        }
    )
    t_idx = _indices([(101, 201), (102, 201)])
    t_data = np.array([1.0, 0.0], dtype="float64")  # second entry is a zero
    monkeypatch.setattr(imp, "_BIO_CODE", {})
    data = imp.build_activities_data(
        activities, t_idx, t_data, _indices([]), np.array([], dtype="float64")
    )
    assert len(data[(imp._ACTIVITIES_DB, "db::a201")]["exchanges"]) == 1


def test_build_biosphere_data_splits_categories():
    bio = pd.DataFrame(
        {
            "bioflow_id": [301, 999],
            "is_synthetic_correction": [False, True],
            "name": ["Water", "AWARE correction"],
            "categories": ["natural resource::in water", ""],
            "unit": ["m3", ""],
            "database": ["biosphere3", "agribalyse-ef31-correction"],
            "code": ["b301", "aware-x"],
        }
    )
    data = imp.build_biosphere_data(bio)
    water = data[(imp._BIOSPHERE_DB, "biosphere3::b301")]
    assert water["categories"] == ("natural resource", "in water")
    assert water["type"] == "emission"
    corr = data[(imp._BIOSPHERE_DB, "agribalyse-ef31-correction::aware-x")]
    assert corr["categories"] == ()


def test_build_methods_reads_characterization(tmp_path, monkeypatch):
    slug = "methods/demo"
    mdir = tmp_path / slug
    mdir.mkdir(parents=True)
    np.save(mdir / "characterization.indices.npy", _indices([(301, 301)]))
    np.save(mdir / "characterization.data.npy", np.array([10.0], dtype="float64"))
    (mdir / "datapackage.json").write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "matrix": "characterization_matrix",
                        "kind": "indices",
                        "path": "characterization.indices.npy",
                    },
                    {
                        "matrix": "characterization_matrix",
                        "kind": "data",
                        "path": "characterization.data.npy",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(imp, "_BIO_CODE", {301: "biosphere3::b301"})
    methods = imp.build_methods(tmp_path, [{"key": ["m", "n"], "slug": "demo", "path": slug}])
    assert methods == [(("m", "n"), [((imp._BIOSPHERE_DB, "biosphere3::b301"), 10.0)])]


# --------------------------------------------------------------------------- #
# Export validation + guards
# --------------------------------------------------------------------------- #


def test_load_activities_rejects_pre_refactor_export(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    # old-style: no production_product_id / database / code columns
    pd.DataFrame({"col_id": [1], "name": ["x"]}).to_parquet(meta / "activities.parquet")
    with pytest.raises(imp.ImportError_, match="before the importer refactor"):
        imp._load_activities(tmp_path)


def test_load_activities_missing_file(tmp_path):
    with pytest.raises(imp.ImportError_, match="missing"):
        imp._load_activities(tmp_path)


def test_guard_existing_databases_blocks_without_overwrite():
    class FakeBD:
        databases: ClassVar[dict] = {imp._ACTIVITIES_DB: {}}

    with pytest.raises(imp.ImportError_, match="--overwrite"):
        imp._guard_existing_databases(FakeBD(), overwrite=False)


def test_guard_existing_databases_deletes_with_overwrite():
    class FakeDatabases(dict):
        def __delitem__(self, k):
            super().__delitem__(k)

    class FakeBD:
        databases = FakeDatabases({imp._ACTIVITIES_DB: {}, imp._BIOSPHERE_DB: {}})

    bd = FakeBD()
    imp._guard_existing_databases(bd, overwrite=True)
    assert imp._ACTIVITIES_DB not in bd.databases
    assert imp._BIOSPHERE_DB not in bd.databases


def test_main_returns_2_on_missing_export(tmp_path):
    rc = imp.main(["--bundle-dir", str(tmp_path), "--no-verify"])
    assert rc == 2
