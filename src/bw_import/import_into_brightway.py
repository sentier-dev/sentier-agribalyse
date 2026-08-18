"""Import the Agribalyse 3.2 x ecoinvent 3.9.1 x EF v3.1 export into Brightway.

Run this **inside the Python environment where Brightway (or Activity Browser)
is installed** — it writes into *your* Brightway data directory using *your*
``bw2data``. It works on both Brightway generations (legacy ``bw2data`` 3.x and
``bw2data`` 4.x / bw2.5); nothing here is pinned, and it imports nothing from the
``sentier-agribalyse`` package. Its only inputs are the exported files:

    metadata/activities.parquet     col_id -> (database, code, name, unit,
                                    location, reference_product,
                                    production_product_id)
    metadata/biosphere.parquet      bioflow_id -> (database, code, name,
                                    categories, unit, is_synthetic_correction)
    metadata/methods.json           method keys + datapackage paths
    metadata/parity_samples.json    products with expected scores (--verify)
    inventory/                      technosphere + biosphere bw_processing arrays
    methods/<slug>/                 one characterization datapackage per method

The technosphere is a square, diagonal-aligned matrix: the activity at column
index ``i`` produces the product at row index ``i`` (its reference product). The
shipped ``production_product_id`` records that pairing per column, which is what
lets us tell a **production** exchange (not sign-flipped by bw2data) from a
**technosphere** input (sign-flipped). We reproduce the exact signs the export's
backtested scores were computed with, then ``--verify`` re-scores a few products
to prove it.

Usage::

    python import_into_brightway.py --verify 3
    python import_into_brightway.py --project my-project --overwrite --no-verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_PROJECT = "agribalyse-ef31"
_ACTIVITIES_DB = "agribalyse-ef31"
_BIOSPHERE_DB = "agribalyse-ef31-biosphere"
_DEFAULT_TOLERANCE = 1e-6

_REQUIRED_ACTIVITY_COLUMNS = {
    "col_id",
    "name",
    "unit",
    "location",
    "reference_product",
    "database",
    "code",
    "production_product_id",
}
_REQUIRED_BIOSPHERE_COLUMNS = {
    "bioflow_id",
    "is_synthetic_correction",
    "name",
    "categories",
    "unit",
    "database",
    "code",
}
_CATEGORY_SEP = "::"


class ImportError_(RuntimeError):
    """A fatal, user-actionable importer error (message is printed verbatim)."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        _run(args)
    except ImportError_ as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    return 0


# --------------------------------------------------------------------------- #
# Argument + environment resolution (flags win over .env win over defaults)
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="import_into_brightway",
        description="Import the exported Agribalyse x ecoinvent x EF system into "
        "a bw2data project (works on legacy bw2data 3.x and 4.x).",
    )
    p.add_argument(
        "--project",
        default=None,
        help=f"Target bw2data project (default: $BW_PROJECT or '{_DEFAULT_PROJECT}').",
    )
    p.add_argument(
        "--brightway-dir",
        default=None,
        type=Path,
        help="Brightway data directory (default: $BRIGHTWAY2_DIR or bw2data's per-OS default).",
    )
    # Named --bundle-dir (not --export-dir) for CLI compatibility with the
    # dds-agribalyse-bundle importer; the export layout is format-identical.
    p.add_argument(
        "--bundle-dir",
        default=None,
        type=Path,
        help="Folder holding inventory/, methods/, metadata/ (default: this script's own folder).",
    )
    verify = p.add_mutually_exclusive_group()
    verify.add_argument(
        "--verify",
        type=int,
        default=3,
        metavar="N",
        help="Re-score N products against parity_samples.json (default: 3).",
    )
    verify.add_argument(
        "--no-verify", action="store_true", help="Skip the bw2calc parity round-trip."
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the databases if the project already has them.",
    )
    return p.parse_args(argv)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file with the standard library only (no python-dotenv).

    Supports ``KEY=value`` lines, ``#`` comments, blank lines, ``export KEY=``
    prefixes, and surrounding single/double quotes. Unknown syntax is ignored.
    """
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            env[key] = value
    return env


def _resolve_config(args: argparse.Namespace, bundle_dir: Path) -> tuple[str, Path | None]:
    """Return ``(project, brightway_dir)`` applying flag > env/.env > default."""
    dotenv = _read_dotenv(bundle_dir / ".env")

    def pick(flag_value, env_key):
        if flag_value is not None:
            return flag_value
        if os.environ.get(env_key):
            return os.environ[env_key]
        if dotenv.get(env_key):
            return dotenv[env_key]
        return None

    project = pick(args.project, "BW_PROJECT") or _DEFAULT_PROJECT
    bw_dir_raw = pick(str(args.brightway_dir) if args.brightway_dir else None, "BRIGHTWAY2_DIR")
    brightway_dir = Path(bw_dir_raw).expanduser() if bw_dir_raw else None
    return project, brightway_dir


# --------------------------------------------------------------------------- #
# Bundle loading (pure data — no bw2data yet)
# --------------------------------------------------------------------------- #


def _resolve_bundle_dir(args: argparse.Namespace) -> Path:
    return (args.bundle_dir or Path(__file__).resolve().parent).resolve()


def _load_activities(bundle_dir: Path) -> pd.DataFrame:
    path = bundle_dir / "metadata" / "activities.parquet"
    if not path.is_file():
        raise ImportError_(f"missing {path}. Is --bundle-dir pointing at the bundle?")
    df = pd.read_parquet(path)
    missing = _REQUIRED_ACTIVITY_COLUMNS - set(df.columns)
    if missing:
        raise ImportError_(
            f"{path} is missing columns {sorted(missing)} — this looks like an "
            "export built before the importer refactor. Rebuild it with "
            "dds-build-bw-package."
        )
    return df


def _load_biosphere(bundle_dir: Path) -> pd.DataFrame:
    path = bundle_dir / "metadata" / "biosphere.parquet"
    if not path.is_file():
        raise ImportError_(f"missing {path}. Is --bundle-dir pointing at the bundle?")
    df = pd.read_parquet(path)
    missing = _REQUIRED_BIOSPHERE_COLUMNS - set(df.columns)
    if missing:
        raise ImportError_(
            f"{path} is missing columns {sorted(missing)} — rebuild the export "
            "with a current dds-build-bw-package."
        )
    return df


def _load_methods_index(bundle_dir: Path) -> list[dict]:
    path = bundle_dir / "metadata" / "methods.json"
    if not path.is_file():
        raise ImportError_(f"missing {path}.")
    return json.loads(path.read_text())["methods"]


def _read_vector(dp_dir: Path, matrix: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one bw_processing persistent vector: ``(indices, data)`` arrays.

    Resolves file names from ``datapackage.json`` (directory filesystem), so it
    does not assume the ``<name>.indices.npy`` convention.
    """
    dp = json.loads((dp_dir / "datapackage.json").read_text())
    idx_path = data_path = None
    for res in dp["resources"]:
        if res.get("matrix") != matrix:
            continue
        if res.get("kind") == "indices":
            idx_path = res["path"]
        elif res.get("kind") == "data":
            data_path = res["path"]
    if idx_path is None or data_path is None:
        raise ImportError_(f"datapackage in {dp_dir} has no '{matrix}' vector.")
    indices = np.load(dp_dir / idx_path)
    data = np.load(dp_dir / data_path)
    return indices, data


# --------------------------------------------------------------------------- #
# Reconstruction (pure data -> bw2data-shaped dicts; testable without bw2data)
# --------------------------------------------------------------------------- #


def _safe_code(database: str, code: str) -> str:
    return f"{database}::{code}"


def build_biosphere_data(biosphere: pd.DataFrame) -> dict:
    """``biosphere.parquet`` -> ``{key: node_dict}`` for ``Database.write``."""
    data = {}
    for row in biosphere.itertuples(index=False):
        cats = str(getattr(row, "categories", "") or "")
        categories = tuple(c for c in cats.split(_CATEGORY_SEP) if c)
        key = (_BIOSPHERE_DB, _safe_code(str(row.database), str(row.code)))
        data[key] = {
            "name": str(row.name),
            "categories": categories,
            "unit": str(getattr(row, "unit", "") or ""),
            "type": "emission",
            "exchanges": [],
        }
    return data


def build_activities_data(
    activities: pd.DataFrame,
    technosphere_indices: np.ndarray,
    technosphere_data: np.ndarray,
    biosphere_indices: np.ndarray,
    biosphere_data: np.ndarray,
) -> dict:
    """Reconstruct the activities database (with exchanges) from the datapackages.

    Sign convention, matching the bundle's published scores exactly:

    * the production exchange (row product == the column's
      ``production_product_id``) keeps its stored amount (bw2data does not flip
      production);
    * every other technosphere entry is a technosphere input whose amount is
      **negated** (bw2data re-applies the technosphere sign flip, so negating
      here restores the stored matrix value — a positive off-diagonal becomes a
      negative amount, i.e. a substitution);
    * biosphere entries keep their stored amount (no sign flip).
    """
    key_by_col: dict[int, tuple[str, str]] = {}
    producer_by_product: dict[int, int] = {}
    for row in activities.itertuples(index=False):
        col_id = int(row.col_id)
        key_by_col[col_id] = (_ACTIVITIES_DB, _safe_code(str(row.database), str(row.code)))
        producer_by_product[int(row.production_product_id)] = col_id

    exchanges: dict[int, list[dict]] = {col_id: [] for col_id in key_by_col}

    t_rows = technosphere_indices["row"]
    t_cols = technosphere_indices["col"]
    for pid, aid, value in zip(t_rows, t_cols, technosphere_data, strict=True):
        value = float(value)
        if value == 0.0:
            continue
        col_id = int(aid)
        product_id = int(pid)
        self_key = key_by_col[col_id]
        if producer_by_product.get(product_id) == col_id:
            exchanges[col_id].append({"input": self_key, "amount": value, "type": "production"})
        else:
            producer_col = producer_by_product.get(product_id)
            if producer_col is None:
                raise ImportError_(
                    f"technosphere references product {product_id} with no "
                    "producing activity — bundle metadata is inconsistent."
                )
            exchanges[col_id].append(
                {"input": key_by_col[producer_col], "amount": -value, "type": "technosphere"}
            )

    b_rows = biosphere_indices["row"]
    b_cols = biosphere_indices["col"]
    for fid, aid, value in zip(b_rows, b_cols, biosphere_data, strict=True):
        value = float(value)
        if value == 0.0:
            continue
        col_id = int(aid)
        exchanges[col_id].append(
            {
                "input": (_BIOSPHERE_DB, _bio_code_lookup(fid)),
                "amount": value,
                "type": "biosphere",
            }
        )

    data = {}
    for row in activities.itertuples(index=False):
        col_id = int(row.col_id)
        data[key_by_col[col_id]] = {
            "name": str(row.name),
            "unit": str(getattr(row, "unit", "") or ""),
            "location": str(getattr(row, "location", "") or ""),
            "reference product": str(getattr(row, "reference_product", "") or ""),
            "type": "process",
            "exchanges": exchanges[col_id],
        }
    return data


# Biosphere consolidated-code lookup is closed over per-run (set by _run). Kept
# module-level so build_activities_data stays a pure function of its arguments
# in tests, where the fixture installs a lookup covering its flows.
_BIO_CODE: dict[int, str] = {}


def _bio_code_lookup(bioflow_id) -> str:
    code = _BIO_CODE.get(int(bioflow_id))
    if code is None:
        raise ImportError_(
            f"biosphere flow {int(bioflow_id)} used in inventory but absent from "
            "biosphere.parquet — bundle metadata is inconsistent."
        )
    return code


def build_methods(
    bundle_dir: Path, methods_index: list[dict]
) -> list[tuple[tuple[str, ...], list[tuple[tuple[str, str], float]]]]:
    """Read each method's characterization datapackage into ``(name, cfs)``."""
    out = []
    for entry in methods_index:
        name = tuple(entry["key"])
        dp_dir = bundle_dir / entry["path"]
        indices, data = _read_vector(dp_dir, "characterization_matrix")
        cfs = [
            ((_BIOSPHERE_DB, _bio_code_lookup(fid)), float(cf))
            for fid, cf in zip(indices["row"], data, strict=True)
        ]
        out.append((name, cfs))
    return out


# --------------------------------------------------------------------------- #
# The run (the only part that touches bw2data)
# --------------------------------------------------------------------------- #


def _run(args: argparse.Namespace) -> None:
    bundle_dir = _resolve_bundle_dir(args)
    activities = _load_activities(bundle_dir)
    biosphere = _load_biosphere(bundle_dir)
    methods_index = _load_methods_index(bundle_dir)

    # Populate the biosphere consolidated-code lookup before reconstructing
    # inventory/methods (both resolve biosphere ids to consolidated codes).
    global _BIO_CODE
    _BIO_CODE = {
        int(r.bioflow_id): _safe_code(str(r.database), str(r.code))
        for r in biosphere.itertuples(index=False)
    }

    project, brightway_dir = _resolve_config(args, bundle_dir)

    # BRIGHTWAY2_DIR must be set before bw2data is imported.
    if brightway_dir is not None:
        brightway_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BRIGHTWAY2_DIR"] = str(brightway_dir)

    try:
        import bw2data as bd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError_(
            "could not import bw2data. Run this script from the Python "
            "environment where Activity Browser / Brightway is installed "
            f"(underlying error: {exc})."
        ) from exc

    bd.projects.set_current(project)
    # bw2data 4.x exposes projects.dir as a Path; 3.x as a str. Show the parent
    # (the Brightway data root) when we can, else just the project dir.
    proj_dir = Path(str(bd.projects.dir))
    print(f"Project: {project!r}  (Brightway data dir: {proj_dir.parent})")

    _guard_existing_databases(bd, args.overwrite)

    print(f"Writing biosphere database {_BIOSPHERE_DB!r} ({len(biosphere)} flows)...")
    bio_data = build_biosphere_data(biosphere)
    bd.Database(_BIOSPHERE_DB).write(bio_data)

    print(f"Reconstructing inventory ({len(activities)} activities)...")
    t_idx, t_data = _read_vector(bundle_dir / "inventory", "technosphere_matrix")
    b_idx, b_data = _read_vector(bundle_dir / "inventory", "biosphere_matrix")
    act_data = build_activities_data(activities, t_idx, t_data, b_idx, b_data)
    print(f"Writing activities database {_ACTIVITIES_DB!r}...")
    bd.Database(_ACTIVITIES_DB).write(act_data)

    print(f"Writing {len(methods_index)} LCIA methods...")
    for name, cfs in build_methods(bundle_dir, methods_index):
        method = bd.Method(name)
        if not method.registered:
            method.register()
        method.write(cfs)

    if args.no_verify or args.verify <= 0:
        print("Done. (parity check skipped)")
        return
    _verify(bd, bundle_dir, activities, args.verify)


def _guard_existing_databases(bd, overwrite: bool) -> None:
    existing = [db for db in (_ACTIVITIES_DB, _BIOSPHERE_DB) if db in bd.databases]
    if not existing:
        return
    if not overwrite:
        raise ImportError_(
            f"project already has database(s) {existing}. Re-run with "
            "--overwrite to replace them, or --project NAME to import "
            "into a different project."
        )
    for db in existing:
        print(f"Overwriting existing database {db!r}...")
        del bd.databases[db]


def _get_node(bd, database: str, code: str):
    """Resolve a ``(database, code)`` node across bw2data generations."""
    get_node = getattr(bd, "get_node", None)
    if get_node is not None:  # bw2data 4.x
        return get_node(database=database, code=code)
    return bd.Database(database).get(code)  # bw2data 3.x


def _score(bd, activity, method_name) -> float:
    import bw2calc as bc

    prepare = getattr(bd, "prepare_lca_inputs", None)
    if prepare is not None:  # bw2data 4.x / bw2calc 2.x
        fu, data_objs, _ = prepare({activity: 1.0}, method=method_name)
        lca = bc.LCA(fu, data_objs=data_objs)
    else:  # legacy bw2calc 1.x
        lca = bc.LCA({activity: 1.0}, method=method_name)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def _verify(bd, bundle_dir: Path, activities: pd.DataFrame, n: int) -> None:
    path = bundle_dir / "metadata" / "parity_samples.json"
    if not path.is_file():
        print("No parity_samples.json in bundle — skipping verification.")
        return
    payload = json.loads(path.read_text())
    tolerance = float(payload.get("tolerance", _DEFAULT_TOLERANCE))
    samples = payload.get("samples", [])[: max(0, n)]
    if not samples:
        print("No parity samples to check.")
        return

    key_by_col = {
        int(r.col_id): (_ACTIVITIES_DB, _safe_code(str(r.database), str(r.code)))
        for r in activities.itertuples(index=False)
    }
    max_rel = 0.0
    failures = []
    n_checked = 0
    for sample in samples:
        col_id = int(sample["activity_col_id"])
        _, code = key_by_col[col_id]
        activity = _get_node(bd, _ACTIVITIES_DB, code)
        for method_key, expected in sample["expected"]:
            got = _score(bd, activity, tuple(method_key))
            denom = abs(expected) if expected else 1.0
            rel = abs(got - expected) / denom
            max_rel = max(max_rel, rel)
            n_checked += 1
            if rel > tolerance:
                failures.append((tuple(method_key), expected, got, rel))

    if failures:
        lines = "\n".join(
            f"    {m}: expected {e:.6g}, got {g:.6g} (rel {r:.2e})" for m, e, g, r in failures[:10]
        )
        raise ImportError_(
            f"parity FAILED: {len(failures)}/{n_checked} scores drifted beyond "
            f"tolerance {tolerance:.1e} (max rel {max_rel:.2e}):\n{lines}"
        )
    print(
        f"Parity OK: {n_checked} score(s) within tolerance {tolerance:.1e} "
        f"(max rel error {max_rel:.2e})."
    )


if __name__ == "__main__":
    sys.exit(main())
