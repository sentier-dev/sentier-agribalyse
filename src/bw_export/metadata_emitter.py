"""``MetadataEmitter`` — importer-sufficient metadata + manifest + examples.

Writes ``metadata/{activities,products,biosphere}.parquet``,
``metadata/{methods,parity_samples}.json``, ``metadata/manifest.json``, and the
self-contained ``run_example.py`` + ``README.md`` into the output root.

The activities and biosphere tables are **load-bearing** for
``import_into_brightway.py``: every technosphere column is fully resolved to
``(database, code, name, unit, location, reference_product)`` plus the
``production_product_id`` (the reference-product row that pins its production
exchange), and every biosphere row carries its ``(database, code)`` key. The
emit fails loudly if any column cannot be named, because a nameless column would
produce an unusable Brightway database rather than a silently degraded one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bw_export.bw_node_types import ActivityMeta, BioMeta
from bw_export.correction_embedder import EmbeddedInventory
from bw_export.datapackage_writer import WriteResult
from scoring.method_slug import MethodSlug

# Resolver callables: integer id -> resolved metadata. Supplied by the caller
# (``CatalogKeyResolver`` in production, plain lambdas in tests) so the emitter
# stays free of catalog/pandas concerns.
ActivityResolver = Callable[[int], ActivityMeta]
BioResolver = Callable[[int], BioMeta]

_CATEGORY_SEP = "::"

_RUN_EXAMPLE = '''\
"""Compute one LCIA score from the exported datapackages.

Requires only: pip install bw_processing bw2calc pandas
Run from inside this folder: python run_example.py
"""
from pathlib import Path
import json

import pandas as pd
import bw_processing as bwp
import bw2calc as bc

HERE = Path(__file__).parent
methods = json.loads((HERE / "metadata" / "methods.json").read_text())
products = pd.read_parquet(HERE / "metadata" / "products.parquet")

# Pick the first product and the first method as a demo.
row = products.iloc[0]
product_id = int(row["product_id"])
method = methods["methods"][0]

inv = bwp.load_datapackage(bwp.generic_directory_filesystem(dirpath=HERE / "inventory"))
mdp = bwp.load_datapackage(bwp.generic_directory_filesystem(dirpath=HERE / method["path"]))
lca = bc.LCA({product_id: 1.0}, data_objs=[inv, mdp])
lca.lci()
lca.lcia()
print(f"{row.get('name')!r} | {method['key']} = {lca.score:.6g}")
'''

_README = """\
# Brightway export

The Agribalyse 3.2 x ecoinvent 3.9.1 x EF v3.1 system you built locally with
sentier-agribalyse (`dds-build-bw-package`), in two ready-to-use shapes:

1. **`bw_processing` datapackages** (`inventory/`, `methods/`) — score directly
   with stock `bw2calc`, no import step, no project.
2. **A one-shot importer** (`import_into_brightway.py`) — build a named
   `bw2data` project + database you can open in Activity Browser.

## A. Score directly with bw2calc

    pip install bw_processing bw2calc pandas
    python run_example.py

Or in your own code:
```python
import bw_processing as bwp
import bw2calc as bc

inv = bwp.load_datapackage(bwp.generic_directory_filesystem(dirpath="inventory"))
method = bwp.load_datapackage(bwp.generic_directory_filesystem(dirpath="methods/<slug>"))
lca = bc.LCA({product_id: 1.0}, data_objs=[inv, method])
lca.lci(); lca.lcia()
print(lca.score)
```

The demand is keyed by the **product id** directly — look it up in
`metadata/products.parquet` (column `product_id`); `metadata/methods.json` lists
method keys and datapackage paths. The AWARE water-use correction is embedded as
a synthetic biosphere flow (flagged in `metadata/biosphere.parquet`), so scores
match the pipeline's backtested numbers exactly. `pip install pypardiso` is
recommended: the technosphere carries zero-diagonal placeholder activities that
scipy's default SuperLU factorization rejects as singular.

## B. Import into Brightway / Activity Browser

Run the importer **inside the same Python environment where Brightway (or
Activity Browser) is installed** — it writes into *your* Brightway data
directory, so it needs *your* `bw2data`. It works on both Brightway generations
(legacy `bw2data` 3.x and `bw2data` 4.x / bw2.5); nothing here is pinned.

    # activate the env that has Activity Browser / Brightway, then:
    python import_into_brightway.py --verify 3

This creates a project (default `agribalyse-ef31`) with two databases — the
activities database and its biosphere — and the 19 EF v3.1 methods, then
re-scores a few products with your own `bw2calc` to confirm parity. Open
Activity Browser and select the `agribalyse-ef31` project.

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--project NAME` | `agribalyse-ef31` (or `$BW_PROJECT`) | Target bw2data project. |
| `--brightway-dir PATH` | bw2data default (or `$BRIGHTWAY2_DIR`) | Where the project is stored. |
| `--verify N` / `--no-verify` | `--verify 3` | Re-score N products against `metadata/parity_samples.json`. |
| `--overwrite` | off | Replace the databases if the project already has them. |
| `--bundle-dir PATH` | the importer's own folder | Location of `inventory/`, `methods/`, `metadata/`. |

Flags win over the environment; the environment (or a `.env` file next to the
importer) wins over the defaults. `.env` keys: `BRIGHTWAY2_DIR`, `BW_PROJECT`.

## Licence

This export contains ecoinvent 3.9.1 numerical values, composed locally from
data you downloaded under **your own ecoinvent licence**. It is for your use
under that licence and **must not be redistributed**. See the
sentier-agribalyse `BOOTSTRAP.md` and the ecoinvent EULA.
"""


@dataclass(frozen=True)
class MetadataEmitter:
    def emit(
        self,
        *,
        out_root: Path,
        embedded: EmbeddedInventory,
        written: WriteResult,
        activity_resolver: ActivityResolver,
        bio_resolver: BioResolver,
        product_catalog: pd.DataFrame,
        parity_scores: dict[int, dict[tuple[str, ...], float]],
        parity_tolerance: float,
        manifest_extra: dict[str, Any],
        parity: dict[str, Any],
    ) -> None:
        out_root = Path(out_root)
        meta = out_root / "metadata"
        meta.mkdir(parents=True, exist_ok=True)

        col_idx_to_id = self._invert(embedded.technosphere_col_id_to_idx)
        row_idx_to_id = self._invert(embedded.technosphere_row_id_to_idx)
        # Matrix index i aligns product-row i with activity-column i: the
        # activity at column i produces the product at row i (its reference
        # product). That pairing is the only reliable way to tell a production
        # exchange from a technosphere input at import time, so ship it.
        production_for_col = {col_idx_to_id[i]: row_idx_to_id[i] for i in col_idx_to_id}

        self._emit_products(meta, written, product_catalog)
        self._emit_activities(meta, col_idx_to_id, production_for_col, activity_resolver)
        self._emit_biosphere(meta, embedded, bio_resolver)
        methods_json = self._emit_methods(written, out_root)
        self._emit_parity_samples(meta, production_for_col, parity_scores, parity_tolerance)

        manifest = {
            **manifest_extra,
            "counts": {
                "n_activities": int(embedded.technosphere.shape[1]),
                "n_products": int(embedded.technosphere.shape[0]),
                "n_biosphere_flows": int(embedded.biosphere.shape[0]),
                "n_synthetic_correction_flows": len(embedded.synthetic_flow_ids),
                "n_methods": len(written.method_paths),
            },
            "parity": parity,
        }
        (meta / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (meta / "methods.json").write_text(json.dumps(methods_json, indent=2))

        (out_root / "run_example.py").write_text(_RUN_EXAMPLE)
        (out_root / "README.md").write_text(_README)

    def _emit_products(
        self, meta: Path, written: WriteResult, product_catalog: pd.DataFrame
    ) -> None:
        df = pd.DataFrame({"product_id": [int(p) for p in written.product_ids]})
        cols = [
            c
            for c in ("product_id", "name", "code", "database", "type", "unit")
            if c in product_catalog.columns
        ]
        cat = product_catalog[cols].copy()
        cat["product_id"] = cat["product_id"].astype("int64")
        df = df.merge(cat, on="product_id", how="left")
        df.sort_values("product_id").to_parquet(meta / "products.parquet", index=False)

    def _emit_activities(
        self,
        meta: Path,
        col_idx_to_id: dict[int, int],
        production_for_col: dict[int, int],
        activity_resolver: ActivityResolver,
    ) -> None:
        rows = []
        for col_id in sorted(col_idx_to_id.values()):
            m = activity_resolver(col_id)
            rows.append(
                {
                    "col_id": int(col_id),
                    "name": m.name,
                    "unit": m.unit,
                    "location": m.location,
                    "reference_product": m.reference_product,
                    "database": m.key[0],
                    "code": m.key[1],
                    "production_product_id": int(production_for_col[col_id]),
                }
            )
        df = pd.DataFrame(rows).sort_values("col_id").reset_index(drop=True)
        self._assert_fully_named(df, ("name", "database", "code"), "activities")
        df.to_parquet(meta / "activities.parquet", index=False)

    def _emit_biosphere(
        self,
        meta: Path,
        embedded: EmbeddedInventory,
        bio_resolver: BioResolver,
    ) -> None:
        rows = []
        for bid in sorted(embedded.biosphere_row_id_to_idx):
            m = bio_resolver(bid)
            rows.append(
                {
                    "bioflow_id": int(bid),
                    "is_synthetic_correction": bool(m.is_synthetic_correction),
                    "name": m.name,
                    "categories": _CATEGORY_SEP.join(m.categories),
                    "unit": m.unit,
                    "database": m.key[0],
                    "code": m.key[1],
                }
            )
        df = pd.DataFrame(rows).sort_values("bioflow_id").reset_index(drop=True)
        self._assert_fully_named(df, ("database", "code"), "biosphere")
        df.to_parquet(meta / "biosphere.parquet", index=False)

    def _emit_methods(self, written: WriteResult, out_root: Path) -> dict:
        methods = []
        for method, path in written.method_paths.items():
            methods.append(
                {
                    "key": list(method),
                    "slug": MethodSlug.encode(method),
                    "path": str(Path(path).relative_to(out_root)),
                }
            )
        return {"methods": sorted(methods, key=lambda m: m["slug"]), "version": 1}

    def _emit_parity_samples(
        self,
        meta: Path,
        production_for_col: dict[int, int],
        parity_scores: dict[int, dict[tuple[str, ...], float]],
        tolerance: float,
    ) -> None:
        # product id -> the activity column that produces it (the one an AB user
        # actually computes on).
        product_to_activity = {pid: cid for cid, pid in production_for_col.items()}
        samples = []
        for pid in sorted(parity_scores):
            cid = product_to_activity.get(pid)
            if cid is None:
                continue
            scores = parity_scores[pid]
            samples.append(
                {
                    "product_id": int(pid),
                    "activity_col_id": int(cid),
                    "expected": [
                        [list(method), float(score)] for method, score in sorted(scores.items())
                    ],
                }
            )
        payload = {"version": 1, "tolerance": float(tolerance), "samples": samples}
        (meta / "parity_samples.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _assert_fully_named(df: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
        """Fail the build if any required column is null/empty for any row.

        A nameless or keyless node makes the imported Brightway database
        unusable (Activity Browser shows blank rows; keys collide). Better to
        fail the build than write a silently degraded export.
        """
        for col in columns:
            series = df[col]
            bad = series.isna() | (series.astype("string").str.len() == 0)
            n_bad = int(bad.sum())
            if n_bad:
                example = df.loc[bad].head(3).to_dict("records")
                raise ValueError(
                    f"{table}.parquet: {n_bad} row(s) have an empty/NA '{col}'. "
                    f"The export would import as unusable nodes. Examples: {example}"
                )

    @staticmethod
    def _invert(id_to_idx: dict[int, int]) -> dict[int, int]:
        """``{id: idx}`` -> ``{idx: id}``, asserting the index space is a bijection."""
        idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}
        if len(idx_to_id) != len(id_to_idx):
            raise ValueError("Index map is not a bijection — duplicate matrix index.")
        return idx_to_id
