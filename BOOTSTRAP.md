# BOOTSTRAP.md — regenerating the ecoinvent-derived working data

This repository **does not ship any proprietary ecoinvent data**. To keep it
publishable under the ecoinvent EULA (v3, 2022-04-01), the following derived
artifacts are gitignored and must be regenerated locally by a **licensed
ecoinvent user** from their own credentials:

| Path | Content |
|---|---|
| `source/ecoinvent-3.9.1-cutoff-exchanges.parquet` | LCI exchange amounts (Data Points) |
| `source/ecoinvent-3.9.1-cutoff-activities.json` | activity nomenclature (name / location / reference product / code) |
| `source/ecoinvent-3.9.1-biosphere-flows.json` | ecoinvent elementary-flow nomenclature |
| `source/ef-v31-methods.json` | EF v3.1 methods with CFs inherited against the ecoinvent biosphere |
| `registry/method_cfs/ecoinvent-3.9.1__*/` | per-method CF parquets keyed by ecoinvent biosphere UUID (rebuilt by `dds-build-registry`) |

> **⚠️ These regenerated files ARE licensed ecoinvent data. NEVER commit them.**
> `.gitignore` and the pre-commit guard (`.githooks/`, see step 6) block them,
> but the responsibility is yours. Publishing more than five (5) ecoinvent Data
> Points, or any ecoinvent nomenclature, breaches the EULA (§6.1 / §7 / §9,
> CHF 100,000 confidentiality penalty).

The rationale and full data classification live in
[`docs/superpowers/specs/2026-06-03-ecoinvent-eula-clean-repo-design.md`](docs/superpowers/specs/2026-06-03-ecoinvent-eula-clean-repo-design.md).

---

## 1. Prerequisites

- A valid **ecoinvent licence** and credentials in `.env`:

  ```dotenv
  ECOINVENT_USERNAME=your-username
  ECOINVENT_PASSWORD=your-password
  ```

  Get these from <https://ecoquery.ecoinvent.org/> after licence purchase.

- The **`[bootstrap]`** extra, which pulls in the import-only toolchain
  (`bw2io`, `bw2data`, `ecoinvent_interface`). These are **not** part of the
  runtime — the pipeline reads only `source/` + `registry/` artifacts once
  bootstrapped:

  ```bash
  pip install -e ".[bootstrap]"
  ```

---

## 2. Import ecoinvent 3.9.1 cutoff into a Brightway project

Use `bw2io.import_ecoinvent_release` with your credentials. This downloads the
release and populates a bw2data project (SQLite `databases.db` + the ecoinvent
biosphere). Run once:

```python
import os
import bw2data as bd
import bw2io as bi

bd.projects.set_current("dds-bootstrap")
bi.import_ecoinvent_release(
    version="3.9.1",
    system_model="cutoff",
    username=os.environ["ECOINVENT_USERNAME"],
    password=os.environ["ECOINVENT_PASSWORD"],
)
# Confirm the database name the rest of the pipeline expects:
assert "ecoinvent-3.9.1-cutoff" in bd.databases
```

Note the on-disk location of that project's `databases.db` (bw2data prints the
project dir; the SQLite lives under `<project>/lci/databases.db`).

---

## 3. Regenerate `source/ecoinvent-3.9.1-cutoff-exchanges.parquet`

There **is** an in-repo, bw2data-free snapshotter for this one. Point it at the
SQLite from step 2:

```bash
dds-snapshot-ecoinvent-exchanges \
  --sqlite /path/to/<project>/lci/databases.db \
  --db-name ecoinvent-3.9.1-cutoff
```

It writes `source/ecoinvent-3.9.1-cutoff-exchanges.parquet`
(columns: `output_database, output_code, input_database, input_code, amount,
type`; deterministically sorted). Implementation:
[`src/cli/snapshot_ecoinvent_exchanges.py`](src/cli/snapshot_ecoinvent_exchanges.py).

---

## 4. Regenerate the three JSON snapshots (manual, one-time)

There is **no in-repo writer** for these yet — they were one-time extractions in
REFACTOR_FINAL phase F6. Reproduce them from the bw2data project (step 2) by
emitting JSON in the exact shapes the readers expect. Each writer must
deterministically sort its records so rebuilds are byte-stable.

### 4a. `source/ecoinvent-3.9.1-cutoff-activities.json`

Reader: [`src/matching/ecoinvent_catalog.py`](src/matching/ecoinvent_catalog.py)
(`EcoinventCatalogBuilder._load_snapshot` / `_row_for`).

```json
{
  "db_name": "ecoinvent-3.9.1-cutoff",
  "activities": [
    {
      "code": "<activity uuid>",
      "name": "<activity name>",
      "unit": "<reference unit>",
      "location": "<RoW | GLO | FR | ...>",
      "reference_product": "<reference product name>"
    }
  ]
}
```

Extract by iterating `bd.Database("ecoinvent-3.9.1-cutoff")`: each activity's
`code`, `name`, `unit`, `location`, and reference-product name (the
`production`/reference exchange). `reference product` (with a space) is also
accepted by the reader.

### 4b. `source/ecoinvent-3.9.1-biosphere-flows.json`

Reader: [`src/matching/bio_registry.py`](src/matching/bio_registry.py)
(`BiosphereRegistryBuilder._rows_from_json`).

```json
{
  "flows": [
    {
      "code": "<biosphere flow uuid>",
      "name": "<flow name>",
      "categories": ["<top>", "<sub>"],
      "unit": "<unit>",
      "cas": "<CAS or null>",
      "synonyms": ["<synonym>", "..."]
    }
  ]
}
```

A bare top-level JSON array is also accepted (the reader unwraps `flows` if
present, else treats the payload as the list). Extract by iterating the
ecoinvent biosphere database from step 2.

### 4c. `source/ef-v31-methods.json`

Reader: [`src/ef/cf_registry.py`](src/ef/cf_registry.py)
(`MethodCfsRegistryBuilder._load_methods_snapshot` / `_cfs_for_method`).

```json
{
  "ef_db_name": "<ef biosphere db name>",
  "methods": {
    "<slug label>": {
      "key": ["<m0>", "<m1>", "<method category>", "<indicator>"],
      "inherited_cfs": [
        { "db": "ecoinvent-3.9.1-biosphere", "code": "<flow uuid>", "amount": 0.0 }
      ]
    }
  }
}
```

`key` is the 4-tuple bw2data method key. `inherited_cfs` are the biosphere3 /
ecoinvent-biosphere CFs that the legacy `EfMethodAugmenter` preserved (the
non-`ef_db_name` entries). The authoritative JRC EF v3.1 CFs themselves come
from the kept public parquet (`source/EF-LCIAMethod_CF(EF-v3.1)__lciamethods_CF.parquet`);
this snapshot only carries the inherited biosphere-keyed CFs.

---

## 5. Rebuild the registry

With the four `source/` files in place, rebuild every registry parquet —
including `registry/method_cfs/ecoinvent-3.9.1__EF*/`:

```bash
dds-build-registry
```

Then run the pipeline as usual (pardiso solver — see `CLAUDE.md`):

```bash
dds-run-end-to-end --solver pardiso     # link + register LCIA + score a sample
```

---

## 6. Never re-commit the regenerated data

The repo blocks these paths in two places:

1. `.gitignore` — the `ecoinvent EULA` block.
2. A pre-commit guard at `.githooks/pre-commit`, wired into the existing
   `pre-commit` framework as the `ecoinvent-eula-guard` hook. Activate once per
   clone (this also installs the ruff hooks):

   ```bash
   pip install pre-commit   # if not already present
   pre-commit install
   ```

   If you don't use the `pre-commit` framework, run the guard as a raw git hook
   instead: `git config core.hooksPath .githooks`. (Don't do both — `core.hooksPath`
   overrides `.git/hooks`, so `pre-commit install` would be bypassed.)

If `git add` ever stages one of the gitignored ecoinvent paths (e.g. via
`git add -f`), the guard aborts the commit. Do not override it.
