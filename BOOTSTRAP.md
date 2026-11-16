# BOOTSTRAP.md — regenerating the ecoinvent-derived working data

This repository **does not ship any licensed ecoinvent data**: no exchange
amounts (Data Points), no characterisation factors from ecoinvent's LCIA
implementation, and no build that embeds them, so it stays publishable under the
ecoinvent EULA (v3, 2022-04-01).

Names are a different matter. ecoinvent publishes the metadata used in its
datasets (activity names, products, elementary flows, geographies, units,
compartments) in its Data Glossary under the Open Database License (ODbL), with
individual entries under the Database Contents License
([archived copy](https://web.archive.org/web/20260911185012/https://glossary.ecoinvent.org/); the glossary host no longer resolves). A single name
can be cited freely; a substantial extract, such as a whole snapshot below, may
only be published under the ODbL with its notice.

The following artifacts are regenerated locally by a **licensed ecoinvent
user** from their own credentials:

| Path | Content | In git? |
|---|---|---|
| `source/ecoinvent-3.9.1-cutoff-exchanges.parquet` | LCI exchange amounts (Data Points) | **licensed: never commit** |
| `source/ef-v31-methods.json` | EF v3.1 methods with CFs attached to the ecoinvent biosphere through ecoinvent's LCIA implementation | **licensed: never commit** |
| `registry/method_cfs/ecoinvent-3.9.1__*/` | per-method CF parquets keyed by ecoinvent biosphere UUID (rebuilt by `dds-build-registry`) | **licensed: never commit** |
| `source/ecoinvent-3.9.1-cutoff-activities.json` | dataset list (name / location / unit / reference product) plus bw2io's `code` (MD5 of ecoinvent's activity and product UUIDs), no amounts | **keep local**: the names are ODbL glossary data, but this list and `code` are not confirmed as published |
| `source/ecoinvent-3.9.1-biosphere-flows.json` | elementary flows (UUID / name / compartment / unit / CAS / synonyms), no amounts | **ODbL, not EULA data**: ecoinvent Data Glossary content; the tracked `source/biosphere3-flows.json` holds the 3.9 version of this list. Ignored because it is regenerated, and the guard refuses it until this reading is confirmed |

> **⚠️ The files marked "licensed" or "keep local", the outputs built from them (section 5) and any built `bw_package/` must NEVER be committed.**
> `.gitignore` and the pre-commit guard (`.githooks/`, see section 5) block them,
> but the responsibility is yours. Publishing more than five (5) ecoinvent Data
> Points breaches the EULA (§6.1 / §7 / §9, CHF 100,000 confidentiality penalty).

---

## 1. Prerequisites

- A valid **ecoinvent licence** and its credentials in `.env` at the repository root:

  ```dotenv
  ECOINVENT_USERNAME=your-username
  ECOINVENT_PASSWORD=your-password
  ```

  They are the login of <https://ecoquery.ecoinvent.org/>.

- The **`bootstrap`** extra, which pulls in the import-only toolchain
  (`bw2io`, `bw2data`, `ecoinvent_interface`). The pipeline itself never uses
  it:

  ```bash
  uv sync --extra pardiso --extra bootstrap
  ```

---

## 2. Bootstrap: one command

```bash
uv run dds-bootstrap-ecoinvent
```

It does everything this file used to describe by hand:

1. downloads ecoinvent 3.9.1 cutoff with your credentials and imports it into
   a private Brightway project under `cache/bootstrap-brightway/` (your own
   Brightway projects are not touched);
2. writes the four `source/` files of the table above.

Count on about five minutes and 1.5 GB under `cache/`. Running it again reuses
the imported project and only rewrites the files. Add `--discard-project` to
delete the Brightway project once the files are written, or `--fresh` to
start from an empty project (after an interrupted import, for instance).

The command ends by printing the next step.

---

## 3. Fetch the flow list and build the registry

One more input is not in git, because it is larger than the repository's
file-size guard: the Brightway Flows consensus elementary flow list
(`source/harmonised-flows-simple.json.gz`, 46 MB, ODbL,
[doi:10.5281/zenodo.23554066](https://doi.org/10.5281/zenodo.23554066)).
It needs no licence. The command checks its sha256:

```bash
uv run dds-fetch-harmonised-flows
```

The Agribalyse inventories themselves need no step: they ship in git as
`source/AGB32_ecoinvent_free.CSV.xz` and are decompressed into `cache/` on
first use.

Then build every registry parquet, including
`registry/method_cfs/ecoinvent-3.9.1__EF*/`:

```bash
uv run dds-build-registry
uv run dds-build-ef-flows-registry
uv run dds-build-biosphere-catalog
uv run dds-build-method-cfs-registry
uv run dds-build-ecoinvent-catalog
```

and run the pipeline (see the README):

```bash
uv run dds-link-all
uv run dds-backtest --solver pardiso
```

---

## 4. If you already have ecoinvent in a Brightway project

`dds-bootstrap-ecoinvent` is the supported path. The exchange file alone can
also be written from any Brightway project that holds
`ecoinvent-3.9.1-cutoff`, without `bw2data`:

```bash
uv run dds-snapshot-ecoinvent-exchanges \
  --sqlite /path/to/<project>/lci/databases.db \
  --db-name ecoinvent-3.9.1-cutoff
```

Brightway keeps one `databases.db` per project. The command stops with an
error when the file holds no exchange of that database, which is what
happens with another project's file. The three JSON files have no stand-alone
writer; their shapes are those `EcoinventMetadataSnapshotter`
([`src/bootstrap/ecoinvent.py`](src/bootstrap/ecoinvent.py)) writes.

---

## 5. Never re-commit the regenerated data

It applies to ADEME's full SimaPro export too, if you hold one:
`source/AGB32_final.CSV` embeds ecoinvent inventories (the
`… - Copied from Ecoinvent U` processes). It is gitignored. The guard refuses
every file under `source/` whose name holds `.csv` in any case, so also a
compressed or renamed copy and one in a subfolder, and a file named
`AGB32_final.*` anywhere. The one exception is the stripped archive
`source/AGB32_ecoinvent_free.CSV.xz`, by its exact path: the guard does not
look into it, a test checks its checksum against the manifest.

This applies equally to everything *derived* from the regenerated data. Once
the pipeline has run, these outputs carry licensed numbers too:

| Path | Written by | Licensed content |
|---|---|---|
| `cache/scoring_packages/<hash>/` | `dds-link-all` | ecoinvent exchange amounts and the registry CFs, as CSR matrices |
| `cache/importer_cache.pkl`, `cache/linked_cache.pkl` | `dds-link-all` | the parsed and linked SimaPro CSV: the linked one holds the unit processes rebuilt from the ecoinvent snapshot, with their amounts, and both hold ADEME's copied ecoinvent processes when the full export was read |
| `registry/method_cfs/<slug>/` | `dds-build-method-cfs-registry` | per-method CFs, including those inherited from `source/ef-v31-methods.json` |
| `registry/cf_comparison_join.parquet`, `dashboard/cf_comparison.csv` | `dds-compare-cfs`, `dds-build-cf-comparison-csv` | the registry CF of each flow (`cf_registry`) |
| `to_review/cf_comparison/` | `CfComparisonExporter` | the mean registry CF of each flow |
| `dashboard/decomp/` | `dds-build-flow-decomp` | per-flow inventory and registry CF for each product |
| `dashboard/product_reasons.json` | `dds-build-product-reasons` | notes written from the decomposition's flow shares and CF differences |
| `registry/ecoinvent_catalog.parquet`, `registry/activity_catalog.parquet` | `dds-build-ecoinvent-catalog`, `dds-link-all` | the dataset list with bw2io's `code`, as in the activities snapshot |
| `bw_package/` | `dds-build-bw-package` | ecoinvent LCI amounts and the registry CFs |

The repo keeps these paths out of git in two places:

1. `.gitignore` — the `ecoinvent EULA` block (the four snapshot files and
   `registry/method_cfs/`), the entries for SimaPro exports under `source/`
   and for `AGB32_final.*`, and the entries `registry/*.parquet`, `cache/`,
   `dashboard/*`, `to_review/*` and `bw_package/`.
2. A pre-commit guard at `.githooks/pre-commit`, which refuses the licensed
   files, the activities and elementary-flow snapshots, every path in the
   table above, any SimaPro
   export under `source/` but the stripped archive, and a file named
   `AGB32_final.*` anywhere, wired into the existing `pre-commit`
   framework as the `ecoinvent-eula-guard` hook. Activate once per clone (this
   also installs the ruff hooks):

   ```bash
   pip install pre-commit   # if not already present
   pre-commit install
   ```

   If you don't use the `pre-commit` framework, run the guard as a raw git hook
   instead: `git config core.hooksPath .githooks`. (Don't do both — `core.hooksPath`
   overrides `.git/hooks`, so `pre-commit install` would be bypassed.)

The guard also refuses a `bw_package` directory anywhere in a path, an archive
of it (`bw_package.zip`, `bw_package.tar.gz` and the like), and the matrix files of a scoring package or a Brightway
export under any directory name (`*.csr.npz`; `technosphere.*.npy`,
`biosphere.*.npy`, `characterization.*.npy`), so an export written with
`--out` is caught as well. Both hold under `src/` and `tests/` too, so a test
fixture cannot be a matrix file of these names. It refuses suffixed copies of the licensed files
(`…-exchanges.parquet.bak`) on purpose: they hold the same data. It checks
renamed and non-ASCII paths. It allows the tracked elementary-flow list
`source/biosphere3-flows.json`,
`registry/method_cfs/_index.json`, `registry/cf_comparison_by_code.parquet`
(SimaPro CFs only) and the product scores under `dashboard/backtest/`.

If `git add -f` stages any of these paths, the guard aborts the commit. Do not
override it.
