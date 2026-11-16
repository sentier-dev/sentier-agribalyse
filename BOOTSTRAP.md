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
> `.gitignore` and the pre-commit guard (`.githooks/`, see section 5) block them,
> but the responsibility is yours. Publishing more than five (5) ecoinvent Data
> Points, or any ecoinvent nomenclature, breaches the EULA (§6.1 / §7 / §9,
> CHF 100,000 confidentiality penalty).

The rationale and full data classification live in
[`docs/superpowers/specs/2026-06-03-ecoinvent-eula-clean-repo-design.md`](docs/superpowers/specs/2026-06-03-ecoinvent-eula-clean-repo-design.md).

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
2. writes the four files of the table above under `source/`.

Count on about five minutes and 1.5 GB under `cache/`. Running it again reuses
the imported project and only rewrites the files. Add `--discard-project` to
delete the Brightway project once the files are written.

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
`… - Copied from Ecoinvent U` processes). It is gitignored, and the guard
refuses any uncompressed `.CSV` under `source/`.

This applies equally to everything *derived* from the regenerated data —
including the Brightway export `bw_package/` written by
`dds-build-bw-package` after the pipeline has run (it embeds ecoinvent
LCI amounts).

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
