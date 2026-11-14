# sentier-agribalyse

```mermaid
flowchart LR
    ADEME["ADEME: Agribalyse 3.2 SimaPro CSV (ecoinvent-free, in git) + reference scores"] -- read --> REPO["sentier-agribalyse"]
    BF["Brightway Flows 1.1.0 (Zenodo)"] -- "dds-fetch-harmonised-flows" --> REPO
    EI["ecoinvent 3.9.1 (licensed)"] -- "snapshot (BOOTSTRAP.md)" --> REPO
    EF["JRC EF v3.1 CF parquet"] -- read --> REPO
    RD["randonneur_data registry"] -- read --> REPO
    REPO -- write --> DASH["dashboard/ (static backtest UI)"]
    REPO -- export --> BW["bw_package/ (Brightway, Activity Browser)"]
    REPO -- author --> RP["source/randonneur_packages/"]
```

Public, ecoinvent-EULA-clean adapter that brings Agribalyse 3.2 into Brightway 2.5.

## What it is

- Imports the ADEME Agribalyse 3.2 SimaPro CSV.
- Links it to ecoinvent 3.9.1 through a tiered `MappingRegistry`. Tiers are data, not code.
- Registers the 19 headline EF v3.1 LCIA methods and scores every product.
- Backtests every score against ADEME reference values in a static React dashboard.
- Exports the linked system as Brightway datapackages plus a standalone importer.

No proprietary ecoinvent data ships in git. The ecoinvent-derived files under `source/` and
`registry/method_cfs/` are gitignored and blocked by a pre-commit guard. Licensed users
regenerate them locally: see [BOOTSTRAP.md](BOOTSTRAP.md).

A clone plus an ecoinvent 3.9.1 licence is enough to run everything. The Agribalyse
inventories ship as `source/AGB32_ecoinvent_free.CSV.xz`: see
[The Agribalyse CSV](#the-agribalyse-csv).

Deep dives (mapping tiers, registry contract, link sequence, solver notes) live in the DdS wiki
page `sentier-agribalyse`.

```mermaid
flowchart LR
    SNAP["dds-snapshot-ecoinvent-exchanges (bootstrap)"] -.-> REG[dds-build-registry]
    PK[dds-build-packages] -. "writes source/" .-> REG
    REG --> EFF[dds-build-ef-flows-registry] --> BIO[dds-build-biosphere-catalog] --> MCF[dds-build-method-cfs-registry] --> LINK[dds-link-all]
    EIC[dds-build-ecoinvent-catalog] --> LINK
    REG --> E2E[dds-run-end-to-end]
    RESET[dds-reset] -.-> LINK
    LINK --> BT[dds-backtest] --> FD[dds-build-flow-decomp] --> PR[dds-build-product-reasons]
    MCF --> CFS[dds-compare-cfs] --> CFC[dds-build-cf-comparison-csv]
    LINK --> DS[dds-decompose-score]
    LINK --> BWP[dds-build-bw-package]
    LINK --> SK[dds-build-skeleton]
    LINK --> MC[dds-mappings-comparison]
    LINK --> LLM[dds-llm-suggest-mappings] -.-> REG
    BP[dds-build-parameters] -.-> LP[dds-list-parameters]
    SP[dds-set-parameter] --> LINK
    CP[dds-clear-parameters] -.-> LINK
```

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/).
- Network access once, for `dds-fetch-harmonised-flows` (46 MB from Zenodo).
- An ecoinvent 3.9.1 licence to regenerate the gitignored `source/` files. Credentials go in `.env` as `ECOINVENT_USERNAME` and `ECOINVENT_PASSWORD`. Bootstrap only: see [BOOTSTRAP.md](BOOTSTRAP.md).
- The `pardiso` extra for scoring. scipy rejects the matrix's zero-diagonal placeholder activities as "exactly singular".

## Install

```bash
uv sync --extra pardiso
```

Add `--extra bw` for the Brightway export, `--extra test` for pytest, `--extra bootstrap` to snapshot ecoinvent, `--extra llm-api` for `--use-api`. pip works too: `pip install -e ".[pardiso]"`.

Activate the EULA guard once per clone:

```bash
pre-commit install
```

## Use

Run every script as `uv run <script>`.

| Command | What it does |
|---|---|
| `dds-snapshot-ecoinvent-exchanges` | Bootstrap only. Dump ecoinvent exchanges from a bw2data SQLite into `source/`. |
| `dds-fetch-harmonised-flows` | Download the pinned Brightway Flows list into `source/` and check its sha256. Once per clone. `--force`. |
| `dds-build-registry` | Build `registry/*.parquet` from every `source/` artifact. Run after the fetch. |
| `dds-strip-ecoinvent` | Maintainers only. Rebuild `source/AGB32_ecoinvent_free.CSV.xz` and its manifest from ADEME's full export. |
| `dds-build-ef-flows-registry` | Write `registry/ef_flows.parquet`, the EF flow universe. |
| `dds-build-biosphere-catalog` | Write `registry/biosphere_catalog.parquet` from the biosphere snapshots. |
| `dds-build-method-cfs-registry` | Write `registry/method_cfs/`, per-method CF parquets plus index. |
| `dds-build-ecoinvent-catalog` | Write `registry/ecoinvent_catalog.parquet` from the activity snapshot. |
| `dds-build-parameters` | Write `registry/parameters.parquet` and `registry/exchange_formulas.parquet` from the CSV. |
| `dds-build-packages` | Author the randonneur datapackages in `source/randonneur_packages/` and the residuals review xlsx. |
| `dds-link-all` | Full link pipeline. Writes the scoring package and `dashboard/run_report.json`. |
| `dds-run-end-to-end` | Smoke test: link, register LCIA, score a small sample. |
| `dds-backtest` | Score every mapped product against ADEME. Writes `dashboard/backtest/` and `dashboard/backtest_pass1.csv`. |
| `dds-compare-cfs` | SimaPro-vs-registry per-flow CF comparison. Writes `dashboard/cf_comparison.csv`. |
| `dds-build-cf-comparison-csv` | Re-flatten `registry/cf_comparison_join.parquet` into `dashboard/cf_comparison.csv`. |
| `dds-build-flow-decomp` | Per-product flow-decomposition JSONs in `dashboard/decomp/`. |
| `dds-build-product-reasons` | LLM outlier notes in `dashboard/product_reasons.json`. Needs the `claude` binary or `--use-api`. |
| `dds-decompose-score` | Explain one `(product, method)` score: top biosphere-flow contributions. |
| `dds-build-bw-package` | Export `bw_package/`: bw_processing datapackages plus a standalone Brightway importer. Needs the `bw` extra. |
| `dds-build-skeleton` | Strip ecoinvent amounts from a scoring package into an AGB-only skeleton. |
| `dds-mappings-comparison` | Regenerate `to_review/mappings_comparison.xlsx` from the cached link. |
| `dds-llm-suggest-mappings` | LLM picks for residual unlinked flows. Appends accepted rows to the reviewed xlsx. |
| `dds-set-parameter` | Override a SimaPro input parameter, then relink and rescore. `--fast`, `--no-rescore`. |
| `dds-list-parameters` | Browse SimaPro parameter names, ranges, and active overrides. `--name-like`. |
| `dds-clear-parameters` | Remove overrides: all, `--name`, or `--product`. |
| `dds-reset` | Delete caches and overrides so the next link re-parses the CSV. `--keep-overrides`. |

Flags worth knowing:

- `--solver pardiso` on `dds-backtest`, `dds-run-end-to-end`, `dds-decompose-score`. Already the default on `dds-build-flow-decomp` and `dds-set-parameter`.
- `--skip-ecoinvent` on `dds-link-all` and `dds-run-end-to-end`; `--skip-linking` on `dds-run-end-to-end`.
- `--no-llm` on `dds-link-all` disables LLM overrides and the curated synonym fallback.

## Typical workflow

1. Licensed users, once: regenerate the ecoinvent-derived `source/` files per [BOOTSTRAP.md](BOOTSTRAP.md).
2. Fetch the flow list, once: `uv run dds-fetch-harmonised-flows`.
3. Build the registry: `uv run dds-build-registry`, then `dds-build-ef-flows-registry`, `dds-build-biosphere-catalog`, `dds-build-method-cfs-registry`, `dds-build-ecoinvent-catalog`.
4. Link: `uv run dds-link-all`.
5. Score: `uv run dds-backtest --solver pardiso`.
6. Dashboard data: `uv run dds-compare-cfs`, then `uv run dds-build-flow-decomp`.
7. Optional tooltips: `uv run dds-build-product-reasons`.
8. View: `python -m http.server 8000 --directory dashboard`, then open `http://localhost:8000/backtest_dashboard.html`.
9. Optional export: `uv run dds-build-bw-package` (needs the `bw` extra). `bw_package/` embeds ecoinvent data; it is gitignored and blocked by the guard.

After replacing the SimaPro CSV run `uv run dds-reset`, then repeat from step 4.

## The Agribalyse CSV

ADEME's SimaPro export of Agribalyse 3.2 embeds 2,264 ecoinvent datasets as process blocks named `… - Copied from Ecoinvent U`. Those are licensed ecoinvent data, so the export cannot be published as is.

What ships is `source/AGB32_ecoinvent_free.CSV.xz`: the same export with those 2,264 blocks removed, 18,176 processes kept, every other byte untouched. The `… - Adapted from Ecoinvent U` processes are ADEME's own modified datasets and stay, as in the Agribalyse 4.0 "without ecoinvent" export. Two small files describe what was removed, with no amounts:

- `source/agribalyse-3.2-stripped-ecoinvent-processes.json`: each removed block by SimaPro identifier, type and sha256.
- `source/agribalyse-3.2-copied-unit-processes.json`: the SimaPro names of the 125 removed blocks that were unit processes.

The pipeline puts the removed datasets back from your own ecoinvent 3.9.1:

- An input that pointed at an aggregated copy is linked to the same ecoinvent activity.
- The 125 copied unit processes (market mixes, crops, seeds) are rebuilt from your ecoinvent exchanges. Inside the Agribalyse library they are what connects ADEME's adapted datasets to everything else, so each rebuilt input goes to the adapted dataset when there is one, to another rebuilt process when it was a unit copy, and to ecoinvent otherwise. Two of the 125 are ADEME's combustion extracts (air emissions of a transport dataset per kilogram of fuel) and are rebuilt the same way.
- SimaPro files several ecoinvent activities under one shortened name. A rebuilt process follows the block the library actually holds under that name.

Every input is linked. Eleven `recycling of …` treatments are empty in the cut-off model and are dropped without effect.

Which file is read:

- `source/AGB32_final.CSV` (ADEME's full export), when you have it. It is gitignored and the guard refuses it.
- Otherwise the shipped archive, decompressed once into `cache/`.

`dashboard/run_report.json` records the variant under `stages.csv_source` and the rebuild under `stages.unit_copy_rebuild`.

Share of the 2,451 backtested products within ±5% of ADEME's reference, shipped file against full export:

| | Shipped | Full |
|---|---|---|
| Climate change | 99.8% | 99.8% |
| Climate change, fossil / biogenic / land use change | 99.5% / 99.8% / 100.0% | 99.5% / 99.8% / 100.0% |
| Ozone depletion | 100.0% | 100.0% |
| Ionising radiation | 100.0% | 100.0% |
| Photochemical ozone formation | 100.0% | 100.0% |
| Particulate matter | 98.8% | 98.8% |
| Acidification | 93.6% | 93.6% |
| Eutrophication, freshwater / marine / terrestrial | 100.0% / 100.0% / 96.2% | 100.0% / 100.0% / 96.2% |
| Land use | 100.0% | 100.0% |
| Energy resources / material resources | 100.0% / 100.0% | 100.0% / 100.0% |
| Human toxicity, non-carcinogenic | 93.8% | 93.2% |
| Ecotoxicity, freshwater | 95.8% | 52.7% |
| Human toxicity, carcinogenic | 36.5% | 44.5% |
| Water use | 22.8% | 29.7% |
| **All 19 indicators** | **91.4%** | **89.9%** |

The two variants differ on four indicators, for one reason: with the shipped file the ecoinvent background keeps its own sub-compartments and locations, where the embedded copies are aggregated blocks. Checked against SimaPro's own factors on the embedded copies, the ecoinvent route is the closer one on freshwater ecotoxicity, carcinogenic toxicity and water use. It moves the product scores toward ADEME's reference on ecotoxicity and away from it on carcinogenic toxicity and water use, two indicators that were already the weakest.

Maintainers holding the full export rebuild the three files with `uv run dds-strip-ecoinvent`. It fails if any process of the aggregated-process deletion list survives, and `tests/integration/test_ecoinvent_free_csv_shipped.py` audits the committed files.

## Changing parameters (no SimaPro required)

AGB 3.2's SimaPro process-level parameters (577 names across 13,725 processes, packaging stage included) are editable from the command line. Only input parameters accept overrides; calculated parameters are formula-derived and refuse with their formula.

```bash
uv run dds-list-parameters --name-like packaging                          # names, counts, value ranges, active overrides
uv run dds-set-parameter Packaging_Weight 0.03 --product EI3CQUNI000025017101234
uv run dds-set-parameter Packaging_Weight 0.03 --all-products             # every process that defines it
uv run dds-clear-parameters                                               # back to the baseline
```

`dds-set-parameter` writes the override to `source/parameter_overrides.csv` (gitignored; a process-specific row beats a `*` row), prints the changed exchange amounts, then reruns `dds-link-all` and `dds-backtest` so the dashboard shows the scenario. `--no-rescore` skips the rerun. `--fast` replays only the scoring-package stage against the cached linked graph (`cache/linked_cache.pkl`) and ratio-patches the amounts; exchanges whose baseline evaluates to 0 are reported and need the full path. Baseline and what-if scoring packages coexist in `cache/scoring_packages/` because the content hash covers the amounts. `dds-reset` deletes the overrides file unless you pass `--keep-overrides`.

Rebuilds are deterministic: after `dds-clear-parameters` the next run reproduces the original scores exactly. This is verified on the full 2,451-product backtest.

### Changing parameters in SimaPro instead

For structural edits (new parameters, changed formulas), edit in SimaPro, re-export with the original settings, replace `source/AGB32_final.CSV`, then run `uv run dds-reset`, `uv run dds-link-all`, `uv run dds-backtest`. `dds-reset` is required because the CSV parse cache does not hash the source file.

## Layout

```
source/        inputs + the randonneur packages we publish (ecoinvent-derived files gitignored)
registry/      built mapping parquets, single source of mapping truth (gitignored, regenerable)
cache/         parquet caches, importer pickle, scoring packages (gitignored)
dashboard/     static React UI + backtest_pass1.csv (committed); other generated data gitignored
to_review/     human-review artifacts
unlinked/      residual unlinked exports
bw_package/    Brightway export (gitignored, EULA-guarded)
src/           the package, flat layout: cli/, registry/, matching/, transforms/, ef/, scoring/,
               reporting/, pipelines/, bw_export/, bw_import/, exports/, llm/
tests/         pytest suite: unit/, integration/, fixtures/
.githooks/     pre-commit ecoinvent EULA guard
BOOTSTRAP.md   how a licensed user regenerates the gitignored source/ files
```

## Licence

Code: MIT (`pyproject.toml`). ecoinvent data is licensed separately and never ships in this
repo; regenerated files stay local under the ecoinvent EULA.

Agribalyse 3.2 is published by ADEME under the Etalab Open Licence 2.0; the shipped CSV is that
export without its embedded ecoinvent datasets. The flow list fetched by
`dds-fetch-harmonised-flows` is Brightway Flows 1.1.0,
[doi:10.5281/zenodo.23554066](https://doi.org/10.5281/zenodo.23554066), under ODbL.
