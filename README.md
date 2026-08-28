# sentier-agribalyse

Sentier-native adapter that imports Agribalyse 3.2 into Brightway 2.5,
links it against ecoinvent 3.9.1, and registers EF v3.1 LCIA methods
for impact scoring.

## Prerequisites

- Python 3.11+
- ecoinvent credentials (`ECOINVENT_USERNAME` / `ECOINVENT_PASSWORD` in `.env`)
- A populated `source/` directory (see [Sources](#sources) below)

> **No proprietary ecoinvent data ships in this repo.** The ecoinvent-derived
> files under `source/` and `registry/method_cfs/ecoinvent-3.9.1__*/` are
> gitignored under the ecoinvent EULA. A licensed user regenerates them locally
> from their own credentials — see **[BOOTSTRAP.md](BOOTSTRAP.md)**.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[pardiso]"   # pardiso = recommended solver (see the solver note)
```

## Commands

After install, the entrypoints are available as console scripts (and
also as `python -m cli.<name>`):

| Command | What it does |
|---|---|
| `dds-build-registry` | Build `registry/*.parquet` from every source — the foundation everything else depends on. |
| `dds-link-all` | Full link pipeline: Brightway setup → ecoinvent → EF layer → CSV → transforms → biosphere matcher → technosphere matcher → write DB → matrix-square purge. |
| `dds-run-end-to-end` | Link + register LCIA + score a sample of products. |
| `dds-backtest` | Score every mapped product vs. ADEME's reference and write parquet diffs to `dashboard/backtest/` plus `dashboard/backtest_pass1.csv` — the data source for the dashboard's **Product %diff** tab. |
| `dds-compare-cfs` | Build the SimaPro-vs-registry per-flow CF comparison (`dashboard/cf_comparison.csv` + `registry/cf_comparison_*.parquet`) that powers the dashboard's **CF comparison** tab. |
| `dds-build-flow-decomp` | Write per-product flow-decomposition JSONs to `dashboard/decomp/` for the dashboard's per-product drill-down panel. |
| `dds-build-product-reasons` | Author per-product, per-outlier-impact LLM explanations into `dashboard/product_reasons.json`, shown in the dashboard's cell tooltips. Needs the local `claude` binary (default) or `--use-api` with an Anthropic key. Optional. |
| `dds-decompose-score` | Explain a single `(product, method)` score: top biosphere flows, technosphere activities, and `(activity, flow)` edges. |
| `dds-build-packages` | Author the publishable randonneur datapackages (`source/randonneur_packages/*.json`) and the residuals review xlsx. |
| `dds-mappings-comparison` | Regenerate `to_review/mappings_comparison.xlsx` from the persisted DB without re-running the link pipeline. |
| `dds-build-bw-package` | Export the linked system as native Brightway artifacts: `bw_processing` datapackages + a standalone importer for Brightway 2.0/2.5 and Activity Browser. See [Export to Brightway](#export-to-brightway--activity-browser). Needs the `bw` extra. |
| `dds-set-parameter` | Override a SimaPro input parameter (e.g. `Packaging_Weight`) without SimaPro, then rescore. See "Changing parameters" below. |
| `dds-list-parameters` | Browse the 577 SimaPro parameter names, definition counts, value ranges, and active overrides. |
| `dds-clear-parameters` | Remove overrides (all, `--name`, or `--product`) — the baseline is restored on the next rebuild. |
| `dds-build-parameters` | Materialize `registry/parameters.parquet` + `registry/exchange_formulas.parquet` from the parsed CSV. |

Common flags on `dds-link-all`:

```
--skip-ecoinvent  Skip ecoinvent download/import (must already be loaded).
--no-llm          Disable LLM overrides AND curated synonym fallback.
--no-write        Skip the database write (dry-run, audit-log inspection).
--no-purge        Skip matrix-square purge (still writes DB).
```

### Changing parameters (no SimaPro required)

The SimaPro process-level parameters (all process-local in AGB 3.2 —
13 725 processes with input parameters, 577 distinct names) are editable
directly:

```bash
dds-list-parameters --name-like packaging          # discover names + ranges
dds-set-parameter Packaging_Weight 0.03 --product EI3CQUNI000025017101234
dds-set-parameter Packaging_Weight 0.03 --all-products   # every defining process
dds-clear-parameters                                # back to baseline values
```

`dds-set-parameter` persists the override to
`source/parameter_overrides.csv` (gitignored what-if state; a
process-specific row beats a `*` row), prints the directly changed
exchange amounts, then reruns `dds-link-all` + `dds-backtest`
(`--no-rescore` to skip). With `--fast` the rescore replays only the
scoring-package emit stage against the linked-graph snapshot
(`cache/linked_cache.pkl`, written by every `dds-link-all` run) —
skipping parse, transforms, and matching. The fast path ratio-patches
amounts (`new_formula_value / baseline_formula_value`), which survives
the transforms' multiplicative rescales; exchanges whose baseline
evaluates to 0 can't be ratio-patched and are reported (use the full
path for those). Only **input** parameters accept overrides —
calculated parameters are formula-derived and refuse with their formula.
Overrides re-evaluate the affected processes' formulas with the same
machinery that baked the original amounts (`bw2parameters`), so an
override set to the original value changes nothing. The scoring-package
content hash covers the exchange amounts, so baseline and what-if
packages coexist in `cache/scoring_packages/`. `dds-reset` deletes the
overrides file (`--keep-overrides` preserves it).

### Changing parameters (updated SimaPro CSV)

For structural edits (new parameters, changed formulas) the pipeline is
fully regenerable from the SimaPro export. After editing in SimaPro,
re-export the database and replace `source/AGB32_final.CSV` (same export
settings as the original), then:

```bash
dds-reset       # clears the scoring-package cache AND the CSV parse cache
dds-link-all    # re-parses the CSV, re-links, rebuilds the scoring package
dds-backtest    # re-scores everything and refreshes the dashboard data
```

`dds-reset` is required: the CSV parse is cached in
`cache/importer_cache.pkl` with no hash of the source file, so without it
a re-run would silently reuse the previous parse. The rebuild is
deterministic — the same CSV always reproduces bit-identical outputs, so
score differences between two runs are attributable to the CSV changes
alone. (`dds-build-registry` is only needed when mapping sources under
`source/` change; parameter edits don't touch it.)

### Full workflow (from scratch)

Run these in order on a clean checkout. Step 0 is a one-time bootstrap
for a licensed ecoinvent user; steps 1-6 rebuild everything the
dashboard renders. All artifacts they write under `source/`,
`registry/`, and the generated `dashboard/` data are gitignored (see
[Dashboard data & the EULA](#dashboard-data--the-ecoinvent-eula)).

```bash
# 0. One-time: regenerate the ecoinvent-derived source/ files from your
#    own ecoinvent licence. See BOOTSTRAP.md. Skip if source/ is already
#    populated.

# 1. Build registry/*.parquet from every source (foundation for the rest).
dds-build-registry

# 2. Full link pipeline → writes the scoring package + run_report.json.
dds-link-all

# 3. Score every product vs ADEME → dashboard/backtest_pass1.csv
#    (Product %diff tab). pardiso is recommended — see the solver note.
dds-backtest --solver pardiso

# 4. SimaPro-vs-ours per-flow CF comparison → dashboard/cf_comparison.csv
#    (CF comparison tab).
dds-compare-cfs

# 5. Per-product flow-decomposition JSONs → dashboard/decomp/
#    (the drill-down panel opened by clicking a product row).
dds-build-flow-decomp --solver pardiso

# 6. Optional: per-product LLM outlier notes → dashboard/product_reasons.json
#    (cell tooltips). Needs the local `claude` binary or --use-api.
dds-build-product-reasons

# 7. Optional: export the linked system for Brightway / Activity Browser
#    (needs: pip install -e ".[bw,pardiso]"). Output bw_package/ is licence-gated.
dds-build-bw-package
```

`dds-run-end-to-end --solver pardiso` is an optional quick smoke that
links + registers LCIA + scores a small sample; `dds-backtest` (step 3)
supersedes it for the dashboard. After the first full run, add
`--skip-ecoinvent` (and on `dds-run-end-to-end`, `--skip-linking`) to
skip the expensive relink.

> **Solver:** use `--solver pardiso` on `dds-backtest`,
> `dds-run-end-to-end`, and `dds-build-flow-decomp`. The scoring-package
> matrix has zero-diagonal placeholder activities that scipy's SuperLU
> rejects as "exactly singular"; pypardiso's pivoting handles them.
> Install it with the `pardiso` extra: `pip install -e ".[pardiso]"`.
> The CLIs fall back to scipy if pardiso is absent.

### View the backtest dashboard locally

After steps 3-6 above, serve `dashboard/` and open the static React UI:

```bash
python -m http.server 8000 --directory dashboard
# then open http://localhost:8000/backtest_dashboard.html
```

The dashboard has three tabs, each driven by a file the workflow writes:

| Tab | Data source | Built by |
|---|---|---|
| **Product %diff** | `dashboard/backtest_pass1.csv` | `dds-backtest` |
| **CF comparison** | `dashboard/cf_comparison.csv` | `dds-compare-cfs` |
| drill-down panel | `dashboard/decomp/<code>.json` | `dds-build-flow-decomp` |
| cell tooltips | `dashboard/product_reasons.json` | `dds-build-product-reasons` |

`backtest_pass1.csv` is emitted every `dds-backtest` run with the
[`NearZeroFloor`](src/reporting/near_zero_floor.py) noise-suppression
rule and the long-name → short-id translation
([`BacktestPass1Emitter`](src/reporting/backtest_dashboard_csv.py))
already applied — no manual export step. The dashboard degrades
gracefully if a data file is missing (that tab is just empty), so
steps 4-6 are only needed for the features they feed.

### Dashboard data & the ecoinvent EULA

The static UI shell (`backtest_dashboard.html`, the vendored
`react*.min.js` / `babel.min.js`, `dds-logo.svg`) and the
impact-category notes (`outlier_reasons.json`) ship in git. The
**generated** data files — `dashboard/cf_comparison.csv`,
`dashboard/product_reasons.json`, and `dashboard/decomp/` — do **not**:
they embed ecoinvent elementary-flow nomenclature (flow name +
compartment of the matched registry flow) and are gitignored under the
ecoinvent EULA. Regenerate them locally with steps 4-6.

### Export to Brightway / Activity Browser

After `dds-link-all`, the whole linked system (Agribalyse 3.2 ×
ecoinvent 3.9.1 × the 19 EF v3.1 methods, AWARE corrections included)
can be exported as native Brightway artifacts:

```bash
pip install -e ".[bw,pardiso]"   # bw2calc + bw_processing + the pardiso solver
                                 # (the parity check needs pardiso on this matrix)
dds-build-bw-package         # writes bw_package/ and parity-checks it
```

The export comes in two ready-to-use shapes (see the generated
`bw_package/README.md` for full usage):

1. **`bw_processing` datapackages** (`inventory/`, `methods/<slug>/`) —
   score directly with stock `bw2calc` 2.x, no import step, no project:
   `cd bw_package && python run_example.py`.
2. **A standalone importer** (`import_into_brightway.py`, copied into the
   export) — run it *inside your Brightway / Activity Browser environment*
   to build a named `bw2data` project + databases + methods. Works on both
   Brightway generations (legacy `bw2data` 3.x and `bw2data` 4.x / bw2.5):
   `cd bw_package && python import_into_brightway.py --verify 3`.

Scores are guaranteed to match the pipeline: the export fails unless a
`bw2calc` round-trip over sampled products reproduces the
`NativeLciaScorer` scores, and the importer's `--verify` re-checks a
sample against `metadata/parity_samples.json` using *your* bw2calc.
`--verify` runs fine on plain scipy but is ~10x faster with
`pypardiso` installed in the Brightway environment, and verifies to
1e-6: bw2data stores processed amounts as float32, so
project-recomputed scores carry ~1e-7 quantization — the generated
README documents both points.

> **EULA:** `bw_package/` contains ecoinvent LCI amounts composed from
> your locally regenerated `source/` data. It is gitignored and blocked
> by the pre-commit guard — never commit or redistribute it.

### Decompose a score

`dds-decompose-score` explains *why* a product scores what it scores
for one `(product, method)` pair. It loads the same cached
[`ScoringPackage`](src/scoring/scoring_package.py) that
`dds-backtest` consumes (resolved via the `content_hash` recorded in
`dashboard/run_report.json`) and prints three ranked tables:

1. **Top flow contributions** — biosphere flows ordered by `|inventory_amount × cf|`.
2. **Top activity contributions** — technosphere activities ordered by their total characterised supply.
3. **Top edge contributions** — individual `(activity, flow)` exchanges, the finest grain.

```bash
dds-decompose-score \
    --database agribalyse-3.2 \
    --code 88b91d4e5a9d46b697fd350423bcd087 \
    --method climate \
    --top-n 15
```

Flags:

```
--method      Short alias (climate, cc_bio, cc_fos, cc_luc, ozone,
              radiation, photo_ox, pm, ht_nc, ht_c, acid,
              e_fw, e_m, e_t, ecotox, land, water, energy, mater)
              or a comma-separated full 4-tuple for ad-hoc methods.
--top-n       Rows per table (default: 15).
--inventory   Also dump the top-N uncharacterised inventory flows
              (by |mass|, regardless of CF) — surfaces the
              "right amount, no CF" diagnostic.
--out DIR     Writes decomp_<code>__<method>.json with all three
              tables for downstream tooling; the human-readable
              summary still prints to stdout.
--solver      scipy (default) or pardiso. The post-refactor
              ScoringPackage matrix has 37 zero-diagonal
              placeholder activities — scipy SuperLU rejects them
              as "exactly singular"; use --solver pardiso if the
              decomposition fails with that error.
```

Prerequisite: `dds-link-all` (or `dds-run-end-to-end`) must have run
once so `dashboard/run_report.json` exists and the corresponding
`ScoringPackage` is in `cache/scoring_packages/`. The short method
aliases match the column short-ids in `dashboard/backtest_pass1.csv`,
so the typical loop is "pick a divergent row in the backtest
dashboard → decompose it".

## Repo layout

```
source/      authoritative inputs + the randonneur packages we publish
cache/       parquet caches + importer pickle (gitignored)
registry/    built MappingRegistry parquets — single source of mapping truth
dashboard/   run reports, audit logs, dashboards
to_review/   human-review artifacts (mappings_comparison.xlsx)
unlinked/    residual unlinked exports (technosphere/biosphere)
.bw_projects/ Brightway project state
src/         the package — flat (no redundant src/sentier_agribalyse/ nesting)
docs/        architecture, behavior-change log, refactor spec, test plan
tests/       pytest suite
```

---

# Architecture

## Sources

Everything the linker knows about lives in `source/`. There are no
hardcoded mapping tables, no inline JSON in matchers, no synonym dicts.
Each file below is consumed by exactly one ingester class in
`src/registry/sources/` and aggregated into `registry/*.parquet` by
`dds-build-registry`.

### Inputs (LCI / LCIA payloads)

| File | Origin | What it carries |
|---|---|---|
| `AGB32_final.CSV` | ADEME — Agribalyse 3.2 SimaPro export | The full Agribalyse 3.2 LCI (~507 MB processes-only export). The CSV the `SimaProImporter` parses. Local-only (gitignored). |
| `AGRIBALYSE3.2_reference_synthese_raw.parquet` | ADEME | Reference scores per product × method. Used by `dds-backtest` as the truth set. |
| `EF-LCIAMethod_CF(EF-v3.1)__lciamethods_CF.parquet` | JRC EF v3.1 release | Native EF v3.1 CFs (~320K rows, 89K resolved EF flows). Drives the `ef` biosphere database build and the LCIA method augmentation. |
| `harmonised-flows-simple.json.gz` | Sentier harmonised flow registry | ~1.6M (name, bucket, uuid) entries — the cross-database flow harmonisation backbone. |

### Mapping truth (Sentier / Agribalyse-specific)

| File | Origin | What it carries |
|---|---|---|
| `placeholder_flow_classification.xlsx` | Sentier placeholder workbook | Four sheets that classify every AGB biosphere flow: 806 to match against ecoinvent v3.9.1 (tier 1), 643 to match against EF v3.1 (tier 6), 193 declared unmatchable (tier 12), plus an off-by-default transitive ecoinvent → EF map. |
| `agribalyse-3.2-ecoinvent-3.10-biosphere.json` | Sentier randonneur package | AGB-3.2 → ecoinvent-3.10 biosphere manual matches. |
| `agribalyse-3.2-correct-ecoinvent-edge-labels.json` | Sentier randonneur package | Edge-label corrections (2 123 rows) applied during `EdgeLabelCorrector`. |
| `agribalyse-3.2-delete-aggregated-ecoinvent-{processes,products}.json` | Sentier randonneur packages | The 4 246 aggregated-ecoinvent rows the `AggregateDeleter` strips before linking. |
| `agribalyse-3.2-extra-unit-conversions.json` | Sentier randonneur package (this repo) | m²↔hectare, m↔km conversions the upstream `generic-brightway-unit-conversions` doesn't ship; needed for GLO market-for tillage / fertilising tech edges. |
| `agribalyse-3.2-custom-technosphere-fixes.json` | This repo (`source/randonneur_packages/`) | Tech-edge name patches for the ~17 AGB references that target ecoinvent datasets renamed/retired between 3.9.1 and 3.10. |

### Mapping truth (curated / LLM-assisted)

| File | Origin | What it carries |
|---|---|---|
| `curated_overrides.json` | Hand-authored, dated | Small, dated file replacing the legacy hardcoded `BIOSPHERE_SYNONYMS`. Each row picks its own tier (typically 1 `CURATED_TARGETED` or 11 `CURATED_SYNONYM_FALLBACK`); rows with `is_unmatchable: true` route to `unmatchable.parquet`. |
| `agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx` | LLM suggestions, human-accepted | Only `decision='accept'` rows are loaded as tier 10 fill-only mappings. Gated by `--no-llm`. |

### Bundled randonneur datapackages (consumed via `RandonneurDataLoader`)

These are pulled from the published `randonneur_data` registry — names
referenced by ingester classes in `src/registry/sources/randonneur_packages.py`:

- `agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches` (96 rows) → tier 2.
- `SimaPro-9-ecoinvent-3.9-biosphere-manual-matches` (580 rows) → tier 3.
- `simapro-9-ecoinvent-3-water-slash-m3` (~39 675 rows) → tier 5; doubles as a CAS index source.
- `simapro-9-ecoinvent-3-context` (101 rows) → context normalisation.
- `Flowmapper-standard-units-harmonization` + `generic-brightway-units-normalization` → unit aliases.
- `generic-brightway-unit-conversions` (98 rows) → unit conversions (replaces hardcoded `UNIT_CONVERSIONS`).
- `agribalyse-3.1.1-biosphere-ecoinvent-3.8-biosphere` → known-unmatchable list.

## Project architecture

The package is **OOP everywhere** by design. Every unit of behaviour is
a class with constructor-injected dependencies; configuration is frozen
dataclasses. Layers under `src/`:

```
config/         Paths, Settings (frozen dataclasses)
core/           Logging, StepTimer, ParquetCache, BrightwayProject, IdleHeartbeat
domain/         Tier, Bucket, Mapping, AuditEntry, MatchOutcome (pure data)
readers/        Json/Gz/Xlsx/Parquet readers, RandonneurDataLoader
registry/       RegistryBuilder + MappingRegistry + indexes
registry/sources/  one ingester class per data source
matching/       BiosphereMatcher, TechnosphereMatcher, AuditLog, StrategyRunner
transforms/     SimaProImporter, AggregateDeleter, EdgeLabelCorrector,
                BiosphereFlowmapApplier, ProductionReclassifier,
                BiosphereLabelNormaliser, BioStrategyChain, …
ef/             EfCfTable, EfDatabase, EfMethodAugmenter
scoring/        ProductActivityResolver, SolverConfigurator, LciaScorer
reporting/      MatrixPurger, RunReport, CoverageReporter, UnlinkedExporter
pipelines/      RegistryBuildPipeline, LinkAllPipeline, EndToEndPipeline, BacktestPipeline
exports/        RandonneurPackagesExporter, MappingsComparisonExporter
cli/            BaseCli + 6 concrete CLI classes
```

### The registry is the contract

`registry/` aggregates every authoritative mapping resource into nine
parquets:

| Parquet | Source(s) | Rows |
|---|---|---|
| `mappings_biosphere.parquet` | placeholder workbook (sheets 1.a, 1.b), randonneur packages, harmonised flows, curated, LLM | 1 613 897 |
| `mappings_technosphere.parquet` | (open extension point) | 0 |
| `unmatchable.parquet` | placeholder "Neither" sheet + 3.1.1 unlinked list | 222 |
| `unit_conversions.parquet` | `generic-brightway-unit-conversions` + extras | 98 |
| `unit_aliases.parquet` | `Flowmapper-standard-units-harmonization` + `generic-brightway-units-normalization` | 80 |
| `context_normalisation.parquet` | `simapro-9-ecoinvent-3-context` | 101 |
| `deletions.parquet` | the two `agribalyse-3.2-delete-aggregated-ecoinvent-{processes,products}.json` | 4 246 |
| `edge_label_corrections.parquet` | `agribalyse-3.2-correct-ecoinvent-edge-labels.json` | 2 123 |
| `target_index_ef.parquet` | EF v3.1 CF parquet | 89 070 |
| `registry.meta.json` | source SHA-256 hashes, row counts, tier dictionary | — |

`MappingRegistry.load(settings)` reads them all in one pass. Indexes
(`TieredNameBucketIndex`, `CasIndex`, `UnitConverter`, `UnmatchableIndex`)
build lazily on first access.

### Pipeline sequence

`LinkAllPipeline.run()` orchestrates:

1. `BrightwayProject.setup()` + `load_ecoinvent()`.
2. `MappingRegistry.load(settings)`.
3. `EfDatabase.install()` — only the EF flows the registry actually targets — followed by `EfMethodAugmenter.apply()`.
4. `SimaProImporter.load()` (cached pickle).
5. `AggregateDeleter`, `InternalAgbLinker`, `RestoreSimaproNamesTransform`, `EdgeLabelCorrector`, `BiosphereFlowmapApplier` (with NaN-cf patch), `StandardLabelNormaliser`, `ProductionReclassifier`.
6. `BiosphereLabelNormaliser` + `BioStrategyChain` (bw2io strategy chain; failures recorded by `StrategyRunner` in `dashboard/suppressed_strategies.parquet`).
7. `BiosphereMatcher.match(sp.data)` — walks registry tiers, records every override into `dashboard/override_audit.parquet`.
8. `TechnosphereMatcher.match(sp)`.
9. `UnlinkedExporter.export(sp.data)` → `unlinked/`.
10. `sp.drop_unlinked()` + `sp.write_database()`.
11. `MatrixPurger.purge_to_square()` — single-pass squareness fix.
12. `CoverageReporter` snapshots pre-write + post-purge.
13. `RunReport.write(...)` → `dashboard/run_report.json`.

## Mapping steps and priority

The matcher walks `priority_tier` ascending and selects the
highest-priority row that satisfies type/unit/context constraints.
Tiers 7+ are **fill-only**: they may only place links onto exchanges
that have no prior link. Tiers are *data*, not code, so the documented
ordering cannot drift from the executed ordering.

| Tier | Name | Source | Override? | Notes |
|---:|---|---|---|---|
| 1 | `CURATED_TARGETED` | Placeholder "match with ecoinvent v3.9.1" sheet (806) + `curated_overrides.json` | yes | AGB → ecoinvent biosphere flows — highest authority. |
| 2 | `RANDONNEUR_AGB_SPECIFIC` | `agribalyse-3.1.1-ecoinvent-3.10-biosphere-manual-matches` (96) | yes | Replaces the residual hardcoded `BIOSPHERE_SYNONYMS`. |
| 3 | `RANDONNEUR_SIMAPRO_BIO` | `SimaPro-9-ecoinvent-3.9-biosphere-manual-matches` (580) | yes | Generic SimaPro→ecoinvent biosphere mappings. |
| 4 | `HARMONISED_FLOWS` | `harmonised-flows-simple.json.gz` | yes | Sentier harmonised flow registry. |
| 5 | `RANDONNEUR_WATER_M3` | `simapro-9-ecoinvent-3-water-slash-m3` (~39 675) | yes | Also CAS-derived disambiguation entries (same band, distinguished by `provenance`). |
| 6 | `EF_PLACEHOLDER` | Placeholder "match with EF v3.1" sheet (643) | yes | Routes flow to the `ef` biosphere database (not `biosphere3`). |
| 7 | `EF_GENERIC` | EF parquet (`name, bucket, unit`) lookup | fill-only | Fallback against any EF flow. |
| 8 | `BIO3_MATCH_DATABASE` | bw2io `match_database` chain | fill-only | Standard biosphere3 strategies. |
| 9 | `CASE_INSENSITIVE_FALLBACK` | `(name_lower, unit, bucket)` | fill-only | Deterministic tie-breaker — no `[0]` non-determinism. |
| 10 | `LLM_OVERRIDES` | `agribalyse-3.2-biosphere-residuals-llm-reviewed.xlsx` (`accept` rows) | fill-only | Gated by `--no-llm`. |
| 11 | `CURATED_SYNONYM_FALLBACK` | `curated_overrides.json` rows tagged `synonym` | fill-only | Same `--no-llm` gate as tier 10. |
| 12 | `UNMATCHABLE` | "Neither" sheet (193) + 3.1.1 unlinked list | n/a | Never produces a link; suppresses warnings. |

Source: `src/domain/tier.py`. The tier int is persisted in
`mappings_biosphere.parquet`'s `priority_tier` column.

### Parallel biosphere model: `ecoinvent-3.9.1-biosphere` + `ef`

**Every AGB elementary flow is mapped to one of two target flow sets**,
both of which carry the CFs that the registered EF v3.1 LCIA methods
score against:

1. **`ecoinvent-3.9.1-biosphere`** (the biosphere DB shipped with
   ecoinvent 3.9.1, also exposed under the legacy `biosphere3` name) —
   target for AGB flows with a clean ecoinvent equivalent. CFs against
   these flows are registered by `bw2io` when ecoinvent is imported.
2. **`ef`** — a subset of EF v3.1 elementary flows we build locally
   from the JRC EF v3.1 CF parquet (`EfDatabase.install`). Target for
   AGB flows that have no ecoinvent equivalent. CFs against these
   flows come natively from the EF parquet via `EfMethodAugmenter`.

The matcher's target-DB preference order is: explicit `target_db` from
the registry row → `ecoinvent-3.9.1-biosphere` → `ef` → `biosphere3`
(legacy fallback). Source: `src/matching/biosphere.py:_resolve_target`.

| Registry tier | Link target | How it's characterized |
|---|---|---|
| Tier 1 (`CURATED_TARGETED`) — placeholder ecoinvent sheet (806) + curated overrides | `ecoinvent-3.9.1-biosphere` | CFs registered by bw2io on ecoinvent import |
| Tier 6 (`EF_PLACEHOLDER`) — placeholder EF sheet (643) | `ef` — a subset EF v3.1 flow database we build locally | CFs read natively from the EF v3.1 CF parquet |
| Tier 12 (`UNMATCHABLE`) — "Neither" (193) + 3.1.1 unlinked | (no link) | n/a |

Brightway supports this natively: a single LCIA method can carry CFs
keyed to flows across multiple biosphere databases. Every EF v3.1
method therefore has two CF sets — one against
`ecoinvent-3.9.1-biosphere` (from bw2io), one against `ef` (from the
native CF parquet) — and `bw2calc` characterizes each exchange against
whichever database its input points to.

**Why not bridge all EF flows to ecoinvent biosphere?** The placeholder
"EF v3.1 only" sheet exists precisely for flows with no ecoinvent
equivalent. Bridging them collapses fine-grained toxicity / water
variants onto parent flows and picks up a less-specific CF. Linking
directly to the EF flow preserves the JRC-native CF.

## Scoring

`scoring.LciaScorer` runs factorized LCA across many products × many
methods. Demand is always a *product* activity (resolved by
`ProductActivityResolver`); for every (process, method) we
`switch_method() + lcia_calculation()` — never `lcia()` after the first
product, which would leak the prior characterization matrix. The same
factorized `bw2calc.LCA` is reused across products by passing the
integer node id as the demand key.

### LCIA method registration

Only the **19 headline EF v3.1 methods** are registered (16 main + 3
climate-change sub-indicators). The parquet's organics/inorganics
toxicity splits are not used.

The EF v3.1 CF parquet ships per-location CFs for some methods (e.g.
Acidification has country-specific values). For Brightway's
non-regionalized method object we collapse to one global CF per
(method, flow) using:

1. Prefer the row with `LCIAMethod_location = NULL` (JRC's explicit global value).
2. If no NULL row exists, take the arithmetic mean across regional rows.

Implementation: `src/ef/cf_table.py`, `src/ef/method_augmenter.py`.

### Solver

`SolverConfigurator` forces `--solver scipy` on
`dds-run-end-to-end` / `dds-backtest`; pypardiso fails with -1 on the
AGB+ecoinvent schema mix.

## Run-time artifacts

| Path | Producer | Purpose |
|---|---|---|
| `cache/importer_cache.pkl` | `SimaProImporter` | Cached parsed importer (~5 min saved per re-run) |
| `cache/*.parquet` | `ParquetCache` | Sibling parquets for slow xlsx files |
| `registry/*.parquet` | `RegistryBuilder` | Built mapping registry — regeneratable |
| `registry/registry.meta.json` | `RegistryBuilder` | Source SHA-256 hashes, row counts, build time |
| `dashboard/run_report.json` | `LinkAllPipeline` | Per-stage stats, coverage snapshots, drop totals |
| `dashboard/override_audit.parquet` | `AuditLog` | Every match decision (new link / override / unit reject / ambiguous skip) |
| `dashboard/suppressed_strategies.parquet` | `SuppressedStrategyLog` | bw2io strategies that threw |
| `dashboard/backtest/*.parquet` | `BacktestPipeline` | scores / diff_abs / diff_pct / summary |
| `dashboard/backtest_pass1.csv` | `BacktestPass1Emitter` | Dashboard **Product %diff** data source — 19 method short IDs per mapped product |
| `dashboard/cf_comparison.csv` | `CfComparisonCsvEmitter` (`dds-compare-cfs`) | Dashboard **CF comparison** data source — SimaPro-vs-ours matched per-flow CFs. Gitignored (embeds ecoinvent nomenclature). |
| `dashboard/decomp/<code>.json` | `dds-build-flow-decomp` | Per-product flow-decomposition for the drill-down panel. Gitignored. |
| `dashboard/product_reasons.json` | `dds-build-product-reasons` | Per-product LLM outlier notes for cell tooltips. Gitignored. |
| `dashboard/backtest_dashboard.html` | hand-maintained | Static React UI (vendored react/babel, `dds-logo.svg`); serve with `python -m http.server --directory dashboard` |
| `unlinked/technosphere_unlinked.json` | `UnlinkedExporter` | Residual unlinked technosphere names |
| `unlinked/biosphere_unlinked.xlsx` | `UnlinkedExporter` | Residual unlinked biosphere flows |
| `to_review/mappings_comparison.xlsx` | `MappingsComparisonExporter` | Placeholder ecoinvent / EF / Neither / novel comparison |
