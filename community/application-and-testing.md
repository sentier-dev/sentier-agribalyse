# Apply and test AGRIBALYSE contribution layers

## Preserve a reproducible baseline

Use the official release and the existing import path. Record the source hashes, repository revision, licensed background release, methods, current curated mappings and active parameter overrides. Retain baseline run_report, scoring package and backtest. The existing importer already includes transforms and mappings: do not label its outputs an untouched vendor import.

Use separate baseline and candidate working checkouts with independent local source/cache/registry/output folders. Obtain licensed inputs under their terms. Never use shared mutable caches between these builds.

## Existing contribution hooks

- **Parameter scenarios:** dds-set-parameter writes reversible overrides; dds-list-parameters and dds-clear-parameters inspect and roll them back. Apply only in the candidate checkout. Example from the existing README: `uv run dds-set-parameter Packaging_Weight 0.03 --product <process-code>`. This command relinks/rescores; use --no-rescore for staging only, followed by explicit linking and backtesting. New scenario values are not automatically corrections to ADEME's canonical inventory.
- **Curated biosphere mappings:** registry/sources/curated.py reads source/curated_overrides.json into the existing MappingRegistry. In the candidate checkout, retain baseline entries and add reviewed proposal entries with identity, context, units, conversion, tier and evidence. Rebuild the registry, relink and backtest. Do not build a second matcher or assume identical factors make distinct chemical species equivalent.
- **Technosphere fixes:** retain the existing randonneur package format and public custom-technosphere-fixes migration. Review ordering and target identity before applying a change.
- **Application/export:** use dds-build-bw-package and its parity verifier to check the candidate against stock bw2calc. Compare baseline and candidate results under equivalent inputs, boundaries and methods. Keep unresolved exchanges and explain expected changes.

--no-llm disables both LLM overrides and curated synonym fallback; it is not a neutral comparison flag. Hold it fixed across runs or explicitly treat its change as part of the candidate. dds-reset deletes overrides unless --keep-overrides is given; use it only within an isolated build.

## Validation and consensus

Attach parameter/flow-level regression checks, counts, unresolved links, expected score deltas and equivalent-scope backtests to the contribution. Normalize public aggregate summaries using community/templates/run-summary.json, and compare them with scripts/community.py. Private inventories stay local; authorized aggregate results and reproducible metadata support public review.

No source import or new 3.2 scores were computed as part of adding this workflow. It documents the existing hooks and provides coordination/comparison checks; missing licensed inputs must remain explicit. Partner approval and dataset-provider response require their own evidence.
