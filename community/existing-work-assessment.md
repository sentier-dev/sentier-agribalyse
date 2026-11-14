# Existing AGRIBALYSE work and remaining community layer

Assessed 7 October 2026 from the public repository and its public PRs.

The existing project already supplies the canonical version-specific import, tiered MappingRegistry, curated overrides, reversible parameter scenarios, backtesting, dashboard and parity-checked Brightway export. It is therefore the central project to extend, not a dependency for a separate feedback-only repository.

[PR 4](https://github.com/sentier-dev/sentier-agribalyse/pull/4) reports parameter editing in response to an ADEME what-if request. Its checklist explicitly defers the full real-data parameter rescore; fixture tests and reported checks must not be upgraded to independent real-data verification. [PR 2](https://github.com/sentier-dev/sentier-agribalyse/pull/2) reports Brightway parity/export checks. These public reports are useful implementation evidence, not proof of new partner consensus.

The Brightcon BAFU toolkit is a distinct contribution in sentier-models PR 2. No AGRIBALYSE hackathon minutes or newly approved partner dataset corrections have been supplied. Gather those records here, attach candidate implementations and tests, and retain dissent and provider responses alongside the canonical import.

This change adds the public contribution lifecycle and comparison tooling around existing import/application hooks. It does not implement a new importer, alter dataset scores or assert that a 4.0 release is public.
