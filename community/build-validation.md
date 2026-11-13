# Evidence and build validation

An import that links every exchange can still score incorrectly. Use three separate checks: **exchange reconciliation**, **characterization coverage**, and **equivalent-scope result comparison**. These checks transfer the engineering lessons from [lci-bafu-catalog's importer-parity discussion](https://github.com/sentier-dev/sentier-bafu/issues/2) and [method/build-identity findings](https://github.com/sentier-dev/sentier-bafu/issues/5).

## Account for every changed row

Report process, production, technosphere and biosphere counts at the source, imported and served stages. Each change needs a unique id, transition, exchange kind, signed row-count delta, reason and evidence link. The audit fails if the measured delta differs from the recorded changes. An equal total alone does not establish parity: compare stable exchange identities, amounts, units, compartments and declared supplier locations as well.

Keep declared supplier geography separately from resolved target geography. Stable dataset identifiers should determine links where available; a link resolved by identifier does not erase contradictory source metadata. Preserve raw lifecycle and aggregation signals rather than burying them in comments or assuming a sector folder captures retirement.

AGRIBALYSE already records pre/post link coverage and matrix drops through its dangling-edge auditor. Keep those reports, the curated mapping revisions, parameter overrides and export-parity results with each candidate. The BAFU issue counts are not AGRIBALYSE thresholds; reproduce AGRIBALYSE's own source and served exchange identities.

## Check characterization, not just linking

For each method, list the emitted flow keys, the flow keys required by its documented policy, the actual factors and explicit exclusions with evidence. Use complete identities (database/code, compartment, unit and region as appropriate), not substance name alone. **A supplied zero factor counts as characterized; an absent factor does not.** Never infer fossil versus non-fossil methane or copy an AR6 factor into EF 3.1 without the method source.

The required-flow list must come from the measured final inventory and a cited method policy. This tool checks the submitted audit; it cannot prove that a manually supplied inventory list is complete. Named exception evidence is still subject to review.

```sh
python3 scripts/build_checks.py path/to/build-audit.json
```

Start with [the synthetic audit example](../community/templates/build-audit.json). It demonstrates the format; its values and hashes are synthetic and certify no dataset build. A failed audit exits nonzero.

### Integrated AGRIBALYSE audit

The scoring-package emission path now audits the **final** biosphere rows against the method CF tables and catalog. It records each method's characterized count, explicit-zero count, missing ids and missing core gases in `run_report.json` under `stages.characterization_audit`. For climate-change methods it stops before emitting a package if a used CO2/CH4/N2O air-flow variant in the explicit core-gas policy has no source factor, or a used flow lacks catalog identity. Non-finite and duplicate CF entries also fail.

This is a coverage gate, not a new method derivation: it assigns no factors, changes no source amounts, and leaves all other missing rows visible for method-specific review. It does not cover every possible greenhouse gas or validate CF units and scientific correctness. The current adapter's existing complete input hash already includes CF tables and corrections; retain that exact identity alongside the portable summary.

Regression tests use synthetic CO2, methane and ammonia flows. The test demonstrates that unqualified methane cannot silently become zero and that explicit biogenic CO2 zero remains valid. Licensed inventories were not used to claim a new baseline reproduction.

## Pin the inputs and the outputs

Keep source hashes and implementation revisions alongside actual SHA-256 hashes of output files. Hash the complete file bytes locally, for example with `shasum -a 256 path/to/output`.

The audit emits a **summary_sha256** over the dataset, scope, counts, changes, characterization audit, probes and applied packages. It ignores timestamps, serving locations and importer revisions, canonicalizes probe floats to 12 significant digits, and sorts unordered identity lists. That makes aggregate summaries comparable across rebuilds with negligible floating-point noise. It is a summary identity, **not a hash of the complete inventory** and not proof of numerical equivalence. The exact output-file hashes remain separate and may differ across platforms. Record score tolerances explicitly in the application test.

## Validate the contribution's actual claim

- **Import or mapping repair:** reproduce source identity and amounts; explain every added, removed, converted or redirected exchange; compare coverage and scores.
- **Representation/compression:** demonstrate lossless reconstruction within stated tolerances across the complete claimed columns, and score parity. A successful family round-trip alone is not an LCIA validation.
- **Disaggregation/reconstruction:** give the new inventory its own identity. Compare against the aggregated original; even a high mass-coverage ratio does not imply matching impacts.
- **Scenario or forecast:** hold scope fixed and record assumptions, uncertainty and held-out validation. A scenario is not automatically a correction of the provider's source dataset.

Publish only permitted aggregate evidence and metadata. Inventories, licensed background files and confidential partner evidence stay local. A passing technical check is not partner consensus; [the review workflow](workflow.md) records those decisions separately.
