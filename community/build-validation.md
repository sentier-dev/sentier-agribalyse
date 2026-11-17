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

The scoring-package emission path audits the **final** biosphere rows against the method CF tables and catalog. For the headline climate-change method (a method key with the part `climate change`; the fossil, biogenic and land-use sub-indicators are not audited separately) it stops before emitting a package if a used CO2/CH4/N2O air-flow variant in the explicit core-gas policy has no source factor or a factor of zero, or a used flow lacks catalog identity. Non-finite and duplicate CF entries and duplicate catalog rows also fail. The receipt says `not_audited` rather than `passed`, with the reason in `not_audited_reason`, when the registry holds no such method or when no emitted flow was recognised as a core gas: each climate method records `core_gas_flows_checked`, and a pass needs it above 0. The package is still emitted in both cases, with a warning, since nothing was found wrong; but nothing was certified either.

**A zero the method defines is not a missing factor.** EF 3.1 scores biogenic CO2 emissions to air as zero. The JRC table says so with explicit zero factors, but on its own flow identities; the ecoinvent flow the inventory emits (`Carbon dioxide, non-fossil`) has no row of its own. The audit reports such a flow under `zero_by_method_definition`, apart from `missing_core_gases`, and only while three things hold: the policy names the species, the catalog knows the method's flows for it, and every one of them carries a factor of exactly 0 in that method. If the method's table omits the species, lists it incompletely or gives it a factor, the flow is a missing factor again and the audit fails. Fossil CO2, methane and N2O are never excused this way, and a factor row that says 0 for one of them fails the audit as well (`zero_factor_core_gases`): it would score the gas zero as surely as no row.

The receipt is written to `dashboard/characterization_audit.json` for every candidate whose audit was made, passed or failed, with the candidate's content hash. An audit that cannot be made and raises, because the catalog or a factor table is inconsistent, writes no receipt: the earlier one stays, and its content hash says which candidate it describes. It holds counts per method (characterized, explicit zero, uncharacterized; for the climate method also the core-gas flows it checked) and names only the core gases; it does not list the id of every flow a method leaves uncharacterized, which is most flows in most methods. When the audit fails, no package is written and `run_report.json` is left exactly as the last successful run wrote it, so every command that follows it still finds the last emitted package. When it passes or reports `not_audited`, the same counts are recorded in `run_report.json` under `stages.characterization_audit`.

This is a coverage gate, not a new method derivation: it assigns no factors and changes no source amounts. It does not cover every possible greenhouse gas or validate CF units and scientific correctness. The current adapter's existing complete input hash already includes CF tables and corrections; retain that exact identity alongside the portable summary.

Regression tests use synthetic CO2, methane and ammonia flows: unqualified methane cannot silently become zero, an explicit biogenic CO2 zero remains valid, and a method-defined zero is accepted only with the method's own zero rows. One test reads the public JRC table in `source/` and runs in CI. Further tests run against the built registry where a licensed user has it and are skipped elsewhere. No new baseline reproduction is claimed.

## Pin the inputs and the outputs

Keep source hashes and implementation revisions alongside actual SHA-256 hashes of output files. Hash the complete file bytes locally, for example with `shasum -a 256 path/to/output`.

The audit emits a **summary_sha256** over the dataset, scope, counts, changes, characterization audit, probes and applied packages. It ignores timestamps, serving locations and importer revisions, canonicalizes probe floats to 12 significant digits, and sorts unordered identity lists. That makes aggregate summaries comparable across rebuilds with negligible floating-point noise. It is a summary identity, **not a hash of the complete inventory** and not proof of numerical equivalence. The exact output-file hashes remain separate and may differ across platforms. Record score tolerances explicitly in the application test.

## Validate the contribution's actual claim

- **Import or mapping repair:** reproduce source identity and amounts; explain every added, removed, converted or redirected exchange; compare coverage and scores.
- **Representation/compression:** demonstrate lossless reconstruction within stated tolerances across the complete claimed columns, and score parity. A successful family round-trip alone is not an LCIA validation.
- **Disaggregation/reconstruction:** give the new inventory its own identity. Compare against the aggregated original; even a high mass-coverage ratio does not imply matching impacts.
- **Scenario or forecast:** hold scope fixed and record assumptions, uncertainty and held-out validation. A scenario is not automatically a correction of the provider's source dataset.

Publish only permitted aggregate evidence and metadata. Inventories, licensed background files and confidential partner evidence stay local. A passing technical check is not partner consensus; [the review workflow](workflow.md) records those decisions separately.
