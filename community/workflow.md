# Contribution lifecycle

1. **Capture:** link a partner observation, hackathon contribution, source report or reproduction. Record release, source format, background, method, functional unit, allocation, geography and boundary. Publish authorized summaries rather than confidential transcripts.
2. **Propose a layer:** use a contribution record to pin the baseline, contribution implementation and payload, ownership, intended effect, required checks, reviewers and recipient project. Existing implementation hooks are used rather than a second matching engine.
3. **Apply and test:** run baseline and candidate separately with identical source/background/method pins. Test integrity and linkage, source and unit identity, exchange completeness, scope, changed outputs and LCIA parity as appropriate. Preserve residuals and explain intended changes. Record missing inputs and skipped checks.
4. **Review:** contributors and affected partners examine the evidence and dissent. All required reviewers must approve through linked public records. Every required validation check must have a passed result with evidence; incomplete runs cannot be promoted. Automated record checks verify documentation, not the truth of an approval or scientific result.
5. **Return consensus:** prepare an evidence-backed recommendation for BAFU or ADEME. Distinguish upstream dataset defects, methodological improvements, local scenarios and importer defects. Send only with authorization; link delivery and responses and track implementation or reopening.

Contribution status: proposed → testing → reviewing → accepted / rejected / superseded. A useful scenario need not become an official correction. A merged importer PR alone is not dataset-provider endorsement.

No owners, reviewers or provider contacts are assigned without confirming their participation. An assistant's synthesis remains a proposal. Consensus is represented by explicit approval records with objections and evidence, not a vote count or silence.

## Bind review to the tested candidate

Each approval includes `reviewer`, `evidence_url`, the complete `baseline` object and `layer_revision`. Acceptance checks these against the contribution record, so approval of an earlier revision cannot promote a changed candidate. A link records the actual reviewer decision; automatic validation does not manufacture it.
