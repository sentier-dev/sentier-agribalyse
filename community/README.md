# Community contributions and consensus

This public project is the central working space for AGRIBALYSE, spanning dataset releases. Its purpose is to make the canonical import easy to reproduce, collect contributions and discussions from partners and hackathon participants, apply proposed improvements as traceable layers, and test them against the canonical baseline.

The resulting evidence supports an explicit review and consensus process. Agreed findings become actionable recommendations to ADEME for improving the source dataset, with delivery and provider responses recorded here. Sentier provides the collaboration and testing infrastructure; the recommendations concern AGRIBALYSE.

The working sequence is: canonical import → contributor layers → application and testing → reviewed consensus → recommendations to ADEME.

Keep the canonical source and its import reproducible. Store each proposed correction, mapping, parameter scenario or process model as a separate contribution with a source release, pinned implementation revision, evidence and validation plan. Apply it in an isolated candidate build; retain the baseline and comparison outputs. A local experiment does not change the provider's official dataset.

Start with [the contribution template](templates/contribution.json), [the intake template](templates/intake.md), [application and testing](application-and-testing.md), [build validation](build-validation.md), and [the workflow](workflow.md). A record lives in `community/contributions/<id>.json`: the file is named after its `id` (letters, digits, `.`, `-`, `_`), and every record needs an id and a title, because the site titles and links each card with them. Evidence, proposals, runs and provider responses belong here together; this is more than a collection of feedback notes.

```sh
python3 scripts/community.py check
python3 scripts/community.py compare baseline-summary.json candidate-summary.json --out .local/comparisons/example.json
```

Comparisons require the same baseline identity and scope, matching metrics and finite values. They report deltas; they do not decide whether a change is scientifically correct. Acceptance requires actual validation evidence and named reviewer approvals.

No dataset-provider endorsement or new partner consensus is assumed. No packet is automatically sent. Public collaboration records must not expose confidential discussions or restricted source exports.
