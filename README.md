<p><img src="site/assets/sentier.svg" alt="sentier.dev" width="160"></p>

# AGRIBALYSE · a shared dataset workspace

**Reproduce the import. Bring contributions into application and testing. Build reviewed consensus and return actionable recommendations to ADEME.**

This public Sentier project combines the existing AGRIBALYSE adapter with a central space for partner evidence, hackathon contributions, candidate improvements and source-dataset discussions. The project spans releases; its current documented adapter imports **AGRIBALYSE 3.2** with **ecoinvent 3.9.1** and **EF 3.1**. Public participation requires no private or unreleased adapter.

[Import & use](IMPORT.md) · [Findings & shared lessons](community/findings.md) · [Evidence & testing](community/build-validation.md) · [Contribute](community/README.md) · [Consensus](community/workflow.md)

## Start with the existing adapter

The adapter imports ADEME's SimaPro CSV, links through a tiered MappingRegistry, registers the 19 headline EF 3.1 methods, backtests against ADEME reference results and exports a parity-checked Brightway package. The detailed commands, prerequisites, solver setup, parameter tools and architecture remain in [the import and application reference](IMPORT.md).

```sh
git clone https://github.com/sentier-dev/sentier-agribalyse.git
cd sentier-agribalyse
uv sync --extra test
```

Obtain the source export and licensed background inputs locally; follow [BOOTSTRAP.md](BOOTSTRAP.md) and [IMPORT.md](IMPORT.md). Source snapshots, background-derived CF tables, scoring caches and Brightway exports remain gitignored; the pre-commit guard also refuses the ones holding ecoinvent amounts, CFs or the dataset list. The code licence does not relicense those artifacts.

## Bring evidence, then test the claim

| Step | What belongs here |
|---|---|
| Canonical import | Source release, hashes, background version and reproducible adapter path |
| Partner contributions | Findings, curated mappings, parameter scenarios and process models |
| Application & testing | Isolated candidates, named exchange changes, coverage and equivalent-scope comparisons |
| Reviewed consensus | Actual reviewer decisions, objections and their disposition |
| Recommendations to ADEME | Agreed findings, delivery evidence and provider responses |

A **source finding** needs reproduction against the named AGRIBALYSE release. An **adapter defect** needs evidence of the transformation that introduced it. A **scenario** needs assumptions and application tests; it is not automatically a correction of ADEME's data. [Open a finding or discussion](https://github.com/sentier-dev/sentier-agribalyse/issues/new/choose), or start with [a contribution record](community/templates/contribution.json).

Reuse the existing MappingRegistry, parameter overrides, dangling-edge auditor, backtests and export parity verifier. The [application guide](community/application-and-testing.md) explains how to retain separate baseline and candidate builds.

## Shared lessons from the BAFU workspace

The [review ledger](community/findings.md) brings relevant lci-bafu-catalog lessons into AGRIBALYSE:

- Account for source, imported and served exchange counts through named changes.
- Audit final biosphere characterization separately from successful linking: a flow that links but has no factor scores zero without an error. `scripts/build_checks.py` checks this on a submitted build audit; a check inside the adapter's emit stage is proposed in a separate pull request.
- Keep exact input/output hashes alongside a portable aggregate-summary identity. The summary is not a complete inventory fingerprint.
- Give reconstructed inventories separate identities and evidence. BAFU-specific values, prefixes and factor choices are not copied into AGRIBALYSE.

```sh
python3 scripts/community.py check
python3 -m unittest discover -s community/tests
python3 scripts/build_checks.py path/to/build-audit.json
uv run pytest -q
```

Read [build validation](community/build-validation.md) for the checks' precise scope and limitations. These additions are reviewed through [PR #6](https://github.com/sentier-dev/sentier-agribalyse/pull/6); synthetic tests do not establish a newly reproduced licensed baseline or new partner consensus.

## Documentation site

**[Open the AGRIBALYSE review preview](https://mklarmann.github.io/sentier-agribalyse/).** The responsive site shares Sentier's visual identity with [the BAFU / UVEK site](https://sentier-dev.github.io/sentier-bafu/). Its public pages cover import, findings, application tests and consensus. The preview displays a review notice and links to the proposed source in [PR #6](https://github.com/sentier-dev/sentier-agribalyse/pull/6).

The [Documentation workflow](.github/workflows/pages.yml) checks pull requests and is ready to publish from `main` at `https://sentier-dev.github.io/sentier-agribalyse/`. The canonical site becomes available after a maintainer merges this PR and enables **Settings → Pages → Source: GitHub Actions**. Build locally with `uv run --no-project --with markdown==3.7 scripts/build_site.py`; see [site maintenance](site/README.md). Generated pages contain coordination material only, with no licensed inventory downloads.

## Data, credit and participation

Code: MIT. ADEME source material and ecoinvent backgrounds retain their applicable terms; [data and attribution](community/data-and-attribution.md) explains the artifact distinctions. Licensed inputs and confidential evidence stay local. Reviewer agreement and provider responses require evidence in [the workflow](community/workflow.md) and [recommendation tracker](community/recommendations/tracker.json).
