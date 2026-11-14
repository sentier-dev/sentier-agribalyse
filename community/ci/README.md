# Documentation and contribution CI

The [Documentation workflow](../../.github/workflows/pages.yml) validates contribution records, runs the synthetic community tests and build-audit checks, then builds and checks the public documentation site. Pull requests never deploy. Updates to `main` publish only the generated `_site/` artifact through GitHub Pages, once a repository administrator has enabled **Settings → Pages → Source: GitHub Actions**.

Run the same contribution checks locally with `python3 scripts/community.py check`, `python3 -m unittest discover -s community/tests` and `python3 scripts/build_checks.py community/templates/build-audit.json`. Site maintenance and hosting links are documented in [site/README.md](../../site/README.md) and the main README.

The supplied `github-actions.yml` remains a standalone contribution-check template; the Documentation workflow already runs those checks. The portable `.gitlab-ci.yml` provides equivalent validation and Pages publication for a GitLab mirror.

The existing pytest CI also collects the community record and build-audit regressions through `tests/unit/test_community_records.py`. This keeps the collaboration checks connected to the adapter suite.
