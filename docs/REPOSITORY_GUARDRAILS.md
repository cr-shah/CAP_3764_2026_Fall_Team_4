# Repository Guardrails

This project uses pull requests and the `verify` GitHub Actions check to prevent another unreviewed or broken change from landing directly on `main`.

## Automated verification

The workflow at `.github/workflows/verify.yml` runs for every pull request and every push to `main`. It creates the exact Conda environment, checks changed files for whitespace errors, parses every tracked notebook and JSON report, and runs the complete synthetic/unit test suite. The ignored DataCo dataset is deliberately not required in CI.

The local data and environment diagnostic remains available separately:

```bash
python -m src.project_preflight
```

## Protect `main` after this PR is green

An administrator should create an active branch ruleset in **Settings → Rules → Rulesets → New ruleset → New branch ruleset**:

1. Name it `Protect main`, set enforcement to **Active**, and target the default branch (or include `main`).
2. Enable **Require a pull request before merging** with at least one required approval.
3. Enable **Require status checks to pass before merging** and select the `verify` check after it has completed successfully on this pull request. Require the branch to be up to date before merging.
4. Enable **Require conversation resolution before merging**.
5. Leave **Block force pushes** enabled and do not permit branch deletion.
6. Avoid broad bypass actors. If an emergency bypass is retained for the repository owner, treat every use as an incident that requires a follow-up pull request.

Do not activate a required status check before GitHub has seen the check run at least once. Otherwise the required check may not be selectable or could leave merges permanently pending.

## Normal contribution flow

```bash
git switch main
git pull --ff-only
git switch -c feature/short-description
# make and test changes
git push -u origin feature/short-description
```

Open a pull request, wait for `verify`, resolve review conversations, and merge through GitHub. Direct pushes, force pushes, and history rewrites on `main` are outside the supported workflow.
