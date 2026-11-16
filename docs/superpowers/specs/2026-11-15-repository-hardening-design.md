# Repository Hardening Design

## Intent

Make the current repository safe to continue using after the Submission 2 merges by adding automated verification, repairing reproducibility gaps, clarifying the supported workflow, and aligning evaluation reporting with the order-level prediction question. Preserve the frozen model, threshold, and reported final-test results.

## Goals

- Run the complete test suite automatically for pushes and pull requests.
- Fail early with a clear explanation when the raw CSV differs only because of CRLF line endings or is genuinely the wrong source file.
- Make the declared Conda environment reproduce the versions used for the tracked evaluation artifacts, including Plotly and SciPy.
- Add a lightweight environment/data preflight command with automated tests.
- Update the README so modeling, evaluation, and reproduction commands match the repository's actual state.
- Add supplemental unique-order metrics for the already-frozen validation and test predictions without selecting or changing the model, features, or threshold.
- Normalize and verify the committed numerical notebook execution state without changing its analysis source.

## Non-goals

- No final course report or presentation.
- No model retraining, feature expansion, hyperparameter search, calibration, or threshold change.
- No reinterpretation of the held-out test set as available for further tuning.
- No raw or processed customer data committed to Git.
- No rewriting or deleting existing Git history.

## Constraints

- The final test has already been scored. New order-level metrics are post-hoc, supplemental reporting only.
- The original checkout contains an unrelated uncommitted notebook rerun and must remain untouched.
- CI cannot use the ignored raw dataset, so it will run synthetic/unit tests and structural validation only.
- Branch-protection settings are a repository administration change performed after the PR exists and CI is green.

## Approaches

1. **Single hardening PR (selected):** CI, preflight validation, exact environment versions, documentation, notebook normalization, and supplemental order-level reporting together. This gives one auditable correction point and one review/check cycle.
2. **Separate infrastructure and evaluation PRs:** cleaner individual diffs but creates ordering dependencies and leaves the repository partially hardened between merges.
3. **Minimal CI/documentation patch:** fastest, but leaves the raw-file failure, dependency drift, and item/order reporting mismatch unresolved.

## Proposed Implementation

### Automated safeguards

- Add a GitHub Actions workflow using the declared Python/Conda-compatible dependency versions.
- Validate notebook JSON and tracked JSON artifacts, run `git diff --check`, and run all tests.
- Document the exact branch-protection settings to enable after merge: require pull requests and the CI check, and block force pushes to `main`.

### Reproducibility and preflight

- Pin the exact Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter, pytest, Plotly, and SciPy versions used or verified by the project.
- Add a read-only preflight module that reports missing files/packages, distinguishes canonical SHA-256 from CRLF-only differences, verifies processed-data structure when present, and exits nonzero for actionable problems.
- Test the preflight logic entirely with synthetic temporary files.

### Supplemental order-level evaluation

- Extend final evaluation reporting to aggregate one prediction and target per `order_id` only after verifying target, features, and score are constant within each order.
- Report validation and final-test order-level AP, precision, recall, alert counts, and confusion counts alongside—not instead of—the frozen item-level results.
- Keep the existing item-level rows and frozen configuration unchanged.
- Mark the order-level results as post-hoc supplemental metrics and prohibit their use for model or threshold changes.

### Documentation and notebooks

- Update the README with the completed Submission 2 workflow, final frozen results, preflight/reproduction commands, limitations, and links to evaluation documents.
- Re-execute the numerical notebook in the declared environment, verify zero errors and unchanged source cells, and commit deterministic execution metadata/output only if the generated figures remain consistent.
- Leave the preprocessing notebook intentionally unexecuted unless execution is required to validate a changed code path.

## Verification

- Run the full unit suite.
- Run preflight against synthetic valid, CRLF-only, wrong-content, and missing-file cases.
- Reproduce all evaluation runners against the canonical raw source in an isolated temporary copy and compare key metrics/counts with tracked artifacts using explicit tolerances.
- Execute the numerical and categorical notebooks from clean copies with zero errors.
- Validate workflow YAML, notebook JSON, report JSON/CSV, and `git diff --check`.
- Review the final branch diff against `origin/main` before push and PR creation.

## Success Criteria

The hardening PR is mergeable, CI is green, all local tests and reproduction checks pass, the README reflects the actual project, order-level supplemental metrics are clearly non-tuning results, and no frozen model decision or original user work is altered.
