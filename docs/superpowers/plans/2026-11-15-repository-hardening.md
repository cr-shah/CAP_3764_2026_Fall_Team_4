# Repository Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the merged repository reproducible, automatically verified, operationally documented, and aligned with its order-level prediction question without changing the frozen model or original final-test result.

**Architecture:** Add two isolated modules: a read-only project preflight checker and a post-hoc order-level evaluator. Keep all files hashed by `FINAL_MODEL_CONFIG.json` unchanged. Add CI/configuration and documentation around those modules, then verify notebooks and every existing evaluation runner from a clean snapshot.

**Tech Stack:** Python 3.11.18, pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.5.2, SciPy 1.17.1, pytest 8.3.5, Conda, GitHub Actions, Jupyter/nbconvert.

**Spec:** `docs/superpowers/specs/2026-11-15-repository-hardening-design.md`

## Global Constraints

- Do not modify any path listed under `implementation_sha256` or `reference_report_sha256` in `docs/FINAL_MODEL_CONFIG.json`.
- Do not change model inputs, estimator settings, training data, threshold, or existing final-test result.
- Treat new order-level metrics as post-hoc supplemental reporting only.
- Do not commit the raw or processed DataCo CSV.
- Preserve the original checkout's uncommitted numerical-notebook rerun.
- Use TDD for every new Python behavior: test, observe the expected failure, implement minimally, then run the full suite.

## Review Focus

- A CRLF-only raw CSV must be identified separately from genuinely incorrect content; Task 1 tests both cases.
- Preflight must not rewrite user data or fail merely because the optional processed CSV is absent; Task 1 tests read-only and missing-file behavior.
- Multiple items in one order must have identical target, features, and score before aggregation; Task 2 tests every inconsistency.
- Order-level metrics must use one order once and preserve the frozen `>= 0.50` rule; Task 2 tests literal confusion counts and threshold equality.
- The supplemental runner must verify the frozen configuration and tracked final results before scoring; Task 2 tests contract drift rejection.

---

### Task 1: Read-only project preflight

**Files:**
- Create: `src/project_preflight.py`
- Create: `tests/test_project_preflight.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`, `sha256_normalized_newlines(path: Path) -> str`, `inspect_raw_file(path: Path, expected_sha256: str) -> dict[str, Any]`, `inspect_runtime() -> dict[str, Any]`, `run_preflight(raw_path: Path | None = None, processed_path: Path | None = None) -> dict[str, Any]`, and CLI `python -m src.project_preflight`.

- [ ] **Step 1: Write failing tests for canonical, CRLF-only, wrong, and missing raw files**

Use literal byte fixtures and expected statuses `ok`, `crlf_only_mismatch`, `content_mismatch`, and `missing`; verify files remain byte-identical after inspection.

- [ ] **Step 2: Run the new test file and verify failure because `src.project_preflight` is missing**

Run: `python -m pytest tests/test_project_preflight.py -q`

- [ ] **Step 3: Implement raw-file inspection minimally**

Stream bytes for both hashes; normalize only CRLF pairs for the diagnostic hash and never write the input file.

- [ ] **Step 4: Add failing tests for runtime/package reporting and optional processed-data structural checks**

Assert required package presence/version fields; processed absence is informational, while a present CSV lacking `order_id` or `late_delivery_risk` fails.

- [ ] **Step 5: Implement runtime and processed-data checks plus JSON CLI output/exit status**

Exit zero only when required runtime packages are importable and the raw source is canonical; include an actionable CRLF normalization message without modifying the file.

- [ ] **Step 6: Run targeted and full tests**

Run: `python -m pytest tests/test_project_preflight.py -q && python -m pytest -q`

- [ ] **Step 7: Commit**

Commit message: `Add read-only project preflight checks`

### Task 2: Supplemental unique-order evaluation

**Files:**
- Create: `src/order_level_evaluation.py`
- Create: `tests/test_order_level_evaluation.py`
- Create: `reports/order_level_evaluation.json`
- Create: `reports/order_level_evaluation.csv`
- Create: `docs/ORDER_LEVEL_EVALUATION.md`

**Interfaces:**
- Consumes: frozen configuration loader and model/scoring helpers without modifying their source.
- Produces: `aggregate_orders(data: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame`, `score_orders(orders: pd.DataFrame, partition: str, threshold: float) -> dict[str, Any]`, and `run_order_level_evaluation(...) -> dict[str, Any]`.

- [ ] **Step 1: Write failing aggregation tests**

Use literal multi-item orders; assert exactly one row per order and rejection when an order has inconsistent target, either feature, timestamp, or model score.

- [ ] **Step 2: Run targeted tests and verify failure because the module is missing**

Run: `python -m pytest tests/test_order_level_evaluation.py -q`

- [ ] **Step 3: Implement minimal verified aggregation**

Require nonmissing `order_id`; verify invariants with `groupby(...).nunique(dropna=False)` before selecting the first identical row.

- [ ] **Step 4: Write failing metric tests**

Assert hand-calculated AP, precision, recall, TP/FP/FN/TN, alert rate, and `>=` threshold equality with each order weighted once.

- [ ] **Step 5: Implement order scoring using existing metric definitions**

Return the same naming conventions as final item-level results plus `unit: unique_order` and `post_hoc_supplemental: true`.

- [ ] **Step 6: Write failing runner-contract tests**

Assert rejection of changed frozen configuration, mismatched tracked item-level counts, or altered model fingerprint/validation results before supplemental test reporting.

- [ ] **Step 7: Implement the runner without changing frozen files**

Load and verify the frozen configuration, recreate the training-only tree, reproduce tracked validation/test item-level metrics, then aggregate existing predictions by order and write separate JSON/CSV outputs.

- [ ] **Step 8: Run targeted and full tests**

Run: `python -m pytest tests/test_order_level_evaluation.py -q && python -m pytest -q`

- [ ] **Step 9: Generate canonical reports and write the interpretation document**

Run against the canonical raw source; document that results are post-hoc, cannot justify retuning, and complement rather than replace item-weighted metrics.

- [ ] **Step 10: Commit**

Commit message: `Add supplemental order-level evaluation`

### Task 3: Exact environment and automated GitHub checks

**Files:**
- Modify: `environment.yml`
- Create: `.github/workflows/verify.yml`
- Create: `docs/REPOSITORY_GUARDRAILS.md`

**Interfaces:**
- Consumes: `python -m src.project_preflight` and the complete pytest suite.
- Produces: GitHub check named `verify` and exact branch-protection instructions.

- [ ] **Step 1: Pin verified dependency versions**

Use Python 3.11.18, pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.5.2, matplotlib 3.9.4, seaborn 0.13.2, JupyterLab 4.2.7, ipykernel 6.29.5, pytest 8.3.5, Plotly 5.24.1, and SciPy 1.17.1.

- [ ] **Step 2: Add CI configuration**

On pushes to `main` and pull requests, create the Conda environment, run `git diff --check`, validate notebook/report JSON, and run `pytest -q`. Do not require the ignored raw dataset.

- [ ] **Step 3: Document branch protection**

Require PRs, the `verify` check, conversation resolution, and blocked force pushes/deletions on `main`; document that the owner enables this only after the workflow succeeds on the PR.

- [ ] **Step 4: Validate configuration locally**

Parse YAML with an available parser, run every CI shell command locally, and run the full tests.

- [ ] **Step 5: Commit**

Commit message: `Add reproducible environment and CI guardrails`

### Task 4: Current documentation and notebook state

**Files:**
- Modify: `README.md`
- Modify: `notebooks/02_numerical_eda.ipynb`

**Interfaces:**
- Consumes: preflight CLI, evaluation commands, final and supplemental reports.
- Produces: one accurate entry point for setup, project status, model results, limitations, and verification.

- [ ] **Step 1: Update README**

Add preflight/setup commands, completed Submission 2 stages, final frozen metrics, order-level supplement, test/CI instructions, and warnings that the final test cannot be reused for tuning.

- [ ] **Step 2: Re-execute numerical notebook in the exact environment**

Verify source-cell content matches `origin/main`, zero execution errors, sequential execution counts for nonempty code cells, and regenerated numerical figures have unchanged analytical values.

- [ ] **Step 3: Execute categorical notebook in the exact environment**

Verify zero errors and that Plotly imports successfully; do not commit incidental binary differences unless analytical content changed.

- [ ] **Step 4: Run the entire verification matrix**

Run all 73+ tests, JSON/YAML/notebook validation, `git diff --check`, preflight canonical/CRLF diagnostics, all five original evaluation runners into temporary paths, and the supplemental runner. Compare tracked key metrics/counts with explicit tolerances.

- [ ] **Step 5: Commit**

Commit message: `Update project status and verified notebooks`

### Task 5: Whole-branch review and pull request

**Files:**
- Review all changes relative to `origin/main`.

- [ ] **Step 1: Invoke `superpowers:requesting-code-review` for the complete branch**

Review especially frozen-file immutability, final-test non-retuning, raw-data privacy, and CI behavior without local data.

- [ ] **Step 2: Address every confirmed finding test-first**

Re-run targeted tests after each fix and the full verification matrix afterward.

- [ ] **Step 3: Invoke `superpowers:verification-before-completion`**

Capture fresh test/reproduction outputs and verify a clean worktree except intended commits.

- [ ] **Step 4: Push and create the pull request**

Use a concise summary, explicit non-goals, validation evidence, and post-merge branch-protection instructions; attach the PR to this task.
