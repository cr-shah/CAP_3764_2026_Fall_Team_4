# Predicting Late Delivery Risk in Supply Chain Operations

## Overview

This repository contains a CAP 3764 Fall 2026 team project examining
late-delivery risk in supply-chain operations. The current reproducible work
covers data loading, validation, deterministic cleaning, privacy and leakage
protection, numerical and categorical exploratory analysis, chronological
model validation, and a frozen final evaluation.

## Problem Statement

The goal is to estimate whether an order is at risk of late delivery
using information available at or near order/fulfillment decision time. The
dataset contains order-item rows, so every chronological partition keeps all
items from the same order together. The approved model uses only scheduled
shipping days and shipping mode; outcome-derived and post-order fields are
excluded.

## Dataset

Download the
[DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
dataset. Place these files in `data/raw/`:

- `DataCoSupplyChainDataset.csv` — required raw data, kept local and ignored by Git
- `DescriptionDataCoSupplyChain.csv` — small tracked data dictionary

The expected raw file contains 180,519 order-item observations and 65,752
unique orders. See [Data Preprocessing](docs/DATA_PREPROCESSING.md) for the
verified checksum, encoding, cleaning decisions, and reproducibility details.

## Environment

Create the shared Python 3.11 Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate cap3764-team4
```

The environment file pins the exact versions used for repository verification.
Check the local runtime and source-data integrity before running the pipeline:

```bash
python -m src.project_preflight
```

This diagnostic is read-only. It reports a missing or wrong raw file separately
from the common case where the correct CSV was converted from LF to CRLF line
endings.

## Preprocessing

Generate the local cleaned dataset with:

```bash
python -m src.preprocessing
```

This reads the immutable raw file using Latin-1 encoding and writes
`data/processed/dataco_clean.csv`. Generated processed files are ignored by Git.
The workflow performs no EDA, learned imputation, encoding, scaling, data
splitting, or modeling.

The companion notebook, `notebooks/01_data_preprocessing.ipynb`, demonstrates
the same preprocessing decisions without charts or predictive analysis.

Feature eligibility and leakage rules are documented in
[Feature Policy](docs/FEATURE_POLICY.md).

## Exploratory Analysis

The Phase 1 exploratory analysis uses the cleaned order-item dataset and is
split across two notebooks:

- `notebooks/02_numerical_eda.ipynb` examines numerical distributions,
  shipping-duration patterns, outcome comparisons, correlations, and outliers.
- `notebooks/03_categorical_eda.ipynb` examines categorical volumes and
  late-delivery rates, including geographic, product, shipping-mode, and
  leakage-focused views.

Generated charts are stored under `reports/figures/numerical/` and
`reports/figures/categorical/`. The findings are descriptive associations, not
causal claims or evidence that a field is eligible for predictive modeling.

## Modeling and Evaluation

Submission 2 uses chronological, order-grouped train, validation, and final-test
partitions. Validation work compared the always-late reference, a fixed shallow
decision tree, alert policies, and a small training-only AutoML-style search.
The recorded final choice is the depth-3 decision tree at a fixed 0.50 alert
threshold, fitted on training data only.

The frozen final-test item-level result is:

| Unit | Rows/orders | Average precision | Precision | Recall |
|---|---:|---:|---:|---:|
| Order item | 22,778 | 0.749683 | 84.65% | 54.02% |
| Unique order (supplemental) | 13,151 | 0.749332 | 84.76% | 53.72% |

The unique-order row is a post-hoc reporting supplement that weights every
`order_id` once. It reproduces the frozen item-level contract before aggregating
predictions. See [Final Evaluation](docs/FINAL_EVALUATION.md) and
[Supplemental Unique-Order Evaluation](docs/ORDER_LEVEL_EVALUATION.md).

The final test has already been examined. Its results must not be used to tune
the model, features, preprocessing, or threshold. A future candidate requires a
new protocol with an untouched test set.

To reproduce the tracked evaluation artifacts after preflight succeeds:

```bash
python -m src.baseline_evaluation
python -m src.tree_evaluation
python -m src.alert_policy_evaluation
python -m src.automl_evaluation
python -m src.final_evaluation
python -m src.order_level_evaluation
```

## Final Report

The executive summary is
[`reports/executive_summary_report.docx`](reports/executive_summary_report.docx),
a filled copy of the instructor template (585 words of report text, excluding
table cells and captions). Its source text, number provenance, and integrated
teammate findings are in [Executive Summary Source](docs/EXECUTIVE_SUMMARY_SOURCE.md).

The three report figures are drawn only from tracked JSON/CSV reports; no
customer data is read and no model is fitted:

```bash
python -m src.final_visuals
```

| Figure | Shows |
|---|---|
| `reports/figures/final/fig1_model_comparison.png` | Validation AP tie among tuned models and the frozen final-test result |
| `reports/figures/final/fig2_alert_policy_tradeoff.png` | Validation precision/recall for each alert policy and the frozen 0.50 rule |
| `reports/figures/final/fig3_profile_risk_and_errors.png` | Validation profile calibration and where false alerts and misses fall |

## Verification and Contributions

Run the automated tests with:

```bash
pytest -q
```

GitHub Actions runs the same suite and validates tracked notebook/report JSON on
every pull request and push to `main`. Repository administrators should enable
the settings in [Repository Guardrails](docs/REPOSITORY_GUARDRAILS.md) after the
workflow succeeds so changes to `main` require a reviewed, green pull request.

## Repository Structure

```text
CAP_3764_2026_Fall_Team_4/
├── data/
│   ├── raw/          # Local immutable source data and tracked data dictionary
│   └── processed/    # Locally generated, ignored cleaned data
├── docs/             # Workflow, policy, and evaluation documentation
├── notebooks/        # Preprocessing and teammate analysis notebooks
├── reports/          # Tracked evaluation artifacts and generated figures
├── src/              # Loading, validation, cleaning, modeling, and checks
├── tests/            # Synthetic-data unit tests
├── environment.yml
├── LICENSE
└── README.md
```

## Project Status

Preprocessing, Phase 1 EDA, Submission 2 modeling, the frozen final
evaluation, calibration and error analysis, model interpretation, and the final
executive summary with its figure package are complete. The final test has been
examined and must not be used for further tuning.

## Team

- Chaitanya Raj Shah
- Naw Lin Lun Nway
- Param Patel
- Abhiram

## Course

CAP 3764 — Fall 2026

## License

Original project code and materials created by the team are licensed under the MIT License. The DataCo dataset is not covered by this license and remains subject to the terms of its original source.
