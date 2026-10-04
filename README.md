# Predicting Late Delivery Risk in Supply Chain Operations

## Overview

This repository contains a CAP 3764 Fall 2026 team project examining
late-delivery risk in supply-chain operations. The current reproducible work
covers data loading, validation, deterministic cleaning, privacy, future
leakage protection, and Phase 1 numerical and categorical exploratory analysis.
Modeling and evaluation remain separate later workstreams.

## Problem Statement

The eventual goal is to estimate whether an order is at risk of late delivery
using information available at or near order/fulfillment decision time. The
dataset contains order-item rows, so future evaluation must keep all items from
the same order in the same data partition.

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

Run the automated checks with:

```bash
pytest -q
```

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

## Repository Structure

```text
CAP_3764_2026_Fall_Team_4/
├── data/
│   ├── raw/          # Local immutable source data and tracked data dictionary
│   └── processed/    # Locally generated, ignored cleaned data
├── docs/             # Preprocessing and feature-policy documentation
├── notebooks/        # Preprocessing and teammate analysis notebooks
├── reports/figures/  # Generated analysis/modeling figures
├── src/              # Reusable loading, validation, cleaning, and guardrails
├── tests/            # Synthetic-data unit tests
├── environment.yml
├── LICENSE
└── README.md
```

## Project Status

The deterministic preprocessing foundation and Phase 1 numerical and
categorical EDA are implemented. Modeling and evaluation remain future work.

## Team

- Chaitanya Raj Shah
- Naw Lin Lun Nway
- Param Patel
- Abhiram

## Course

CAP 3764 — Fall 2026

## License

Original project code and materials created by the team are licensed under the MIT License. The DataCo dataset is not covered by this license and remains subject to the terms of its original source.
