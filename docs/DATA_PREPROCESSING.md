# Data Preprocessing

This document describes deterministic preprocessing for the DataCo Smart Supply
Chain for Big Data Analysis dataset. It deliberately excludes exploratory data
analysis, learned transformations, feature selection, and modeling.

## Source integrity

- Source: [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- Raw filename: `DataCoSupplyChainDataset.csv`
- Data dictionary: `DescriptionDataCoSupplyChain.csv`
- Required source encoding: `latin-1`
- Raw file size: `95,729,629` bytes
- SHA-256: `994b3c8d24049cc46bf20e8161fedece9a9b95376cc0dc929921d76cf6b9bc0d`
- Raw dimensions: 180,519 rows by 53 columns

The raw CSV is immutable input. The pipeline reads it but never overwrites it.
The large source file is intentionally ignored by Git; each teammate should
verify the file size and checksum before preprocessing.

## Unit of analysis

**Observation:** The 180,519 rows represent order items. There are 65,752
unique `Order Id` values, and 114,767 rows repeat an already-seen order ID.

**Decision:** Preserve repeated `order_id` values.

**Reason:** An order may legitimately contain multiple products or line items.
Repeated order IDs are not exact duplicate observations.

**Impact:** Future data splitting must group on `order_id`; no order may appear
in both training and test data.

## Column names

**Observation:** Source names contain spaces, capitalization, and parentheses.

**Decision:** Normalize names deterministically to lowercase snake_case.
Normalization collisions raise an error.

**Reason:** Stable names make reusable code portable and prevent silent column
replacement.

**Impact:** Examples include `Late_delivery_risk` becoming
`late_delivery_risk` and `order date (DateOrders)` becoming
`order_date_dateorders`.

## Dates

**Observation:** Both source dates are text fields.

**Decision:** Convert `order_date_dateorders` and
`shipping_date_dateorders` with `pandas.to_datetime(errors="coerce")` and record
successful, failed, and originally missing values.

**Reason:** Datetime values are required for reliable ordering and later time
features. Failed values must remain visible as `NaT` rather than causing row
deletion.

**Impact:** In the verified source, all 180,519 values in each date column
convert successfully and no rows are removed.

## Exact duplicate rows

**Observation:** The verified raw file contains zero exact duplicate rows.

**Decision:** Count and remove only exact full-row duplicates if they occur.

**Reason:** Exact duplicates can double-count an observation, while repeated
`order_id` values reflect valid order-item structure.

**Impact:** Zero rows are removed from the verified source; final row count
remains 180,519.

## Product Description

**Observation:** `product_description` is missing for all 180,519 rows.

**Decision:** Remove it from the cleaned analytical dataset only when it is
confirmed to be 100% missing.

**Reason:** The field contains no information in this source file.

**Impact:** One column is removed and no rows are removed.

## Missing values

**Observation:** Source missingness is concentrated in `product_description`
(180,519), `order_zipcode` (155,679), `customer_lname` (8), and
`customer_zipcode` (3).

**Decision:** Do not call global `dropna` and do not statistically impute any
field. Remove only the confirmed all-missing product description and separately
remove direct PII such as customer surname. Preserve other missing values.

**Reason:** Statistical imputation must be learned from training data only.
Large missingness can also be meaningful and must not be erased without a
research justification.

**Impact:** `order_zipcode` and `customer_zipcode` missing values remain in the
cleaned output for later, split-aware treatment.

## Direct PII

The pipeline never displays PII values. It removes these fields:

| Field | Reason |
|---|---|
| `customer_email` | Direct contact information is unnecessary for the research question. |
| `customer_password` | Credential-like data is unnecessary and must not be modeled. |
| `customer_fname` | A given name is direct PII with no justified analytical role. |
| `customer_lname` | A surname is direct PII with no justified analytical role. |
| `customer_street` | A street address is unnecessarily granular direct location PII. |

Useful geographic categories such as market, region, state, and country are not
automatically removed. Their modeling eligibility still requires prediction-time
review under `docs/FEATURE_POLICY.md`.

## Identifiers

Identifiers remain available for integrity checks and grouping but are marked
`RETAINED_FOR_STRUCTURE_NOT_MODELING`. In particular, `order_id` is essential
for preventing cross-split order overlap. Other ID columns are listed in the
feature policy.

## String whitespace

**Observation:** External CSV fields may contain leading or trailing whitespace.

**Decision:** Strip surrounding whitespace from string values and represent
empty strings as missing values.

**Reason:** This is deterministic cleanup and does not learn information from
the dataset.

**Impact:** Category spelling is made more consistent without statistical
imputation.

## Outliers

Outliers are retained during deterministic preprocessing unless they are proven
to be invalid data. Statistical outlier treatment, if needed, must be justified
later. The pipeline records numeric minima and maxima but does not delete unusual
sales, profit, price, quantity, or geographic observations.

## Operations deliberately deferred

The full dataset is **not** used to fit scalers, imputers, encoders, target
encoders, feature selectors, PCA, SMOTE, or sampling procedures. Any learned
preprocessing must be fitted using training data only after a leakage-safe split.

## Output

The deterministic output is written locally to
`data/processed/dataco_clean.csv`. It contains 180,519 rows and 47 columns for
the verified source. The generated file is ignored by Git; source code and
documentation are the reproducible artifacts.

## Reproduce the output

From the repository root:

```bash
conda env create -f environment.yml
conda activate cap3764-team4
python -m src.preprocessing
pytest -q
```

Before and after preprocessing, verify the raw file checksum if source integrity
is in question. The preprocessing command prints only structural counts and the
output location; it does not print customer records.
