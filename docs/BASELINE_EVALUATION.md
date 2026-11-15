# Submission 2: validation reference baseline

Milestone 1 implements reproducible chronological partitions and an **always
predict late** reference. It does not train a predictive model or choose an
operating threshold. The rubric summary supplied by the modeling owner leaves
model and metric selection to the team; average precision is the proposed
primary metric. The full rubric file was not present in the checkout.

## Prediction contract and Submission 1 reuse

Predict at **order placement** whether delivery will be late:

| Role | Normalized column | Use |
|---|---|---|
| Target | `late_delivery_risk` | 1 = late, 0 = not late; never an input |
| Placement timestamp | `order_date_dateorders` | Chronological partitioning only |
| Order identifier | `order_id` | Keep order items together; never an input |
| Planned shipping duration | `days_for_shipment_scheduled` | Eligible planned-service input |
| Planned shipping service | `shipping_mode` | Eligible planned-service input |

Each row is an **order item**, not a separate order. The verified source has
180,519 rows and 65,752 orders. Scores in this milestone give **each item equal
weight**; orders with more items have more influence. These are not per-order
scores, and the alert counts below count item alerts rather than unique orders.

The run reuses `load_raw_data`, `clean_supply_chain_data`,
`validate_model_features`, and `assert_no_group_overlap` from Submission 1.
Deterministic cleaning precedes partitioning; it removes exact duplicates and
PII, normalizes names/whitespace, and parses dates. The raw source SHA-256 is
checked against `docs/DATA_PREPROCESSING.md` before running. The locally missing
raw file was recovered from the existing `origin/evaluation` Git object and
matched that documented checksum exactly. Existing notebooks are unchanged.

For future predictive models, `select_model_features` uses an explicit allowlist
containing only the two planned-service fields above. Their eligibility assumes
the recorded plan was available at placement, rather than a later revision.
Submission 1 found that shipping mode maps to scheduled duration, so these
features contain redundant information; eligibility does not imply independence.

Actual duration (`days_for_shipping_real`), shipment timestamp
(`shipping_date_dateorders`), and outcome (`delivery_status`) are forbidden.
Derived shipping delays are forbidden for the same reason. All identifiers,
the target, and removed PII are excluded. Other columns remain excluded under
the existing feature policy: for example, `order_status`, financial/profit
fields, payment type, customer/product attributes, and geographic fields need
a documented placement-time availability review before promotion. No policy
override or broadening of the shared feature policy is introduced here.
The JSON report lists every excluded cleaned column.

The constant reference uses **no inputs and no learned preprocessing**. Future
imputation, encoding, scaling, sampling, and feature selection must be fitted
only on training rows, then applied to validation and eventually final test.

## Fixed chronological partitions

Orders are sorted by placement timestamp and then order ID. The timestamp at
zero-based order position `floor(0.60 * n_orders)` starts validation; the
timestamp at `floor(0.80 * n_orders)` starts final test. Membership is determined
by timestamp comparisons, never labels. Boundary ties all move to the later
partition, so percentages may differ slightly. Fractions describe **orders**,
not rows; item counts can differ substantially.

| Partition | Orders | Items | First placement | Last placement |
|---|---:|---:|---|---|
| Training | 39,451 | 118,245 | 2015-01-01 00:00 | 2016-11-21 09:32 |
| Validation | 13,150 | 39,496 | 2016-11-21 10:14 | 2017-07-09 22:53 |
| Final test | 13,151 | 22,778 | 2017-07-09 23:14 | 2018-01-31 23:38 |

The audit verifies strict timestamp ordering and **zero overlapping orders** in
train/validation, train/test, and validation/test. It also requires complete,
parsed timestamps, complete order IDs, a single timestamp per order, nonempty
partitions, and retention of every cleaned row. Conflicting timestamps or ties
that empty a partition raise errors rather than silently changing the split.
The source timestamps have no timezone; no timezone is inferred.

The generated local `data/processed/baseline_split.csv` contains only order ID,
timestamp, and partition, one row per order. It is ignored by Git. Its canonical
UTF-8/LF SHA-256 is
`7fbeb2d5b8f8c4e439166adb9c9a6195b34b872361707cde1221667774d08aaa`.
The tracked JSON report records the hash, source checksum, split boundaries,
counts, runtime versions, and validation results. Input row order does not
change partition membership or the manifest hash. No random seed is needed.

## Metric and validation result

**Average precision (AP)** summarizes precision over recall increases as the
ranking threshold varies; higher is better. Use positive-class scores with
`sklearn.metrics.average_precision_score(y_validation, positive_scores)`, with
class 1 meaning late. AP is not trapezoidal area under a plotted PR curve.
See the [scikit-learn metric definition](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html).

For delivery alerts, **precision** is the fraction of alerts that actually
correspond to late items. Low precision means more unnecessary interventions.
**Recall** is the fraction of late items that receive an alert. Low recall means
more missed late deliveries. AP supports ranking comparison without fixing an
alert threshold; it does not itself choose an acceptable workload or operating
point. Threshold selection will use validation data and explicit alert/miss
tradeoffs in a later milestone.

The reference assigns score 1 and label 1 to every validation item. This score
is a fixed ranking reference, not a calibrated claim of certain lateness.
It has no ranking discrimination: AP equals validation positive prevalence.
An AP of 1 is therefore not implied by its perfect recall.

Validation contains 21,510 late items and 17,986 non-late items:
late prevalence **54.4612%**, non-late prevalence **45.5388%**.

| Reference | Partition | Weighting | AP (primary) | Precision | Recall | F1 | Accuracy | Alert rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Always predict late | Validation | Per item | 0.544612 | 0.544612 | 1.000000 | 0.705177 | 0.544612 | 100% |

True positives = **21,510**; false positives = **17,986**; false negatives =
**0**; true negatives = **0**. The rule catches all late validation items by
alerting on everything, with unnecessary alerts on about 45.54% of items.
There is no comparison against a fitted model in this milestone.

## Reproduce and reuse

From the repository root, place the verified raw CSV in `data/raw/`, then:

```bash
conda env create -f environment.yml
conda activate cap3764-team4
python -m src.baseline_evaluation
python -m pytest -q
```

The command rebuilds deterministic cleaning in memory; it does not require an
existing processed CSV. It writes `reports/always_late_validation.json` and the
ignored local split manifest. Re-running replaces only those generated outputs.
For a comparison run without replacing the tracked report:

```bash
python -m src.baseline_evaluation --report-path data/processed/baseline_check.json
```

The report records the actual runtime versions. `environment.yml` specifies
Python 3.11, pandas 2.2, NumPy 2.1, and scikit-learn 1.5 at minor-version
precision; it is not an exact dependency lockfile.

Verification passed in the environment created from that file: Python 3.11.18,
pandas 2.2.3, NumPy 2.1.3, and scikit-learn 1.5.2. All **34 tests passed**.
Two complete reference runs produced byte-identical JSON reports and split
manifests. Synthetic tests cover repeated orders, ties at both boundaries,
missing/conflicting timestamps, degenerate splits, row-order independence,
pairwise order isolation, feature exclusions, and validation-only scoring.

The reusable contract for the AutoML teammate is:

```python
from src.data_loader import load_raw_data
from src.preprocessing import clean_supply_chain_data
from src.baseline_evaluation import (
    TARGET_COLUMN, chronological_partitions, select_model_features,
)

cleaned, _ = clean_supply_chain_data(load_raw_data())
parts = chronological_partitions(cleaned)
X_train = select_model_features(parts["train"])
y_train = parts["train"][TARGET_COLUMN]
X_validation = select_model_features(parts["validation"])
y_validation = parts["validation"][TARGET_COLUMN]
# Fit preprocessing/models on training only; select using validation AP.
# Do not pass parts["test"] to AutoML training, tuning, or threshold selection.
```

Use the same source checksum, manifest, feature allowlist, positive class,
per-item weighting, and validation AP reference above for a fair comparison.
AutoML's internal resampling must also respect chronology and whole orders;
disable ordinary random row-level internal splits. Pass explicit validation
data if supported. Any agreed feature-policy expansion must be applied to both
the manual and AutoML comparison and documented.

## Final test and review checkpoint

This runner never predicts or scores on final test and reports no test label
counts or prevalence. It uses only test structural fields for the split audit.
Existing Submission 1 EDA covered the full dataset, and the initial repository
inspection reported aggregate labels in the proposed test window. Thus this
is a holdout from predictive evaluation, not a dataset whose distribution has
never been inspected. No test predictive scores have been produced.

Fix model, features, preprocessing, hyperparameters, and alert threshold using
training/validation before the agreed final evaluation. The same final test
must be used by manual models and AutoML, with no test-driven retuning.

Milestone 1 stops at PR review. Base: `main`; feature branch:
`param/submission-2-baseline`, created from fetched `origin/main`. No shared
notebook was edited. After the owner confirms this PR is merged, fetch again
and create the next feature branch from updated `origin/main` before adding
the tree-based baseline. Commits, pushes, PR creation, and merges belong to
the owner.
