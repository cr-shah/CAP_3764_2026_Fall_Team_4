# Submission 2: shallow decision-tree comparison

This milestone adds one tree-based model and compares it with the always-late
reference from merged PR #8 on the **same validation items**. Final test remains
unscored. The shared notebooks, split implementation, feature policy, and
reference report are unchanged.

## Model choice and fitting

The estimator is a `DecisionTreeClassifier` with `max_depth=3`,
`min_samples_leaf=100`, and `random_state=42`; other settings use scikit-learn
1.5.2 defaults. The depth cap and minimum leaf size constrain complexity.
A shallow tree provides readable rules and can represent the different risk
levels among the four shipping-service profiles identified in Submission 1.
Both allowed features are retained for the agreed comparison, although their
one-to-one relationship in this source makes them redundant.

These settings were fixed **before viewing this model's validation score**.
One configuration is fitted on training items only, with no hyperparameter
search, resampling, class reweighting, or refitting on validation. The fitted
tree has depth 3 and four leaves. Its full rules and training class counts per
leaf are included in `reports/tree_validation.json`. At the illustrative 0.5
threshold, the tree alerts for First Class and Second Class shipping.

The pipeline fits the following transformations on training items only:

- Median imputation for scheduled shipping days.
- Most-frequent imputation and one-hot encoding for shipping mode.
- Unknown shipping categories are encoded as all zeros at prediction time.

No scaling is used. Validation passes through the already-fitted pipeline.
The implementation uses positive-class probabilities for AP, not thresholded
labels. Tree probabilities are training leaf proportions; they have not been
calibrated or established as reliable deployment probabilities.
See scikit-learn's documentation for
[decision trees](https://scikit-learn.org/1.5/modules/tree.html) and
[unknown-category encoding](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

## Reused evaluation contract

The prediction is made at **order placement**. Target: `late_delivery_risk`,
with **1 = late**. Scores and alert counts are **per order item**, giving each
row equal weight. Orders containing more items consequently have more influence;
these are not equally weighted per-order scores or unique-order alert counts.

`src/baseline_evaluation.py` supplies the unchanged functions
`chronological_partitions`, `audit_partitions`, `split_manifest`,
`select_model_features`, and `evaluate_always_late`.

| Partition | Unique orders | Order items | Purpose |
|---|---:|---:|---|
| Training | 39,451 | 118,245 | Fit preprocessing and tree |
| Validation | 13,150 | 39,496 | Compare AP and describe alert tradeoffs |
| Final test | 13,151 | 22,778 | Structural audit only; no predictions or scores |

The chronological boundaries remain validation starting **2016-11-21 10:14**
and final test starting **2017-07-09 23:14**, using source timestamps without
an inferred timezone. Orders tied at a boundary timestamp remain together on
the later side. Pairwise order overlaps are zero, and every earlier partition
ends strictly before the next starts.

Before fitting, the runner checks the raw SHA-256 against the agreed source
and compares the feature allowlist, manifest hash, partition counts, date
boundaries, and overlap audit against the PR #8 report. Manifest SHA-256:
`7fbeb2d5b8f8c4e439166adb9c9a6195b34b872361707cde1221667774d08aaa`.
The always-late reference is recomputed on validation and must reproduce its
original class counts and AP. The runner does not rewrite that report or the
local split manifest.

The **only model inputs** are `days_for_shipment_scheduled` and `shipping_mode`,
under the existing assumption that the recorded shipping plan was available at
placement. `order_id` is used only to preserve whole orders; the placement
timestamp is used only for partitioning. All identifiers and the target are
excluded from inputs. Actual shipping duration, shipment date, delivery status,
and any derived shipping delay remain forbidden. PII is removed by shared
cleaning; every review-required or unclassified field remains excluded. No
feature-policy override or expansion is introduced.

## Validation results and delivery-alert tradeoff

Primary metric: **average precision (AP)**, higher is better, calculated with
`average_precision_score(y_validation, positive_class_probability)`. AP uses
the ranking across thresholds; it is not the precision at threshold 0.5 and
is not trapezoidal PR area.

Validation contains **21,510 late** and **17,986 non-late** items, for late
prevalence **54.4612%**. Both rows below use all 39,496 validation items.

| Model | Partition / weighting | AP (primary) | Precision | Recall | F1 | Accuracy | Alert rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Always late | Validation / per item | 0.544612 | 0.544612 | 1.000000 | 0.705177 | 0.544612 | 100.00% |
| Shallow decision tree, threshold 0.5 | Validation / per item | **0.742000** | 0.841939 | 0.531427 | 0.651580 | 0.690475 | 34.38% |

Absolute validation AP gain: **0.197388**. This supports improved prioritization
within this validation period; it is not a final-test result or a statistical
significance claim. With only four service profiles, many items receive tied
scores and the model cannot distinguish risk within a service profile.

| Model / operating point | True alerts (TP) | False alerts (FP) | Missed late items (FN) | Correct non-alerts (TN) | Total alerts |
|---|---:|---:|---:|---:|---:|
| Always late | 21,510 | 17,986 | 0 | 0 | 39,496 |
| Tree at 0.5 | 11,431 | 2,146 | 10,079 | 15,840 | 13,577 |

Precision measures the share of alerts that correspond to late items. Recall
measures the share of late items receiving alerts. At 0.5, the tree avoids
15,840 false alerts but misses 10,079 late items that always-late would flag.
It improves precision while reducing recall; its F1 is also lower than the
always-late rule. AP improvement does not mean every operating metric improves.

The **0.5 threshold is a fixed illustrative operating point**, not an optimized
or approved business threshold. Lower thresholds generally increase alert
volume and recall; a final choice needs an agreed tolerance for missed late
deliveries versus unnecessary interventions. Final model and threshold choices
remain open for training/validation work before any final-test evaluation.

## Reproduction and AutoML handoff

Use the environment in `environment.yml` and the same verified local raw CSV
required by PR #8. From the repository root:

```bash
conda run -n cap3764-team4 python -m src.tree_evaluation
conda run -n cap3764-team4 python -m pytest -q
```

The first command fits the model and writes only
`reports/tree_validation.json`. To reproduce without replacing that report:

```bash
conda run -n cap3764-team4 python -m src.tree_evaluation --report-path data/processed/tree_validation_check.json
```

The JSON records actual runtime versions, estimator parameters, training rules,
source and split hashes, all validation scores, and the final-test reservation.
It does not save test predictions or an estimated test score. The fitted model
is recreated by the code; no model pickle or customer records are committed.

For AutoML, use the loading and split example in
[the PR #8 baseline contract](BASELINE_EVALUATION.md#reproduce-and-reuse).
That code supplies `X_train`, `y_train`, `X_validation`, and `y_validation` with
the same allowlist. For the manual model, the equivalent entry point is:

```python
from src.tree_evaluation import compare_on_validation

model, validation_results = compare_on_validation(parts["train"], parts["validation"])
```

Use validation AP with positive class 1 and per-item weighting for comparison.
AutoML must receive the explicit validation partition; its internal fitting,
resampling, and preprocessing must preserve chronology and whole orders.
Neither manual modeling nor AutoML may use final test for fitting, feature
selection, hyperparameters, early stopping, calibration, or threshold choice.
Evaluate the agreed final models on the same reserved test once choices are
fixed. Earlier full-data EDA and aggregate inspection are disclosed in the
baseline contract; this remains a holdout from predictive evaluation.

## Verification and review checkpoint

All **41 tests passed** with Python 3.11.18, pandas 2.2.3, NumPy 2.1.3, and
scikit-learn 1.5.2. New tests cover training-only imputation/category learning,
unseen validation categories, AP calculated from probabilities rather than
hard labels, deterministic repeated fitting, source/feature/split drift,
single-class training rejection, and final-test isolation. The final-test
isolation test poisons test features and removes test labels after shared
cleaning; fitting and validation scoring still succeed. Existing split,
preprocessing, and leakage tests also pass.

Two complete runs produced byte-identical tree reports. Content hashes confirm
all 36 pre-existing files under `src/`, `tests/`, `docs/`, and `reports/` were
unchanged; the existing local split manifest still matches the PR #8 hash.

This milestone stops for owner review on `param/submission-2-tree-model`.
Git commands and PR creation remain with the owner. No shared notebook was
edited and no final-test predictive evaluation was performed.
