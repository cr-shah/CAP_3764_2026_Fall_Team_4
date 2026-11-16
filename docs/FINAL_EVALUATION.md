# Submission 2: frozen final evaluation

The team explicitly selected the existing **depth-3 decision tree** with a
fixed **0.50 alert threshold** before final-test scoring. The record is
`docs/FINAL_MODEL_CONFIG.json`. This supersedes the earlier milestones' reserved
test status: final test has now been evaluated, without subsequent tuning.

## Configuration recorded before scoring

- Inputs: `days_for_shipment_scheduled` and `shipping_mode` only.
- Target: `late_delivery_risk`, with 1 meaning late; prediction at order placement.
- Tree: depth 3, minimum 100 training items per leaf, seed 42; all other saved
  parameters are unchanged from the original tree.
- Training: **118,245 items / 39,451 orders**, using the original training partition.
  Validation was not added to training.
- Preprocessing: existing training-fitted median imputation for scheduled days,
  most-frequent imputation and one-hot encoding for shipping mode, ignoring
  unknown categories; no scaling.
- Alert rule: late-class probability **>= 0.50**. No threshold search or calibration.

The final configuration was written before any held-out test predictions.
It records the data/split checksums, full tree parameters, and hashes of shared
implementation and prior reports. The evaluator's source was also hashed before
the live test run. Text hashes normalize line endings to LF for portability.
Frozen configuration SHA-256:
`a7110749d19fcbb41d06ceaa97e065a6c6e6d0d564b2da1a5d4faf65219a3288`.

The runner verifies that record, recreates the tree using `fit_tree` on training
only, and reproduces the historical validation metrics before test prediction.
The fitted model fingerprint and configuration hash are checked before/after
evaluation. Both remain unchanged. No model, feature, preprocessing, or threshold
choice was revised after observing test results. Earlier full-data EDA and
aggregate inspection remain disclosed in the baseline write-up.

## Split verification

The runner reuses `chronological_partitions`, `select_model_features`, and
`verify_reference_contract`. The same source CSV checksum is verified:
`994b3c8d24049cc46bf20e8161fedece9a9b95376cc0dc929921d76cf6b9bc0d`.

Split manifest SHA-256:
`7fbeb2d5b8f8c4e439166adb9c9a6195b34b872361707cde1221667774d08aaa`.

All train/validation, train/test, and validation/test order intersections are
**zero**; date ranges are strictly chronological. Test covers placements from
**2017-07-09 23:14** through **2018-01-31 23:38**, with **13,151 orders / 22,778
items**. Timestamps retain their source interpretation without an inferred
timezone. IDs and timestamps remain structural fields, never predictors.

## Final results: all metrics and counts are per item

Each row receives equal weight; an order with multiple items contributes
multiple observations. Counts are item alerts and item errors, not unique-order
alerts or interventions. Both models are evaluated on exactly the same items
within each partition.

| Partition | Model | Items | Average precision | Precision | Recall | False alerts (items) | Missed late items | Total alerts (items) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | Frozen tree, threshold 0.50 | 39,496 | 0.742000 | 84.19% | 53.14% | 2,146 | 10,079 | 13,577 |
| Validation | Always late | 39,496 | 0.544612 | 54.46% | 100.00% | 17,986 | 0 | 39,496 |
| **Final test** | **Frozen tree, threshold 0.50** | **22,778** | **0.749683** | **84.65%** | **54.02%** | **1,223** | **5,743** | **7,970** |
| Final test | Always late | 22,778 | 0.548336 | 54.83% | 100.00% | 10,288 | 0 | 22,778 |

AP uses late-class probabilities and is independent of the operating threshold.
Precision, recall, and alert/error counts use the frozen 0.50 rule. Always-late
assigns score 1 and alerts on everything, so its AP and precision equal the
partition's late prevalence. Test contains **12,490 late** and **10,288 non-late**
items. Tree true positives = **6,747**, true negatives = **9,065**.

At the selected operating point, the tree avoids **9,065 false alerts** compared
with always-late while missing **5,743 late items**. Its test AP improvement over
always-late is **0.201347**. This is the final descriptive comparison, not a
trigger for changing the selected operating point.

## Validation versus test at the same threshold

Tree AP increased by **0.007683**. Precision increased by **0.46 percentage
points**, recall by **0.88 points**, and the alerted fraction by **0.61 points**
(34.38% to 34.99%). Raw count differences also reflect the smaller test set;
they cannot be interpreted as rate changes on their own.

The trained profile scores and alert decisions are identical across partitions:
First Class and Second Class alert; Same Day and Standard Class do not. The
observed profile mix and within-profile late rates differ:

| Shipping profile | Validation item share | Test item share | Validation late rate | Test late rate |
|---|---:|---:|---:|---:|
| First Class / 1 scheduled day | 15.45% | 15.14% | 94.72% | 94.75% |
| Second Class / 2 days | 18.93% | 19.85% | 75.60% | 76.95% |
| Same Day / 0 days | 5.15% | 5.72% | 47.08% | 52.46% |
| Standard Class / 4 days | 60.47% | 59.29% | 38.19% | 37.46% |

For example, the alerted Second Class profile is more common in test and has
a higher observed late rate. These composition/outcome differences account for
the metric changes with fixed predictions per profile; no new model learning
occurred. They do not establish causation or statistical significance. The
model still distinguishes only four profiles, cannot distinguish risk within
a profile, and misses about 46% of late test items at the chosen threshold.
No test-based model, calibration, feature, or threshold search was performed.

## Reproduction and verification

From the repository root, using the existing verified raw CSV and environment:

```bash
conda run -n cap3764-team4 python -m src.final_evaluation
conda run -n cap3764-team4 python -m pytest -q
```

The first command repeats the **same frozen protocol**, writing
`reports/final_evaluation.json` and `reports/final_evaluation.csv`. There are no
CLI options to choose another model, feature set, or threshold. To reproduce
into ignored local paths without replacing the final reports:

```bash
conda run -n cap3764-team4 python -m src.final_evaluation --report-path data/processed/final_check.json --table-path data/processed/final_check.csv
```

Changes to recorded implementation/reference files or frozen settings cause
the evaluator to fail before test scoring. Do not regenerate the configuration
to accommodate a model change after observing this test set.

All **68 tests passed before the live held-out run**, including synthetic tests
for exact threshold equality, per-item confusion counts, rejection of changed
features/settings/reference hashes, training-only fitting, validation checks
before test access, and unchanged model/configuration fingerprints. A synthetic
test reverses held-out labels and verifies that the original training-fitted
model performs poorly, rather than adapting to them. The actual test partition
was scored once in this milestone. Only existing SciPy option deprecation
warnings occurred in tests. Runtime: Python 3.11.18, pandas 2.2.3, NumPy 2.1.3,
scikit-learn 1.5.2.

This milestone stops for owner review on `param/submission-2-final-evaluation`.
No Git commands, commits, pushes, merges, or PR actions were performed.
