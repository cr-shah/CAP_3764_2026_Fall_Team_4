# Submission 2: focused automated model comparison

This bounded scikit-learn search uses the existing `cap3764-team4` environment;
the supplied assignment summary does not require PyCaret. It reuses the
[baseline contract](BASELINE_EVALUATION.md) and
[tree-model contract](TREE_MODEL_EVALUATION.md), without editing their code,
reports, feature policy, or shared notebooks.

## Data and tuning protocol

Target: `late_delivery_risk`, **late = 1**, predicted at order placement. The
only inputs are `shipping_mode` and `days_for_shipment_scheduled`.
`select_model_features` enforces the existing exclusions. Order ID and placement
time are used only to preserve order groups and chronology, never as inputs.

The original 60/20/20 outer split is unchanged. Source checksum, feature lists,
partition counts/timestamps, and zero pairwise order overlap are checked against
the existing reports before fitting. The manifest hash remains
`7fbeb2d5b8f8c4e439166adb9c9a6195b34b872361707cde1221667774d08aaa`.

Only outer training enters tuning. Three expanding folds are formed over
sorted **unique placement timestamps**, then mapped to row positions. All
items from an order and all orders tied at a timestamp remain together.
Every fit window ends strictly before its score window begins. Folds use equal
timestamp-cohort counts, not guaranteed equal calendar durations.

| Fold | Fit orders / items | Score orders / items | Order overlap |
|---|---:|---:|---:|
| 1 | 9,865 / 29,527 | 9,862 / 29,621 | 0 |
| 2 | 19,727 / 59,148 | 9,862 / 29,405 | 0 |
| 3 | 29,589 / 88,553 | 9,862 / 29,692 | 0 |

Each configuration gets a fresh pipeline in each fold: training-fold median
imputation for scheduled days, most-frequent imputation and one-hot encoding
for mode, with unknown categories ignored. This clones the shared tree
preprocessing; no scaling or global learned preprocessing is applied. The
numeric input spans only 0–4 days. Both classes must be present in each fold.
References: scikit-learn's [time-series splitting](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
and [grid search](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html).

## Search and chosen settings

Six configurations across three families give **18 fold fits**, followed by
one outer-training refit per family. The fixed tree is also refitted to verify
its reference result. Search jobs and fitting thread pools are serial;
randomized estimators use seed 42. Search errors raise.

| Family | Small search | Chosen setting | Mean inner-fold AP |
|---|---|---|---:|
| Logistic regression | `C` = 0.1, 1.0 | `C=0.1`, L2, `lbfgs`, `max_iter=1000` | 0.754742 |
| Decision tree | depth = 1, 3 | depth 3, minimum leaf 100 items | 0.754742 |
| Random forest | depth = 2, 3 | depth 3, 100 trees, minimum leaf 100, `max_features=None`, bootstrap enabled | 0.754742 |

Selection uses the unweighted mean of three **per-item AP** fold scores from
late-class probabilities. Settings tied within `1e-12` use their declared grid
order. Family ties prefer logistic regression, then tree, then forest. This
predeclared rule provisionally prefers **logistic regression, C=0.1**, without
claiming an empirical win. Outer validation is not used for tuning or this
preference. The JSON report records all settings and fold scores.

## Original-validation leaderboard

All rows use the same 39,496 validation items, equally weighted per item;
multi-item orders have more influence. AP uses class-1 probabilities, never
hard labels. No operating threshold is selected.

| Validation rank | Model | Validation AP | Gain over always-late | Gain over fixed tree |
|---|---|---:|---:|---:|
| 1 (tie) | Existing depth-3 tree | 0.742000 | 0.197388 | 0.000000 |
| 1 (tie) | Tuned logistic regression | 0.742000 | 0.197388 | 0.000000 |
| 1 (tie) | Tuned decision tree | 0.742000 | 0.197388 | 0.000000 |
| 1 (tie) | Tuned random forest | 0.742000 | 0.197388 | 0.000000 |
| 5 | Always predict late | 0.544612 | 0.000000 | -0.197388 |

Both references were reproduced. Training and validation each contain only
four profiles: shipping mode maps one-to-one to scheduled days. All
selected models rank First Class > Second Class > Same Day > Standard Class.
Their probability values differ slightly, but AP depends on ranking and ties,
so the validation scores are **exactly equal at full stored precision**.
The ensemble adds no validation AP improvement with these inputs.

No model can distinguish items within the same profile. This comparison does
not demonstrate probability calibration, superiority beyond this small search,
or statistical significance. Fold-score standard deviations are descriptive,
not confidence intervals; training histories overlap. Inner CV and outer
validation cover different periods, so their AP levels need not match.

Precision determines how many delivery alerts are correct; recall determines
how many late deliveries receive an alert. The operating threshold remains
**undecided** pending an agreed missed-delivery versus false-alert tradeoff.
Final test is removed before search/refitting/scoring and remains unscored.
Prior aggregate inspection is disclosed in the baseline write-up.

## Reproduce and verify

From the repository root, with the verified raw CSV in its existing location:

```bash
conda run -n cap3764-team4 python -m src.automl_evaluation
conda run -n cap3764-team4 python -m pytest -q
```

Outputs: `reports/automl_validation.json` and `reports/automl_leaderboard.csv`.
Earlier reports and the split manifest are unchanged. For ignored local outputs:

```bash
conda run -n cap3764-team4 python -m src.automl_evaluation --report-path data/processed/automl_validation_check.json --leaderboard-path data/processed/automl_leaderboard_check.csv
```

Use `conda run` or an activated environment on Windows so numerical-library
DLL paths are set correctly. Verification used Python 3.11.18, pandas 2.2.3, NumPy 2.1.3, and
scikit-learn 1.5.2. All **50 tests passed**, including whole-order folds,
timestamp ties, fold-local encoder fitting, invalid folds, late-class scoring,
deterministic ties, validation exclusion from tuning, and poisoned final-test
isolation. Only SciPy L-BFGS option deprecation warnings were reported.
Two complete runs produced byte-identical JSON and CSV outputs. Content hashes
confirmed all 41 pre-existing inspected files, including earlier reports and
the split manifest, remained unchanged.

This milestone stops for owner review on `param/submission-2-automl`.
No Git commands or PR actions are performed by the runner or assistant.
