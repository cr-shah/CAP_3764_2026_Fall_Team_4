# Submission 2: validation alert policies

This milestone enumerates the policies achievable with a single global rule:
**alert when late-class probability >= threshold**. Only the existing validation
predictions and labels determine policy counts. No operating point is chosen.

## Reused configurations and evaluation data

The source checksum, two-feature allowlist, outer chronological partitions,
and zero order overlap are verified against the existing baseline, tree, and
AutoML reports. The manifest hash remains
`7fbeb2d5b8f8c4e439166adb9c9a6195b34b872361707cde1221667774d08aaa`.
Final test is removed before fitting or prediction and remains unscored.

There are no saved fitted-model binaries. The runner recreates the recorded
depth-3 tree and selected logistic regression (`C=0.1`, L2, `lbfgs`) on training
only, using their full stored estimator parameters and shared preprocessing.
The logistic training-row ordering and single-thread fit match the AutoML
runner. No search, calibration, validation fitting, or model selection is repeated.
Both reproduce their recorded validation AP of **0.7420000295494212**.

The only model inputs remain `shipping_mode` and `days_for_shipment_scheduled`.
IDs and placement timestamps determine partition structure only. All existing
post-order, identifier, target, PII, and unreviewed-feature exclusions remain.
Shared notebooks and previous implementations/reports are unchanged.

## All distinct policies: counts are per item

Validation has **39,496 items**, including **21,510 late** items. Each item
contributes equally; these are not counts of unique orders or interventions.
Orders with several items contribute several observations.

| Policy | Shipping profiles alerted | Precision | Recall | False alerts (items) | Missed late items | Total alerts (items) |
|---|---|---:|---:|---:|---:|---:|
| P0 | None | Undefined | 0.00% | 0 | 21,510 | 0 |
| P1 | First Class (1 scheduled day) | 94.72% | 26.87% | 322 | 15,730 | 6,102 |
| P2 | First Class (1 day), Second Class (2 days) | 84.19% | 53.14% | 2,146 | 10,079 | 13,577 |
| P3 | First Class, Second Class, Same Day (0 days) | 79.36% | 57.60% | 3,223 | 9,121 | 15,612 |
| P4 | All four, including Standard Class (4 days) | 54.46% | 100.00% | 17,986 | 0 | 39,496 |

This table applies to **both models**, verified by comparing every item's
alert/no-alert decision for all policies, not merely comparing aggregate counts.
For P0 precision is undefined because there are no alerts; JSON stores `null`
and CSV leaves the precision cell blank. No-alert and alert-all endpoints are
included. Scores are compared at full precision, and tied profiles enter together.

## Why different scores give the same policies

| Validation shipping profile | Tree late score | Logistic late score |
|---|---:|---:|
| First Class / 1 day | 0.956305 | 0.955626 |
| Second Class / 2 days | 0.769035 | 0.768714 |
| Same Day / 0 days | 0.439531 | 0.442680 |
| Standard Class / 4 days | 0.381484 | 0.381599 |

Both models assign one score to each of the same four profiles and rank them
in the same order. A global threshold can only add the next complete score
group, yielding four cumulative alert sets plus no alerts. It cannot select
arbitrary profile subsets or a fraction of a tied profile without a separate
tie-breaking policy. The equivalence is established for observed validation
items, not every possible future input or unknown category.

Let `s_first`, `s_second`, `s_same`, and `s_standard` denote a model's **unrounded**
scores above. Each model's complete threshold intervals are:

| Policy | Threshold interval for that model |
|---|---|
| P0 | `t > s_first` |
| P1 | `s_second < t <= s_first` |
| P2 | `s_same < t <= s_second` |
| P3 | `s_standard < t <= s_same` |
| P4 | `t <= s_standard` |

The JSON and CSV contain full-precision bounds and a representative threshold
for each model/policy. These examples are not selected operating thresholds.
Unbounded endpoints are stored as `null` in JSON. If a model assigns score 1,
the no-alert example can use a threshold above 1, equivalent to disabling alerts.
The probabilities in this document are rounded for display; use report bounds
to reproduce exact boundary behavior. Equality triggers an alert, consistent
with scikit-learn's [precision/recall threshold convention](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.precision_recall_curve.html).

The **same numeric threshold need not yield the same policy**: at `t=0.44`,
the tree yields P2 while logistic regression yields P3. At `t=0.5`, both yield
P2, which was the earlier illustrative point, not an approved business choice.

## Decision left to the team

Inspection of the README, all existing project documentation, and source cells
of the three notebooks found **no documented alert budget, false-alert cost,
or missed-late cost**. Their absence is not an assumption that the costs are zero.
The runner therefore leaves `operating_point` unset and makes no recommendation.

Higher recall requires more alerts and more false alerts. For example, moving
from P2 to P3 adds **2,035 item alerts**, catches **958 additional late items**,
and adds **1,077 false alerts**. The team must decide whether that tradeoff fits
its capacity and priorities. Counts describe the full validation period, not a
daily workload budget. Scores have not been calibrated as deployment risks.

## Reproduce and verify

From the repository root, with the existing environment and verified raw CSV:

```bash
conda run -n cap3764-team4 python -m src.alert_policy_evaluation
conda run -n cap3764-team4 python -m pytest -q
```

Outputs are `reports/validation_alert_policies.json` and
`reports/validation_alert_policies.csv` (five rows per model). To reproduce
without replacing them, add `--report-path data/processed/alert_check.json`
and `--table-path data/processed/alert_check.csv` to the runner command.

All **59 tests passed** in the existing Python 3.11.18 / scikit-learn 1.5.2
environment. Tests cover all endpoints, threshold equality, tied profiles,
unrounded nearby scores, undefined precision, invalid predictions, exact
per-item policy equivalence, fixed training-only configurations, and poisoned
final-test isolation. Existing split, preprocessing, and tuning tests also pass.
Only existing SciPy L-BFGS option deprecation warnings were reported.
Repeated runs produced byte-identical JSON and CSV artifacts. Content hashes
confirmed all 50 pre-existing inspected files, including shared notebooks,
earlier reports, and the local split manifest, remained unchanged.

This milestone stops for review on `param/submission-2-alert-threshold`.
Git commands and PR actions remain with the owner.
