# Executive Summary Source

Source text and number provenance for `reports/executive_summary_report.docx`, a
filled copy of the instructor-provided `executive_summary_report_template.docx`.
The original template is kept unchanged outside the repository.

**Word count:** 585 words of report text, excluding table cells and table/figure
caption labels (33 caption words and 78 table words; 696 in total).

Figures are generated from tracked artifacts only, without customer data or model fitting:

```bash
python -m src.final_visuals
```

| Figure | File | Tracked sources |
|---|---|---|
| 1. Model performance comparison | `reports/figures/final/fig1_model_comparison.png` | `reports/model_comparison.csv` |
| 2. Alert-policy trade-off | `reports/figures/final/fig2_alert_policy_tradeoff.png` | `reports/validation_alert_policies.csv`, `reports/model_comparison.csv`, `reports/order_level_evaluation.csv` |
| 3. Profile risk and error burden | `reports/figures/final/fig3_profile_risk_and_errors.png` | `reports/calibration_analysis.csv` (Person 1) |

## Header

- **Team Number:** 4
- **Date:** November 15, 2026
- **Subject:** Late-Delivery Risk Prediction for Order-Placement Alerts

## Project Goal

Predict, at the moment an order is placed, whether it is at risk of late delivery so operations can prioritize intervention. The objective is useful early alerts with few false alarms, while stating clearly how many late orders the alerts miss.

## Major Findings

We analyzed the DataCo supply-chain dataset: 180,519 order-item records from 65,752 orders placed between January 2015 and January 2018; 54.8% of items were delivered late. Logistic regression, a decision tree, and a random forest were compared against an always-late reference.

**Table 1. Model Performance Comparison (average precision, order items)**

| Model | Validation AP | Final-test AP | Selected |
|---|---|---|---|
| Always late (reference) | 0.545 | 0.548 | Reference |
| Logistic regression (C = 0.1) | 0.742 | Not scored | No |
| Random forest (depth 3) | 0.742 | Not scored | No |
| Depth-3 decision tree | 0.742 | 0.750 | Yes (frozen) |

As shown in Table 1, the three tuned models tied exactly on validation average precision (AP). With only two approved inputs, shipping mode and scheduled days, every model ranks the same four shipping profiles identically, so none is more accurate. We retained the depth-3 tree because its four decisions are easy to read and explain.

On the final test period (13,151 orders, scored once), the frozen tree reached AP 0.750 versus 0.548 for alerting on everything (Figure 1). At the fixed 0.50 rule, 84.8% of alerted orders were late, but only 53.7% of late orders were alerted: 3,345 late orders were missed and 698 alerts were false. Item-level results were nearly identical (84.65% precision, 54.02% recall).

**Figure 1. Model Performance Comparison:** `reports/figures/final/fig1_model_comparison.png`

The frozen rule alerts only First Class and Second Class shipments. Lowering the threshold adds a whole shipping profile at a time, so recall and alert volume rise together (Table 2, Figure 2).

**Table 2. Validation Alert Policies (order items)**

| Profiles alerted | Items alerted | Precision / Recall |
|---|---|---|
| First Class only | 15.4% | 94.7% / 26.9% |
| First + Second Class (frozen 0.50 rule) | 34.4% | 84.2% / 53.1% |
| + Same Day | 39.5% | 79.4% / 57.6% |
| + Standard Class (= always late) | 100% | 54.5% / 100% |

No documented intervention budget or cost of misses shows that 0.50 is economically optimal.

**Figure 2. Alert-Policy Trade-off:** `reports/figures/final/fig2_alert_policy_tradeoff.png`

Validation probabilities track observed late rates within 3.1 percentage points for every profile (Figure 3; Brier score 0.198 versus 0.248 for a prevalence-only forecast). Errors are concentrated: Standard Class holds 90.5% of missed late items and Second Class 85.0% of false alerts.

**Figure 3. Profile Risk and Error Burden (validation):** `reports/figures/final/fig3_profile_risk_and_errors.png`

## Recommendations

1. Use alerts to prioritize First and Second Class orders for proactive handling, treating them as a high-precision watch list (84.8%), not complete coverage.
2. Do not rely on these alerts for Standard Class orders, which hold most missed late deliveries; collect approved order-time information that separates risk within a shipping profile.
3. Set any future threshold from an agreed intervention budget and the cost of misses versus false alerts, then confirm it on new, untouched data.
4. Monitor late rates by shipping profile for drift, and pilot interventions with a control group before claiming impact.

## Analytical Overview

Raw data were checksum-verified, cleaned deterministically, and stripped of direct personal information. A written feature policy excluded identifiers and outcome-derived fields such as delivery status and actual shipping days. Orders were split chronologically into training, validation, and test periods with no shared orders. Preprocessing and models were fit on training data only (118,245 items); settings were searched with three expanding time folds inside training. The model and 0.50 threshold were frozen and hashed before the test set was scored once; it cannot be reused for tuning. The pipeline runs in a pinned Conda environment with automated tests in GitHub Actions.

## Risks and Limitations

Two inputs yield only four risk scores, so the model cannot separate orders within a profile and misses about 46% of late orders. The 0.50 threshold was fixed, not chosen from business costs. No calibration model was fitted, and scores may drift as the shipping mix changes. Results are associations, not causes, and do not demonstrate deployment readiness. Earlier exploratory analysis used the full dataset, which is disclosed. A new model requires a fresh protocol with an untouched test set.

## Number provenance

Every number above comes from a tracked artifact; none was recalculated from the final test.

| Number in report | Source |
|---|---|
| 180,519 rows, 65,752 orders, 54.8% late | `README.md`, `docs/DATA_PREPROCESSING.md`, `notebooks/02_numerical_eda.ipynb` |
| January 2015 to January 2018 order dates; 118,245 training items | `docs/FINAL_MODEL_CONFIG.json` `/split/partitions` |
| Validation AP 0.742 (three-way tie); always-late 0.545 | `reports/model_comparison.csv` validation rows |
| Final-test AP 0.750 (0.749683); always-late 0.548 | `reports/final_evaluation.csv` test rows |
| 13,151 test orders; 84.8% precision; 53.7% recall; 3,345 missed; 698 false alerts | `reports/order_level_evaluation.csv` test row |
| 84.65% item precision; 54.02% item recall | `reports/final_evaluation.csv` test tree row |
| Table 2 alert rates, precision, recall | `reports/validation_alert_policies.csv`, `depth_3_tree` policies P1 to P4 |
| Largest profile gap 3.1 pp; Brier 0.198 vs 0.248 | `reports/calibration_analysis.json` `/calibration_metrics`; `docs/CALIBRATION_ERROR_ANALYSIS.md` |
| Standard Class 90.5% of missed late items; Second Class 85.0% of false alerts | `docs/CALIBRATION_ERROR_ANALYSIS.md` |
| About 46% of late orders missed | `docs/ORDER_LEVEL_EVALUATION.md`, `docs/MODEL_INTERPRETATION.md` |
| Three expanding time folds inside training | `docs/MODEL_INTERPRETATION.md` |

## Integrated teammate findings

- Person 1 (`docs/CALIBRATION_ERROR_ANALYSIS.md`): close profile-level probability
  alignment with no calibrator fitted; errors concentrated in Standard Class (misses)
  and Second Class (false alerts); monitor profile late rates and seek approved
  pre-order features.
- Person 3 (`docs/MODEL_INTERPRETATION.md`): the three tuned models tie on validation
  AP because they rank the four profiles identically; the tree was retained for
  interpretability, not superior AP; the policy supports prioritization rather than
  comprehensive detection.

No model, feature, preprocessing, calibration, or threshold decision was changed for
this report, and the final test was not used for tuning.
