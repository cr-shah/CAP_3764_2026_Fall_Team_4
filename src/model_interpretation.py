"""Reformat tracked evidence without loading customer data or fitting models."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "reports/always_late_validation.json",
    "reports/tree_validation.json",
    "reports/automl_validation.json",
    "reports/validation_alert_policies.json",
    "reports/final_evaluation.json",
    "reports/order_level_evaluation.json",
    "docs/FINAL_MODEL_CONFIG.json",
)
METRICS = ("average_precision", "precision", "recall", "alert_rate")
COUNTS = ("true_alerts", "false_alerts", "missed_late", "correct_non_alerts", "total_alerts")
FIELDS = ("stage", "partition", "unit", "model", "variant", "observations",
          "threshold", *METRICS, *COUNTS, "source", "source_pointer",
          "operating_source", "operating_pointer")


def build_report(root: Path = ROOT) -> dict:
    """Copy full-precision metrics, retaining source paths and JSON pointers."""
    baseline, tree, automl, policies, final, orders, config = [
        json.loads((root / path).read_text(encoding="utf-8")) for path in SOURCES
    ]
    rows = []

    def add(stage, partition, unit, model, variant, ap, source, pointer):
        row = dict.fromkeys(FIELDS)
        row.update(stage=stage, partition=partition, unit=unit, model=model,
                   variant=variant, average_precision=ap, source=source,
                   source_pointer=pointer)
        rows.append(row)
        return row

    def operating(row, result, source, pointer):
        row.update(observations=result.get("items", result.get("orders")),
                   threshold=result["threshold"],
                   operating_source=source, operating_pointer=pointer,
                   missed_late=result.get("missed_late_items", result.get("missed_late_orders")))
        for field in (*METRICS[1:], "true_alerts", "false_alerts",
                      "correct_non_alerts", "total_alerts"):
            row[field] = result[field]

    for i, result in enumerate(automl["search_results"]):
        add("inner_training_cv", "train_expanding_folds", "order_item",
            result["model"], "selected_setting_mean_of_3_fold_APs",
            result["selected_cv_average_precision"], SOURCES[2], f"/search_results/{i}")

    for i, result in enumerate(automl["validation_leaderboard"]):
        row = add("validation", "validation", "order_item", result["model"],
                  result["kind"], result["validation_average_precision"],
                  SOURCES[2], f"/validation_leaderboard/{i}")
        row["observations"] = final["split"]["partitions"]["validation"]["rows"]
        if result["model"] in ("depth_3_tree", "decision_tree", "always_predict_late"):
            index = 1 if result["model"] == "always_predict_late" else 0
            operating(row, final["results"][index], SOURCES[4], f"/results/{index}")
        elif result["model"] == "logistic_regression":
            name = "logistic_regression_C_0_1"
            for j, policy in enumerate(policies["models"][name]["policies"]):
                bounds = policy["threshold_interval"]
                low, high = bounds["lower_exclusive"], bounds["upper_inclusive"]
                if (low is None or low < config["alert_threshold"]) and (
                    high is None or config["alert_threshold"] <= high
                ):
                    operating(row, {**policy, "items": row["observations"],
                                    "threshold": config["alert_threshold"]}, SOURCES[3],
                              f"/models/{name}/policies/{j}")
                    break

    for i, result in enumerate(final["results"]):
        if result["partition"] == "test":
            row = add("final_frozen_test", "test", result["unit"], result["model"],
                      "frozen_reporting", result["average_precision"], SOURCES[4], f"/results/{i}")
            operating(row, result, SOURCES[4], f"/results/{i}")
    for i, result in enumerate(orders["results"]):
        row = add("supplemental_" + ("validation" if result["partition"] == "validation"
                                    else "final_frozen_test"),
                  result["partition"], result["unit"], result["model"],
                  "post_hoc_reporting_only", result["average_precision"], SOURCES[5], f"/results/{i}")
        operating(row, result, SOURCES[5], f"/results/{i}")

    # Cross-check reused references; no fitting, prediction, or raw-data access.
    assert baseline["validation_reference"]["average_precision"] == final["results"][1]["average_precision"]
    assert tree["validation_comparison"][1]["average_precision"] == final["results"][0]["average_precision"]
    assert tree["model"]["parameters"] == config["estimator_parameters"] == final["frozen_estimator_parameters"]
    selected_tree = next(r for r in automl["search_results"] if r["model"] == "decision_tree")
    assert selected_tree["refit_estimator_parameters"] == config["estimator_parameters"]
    assert final["alert_threshold"] == orders["alert_threshold"] == config["alert_threshold"]
    profiles = []
    for profile in policies["models"]["depth_3_tree"]["profile_probabilities"]:
        profiles.append({**profile, "alerts_at_frozen_threshold":
                         profile["late_probability"] >= config["alert_threshold"]})
    return {
        "purpose": "Tracked evidence only; no training, tuning, calibration, or threshold selection",
        "source_sha256": {path: hashlib.sha256((root / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                          for path in SOURCES},
        "notes": [
            "Inner CV AP is the unweighted mean of three per-item fold APs within outer training; no single observation count applies.",
            "Compare models within the same stage and unit. Different periods and weighting are not a controlled model comparison.",
            "Null/blank values mean unreported or inapplicable, never zero. Random-forest threshold metrics were not reported.",
            "Tree reference and selected tree candidate share the frozen configuration; they are not independent models.",
            "Logistic threshold metrics map the already-fixed 0.50 threshold into its tracked validation policy interval; no policy selection occurs.",
            "Unique-order results are post-hoc supplemental reporting, not model selection evidence.",
        ],
        "rows": rows,
        "shipping_profiles": profiles,
        "profile_source": SOURCES[3] + "#/models/depth_3_tree/profile_probabilities",
        "frozen_configuration": config,
    }


def comparison_csv(report: dict) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["rows"])
    return stream.getvalue()


def interpretation_markdown(report: dict) -> str:
    def value(number, percent=False):
        if number is None:
            return "—"
        return f"{100 * number:.2f}%" if percent else f"{number:.6f}"

    lines = [
        "# Frozen model comparison and interpretation", "",
        "Generated from tracked reports by `python -m src.model_interpretation`; no customer data is required.", "",
        "## What was compared", "",
        "AP (average precision) measures ranking across thresholds. Precision is the share of alerts that are late; recall is the share of late cases alerted. AP is not precision at 0.50.", "",
        "| Stage | Unit | Model | Observations | AP | Precision at 0.50 | Recall at 0.50 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    stage_labels = {"inner_training_cv": "Inner training folds (mean)",
                    "validation": "Validation", "final_frozen_test": "Final test (frozen)",
                    "supplemental_validation": "Validation (supplemental)",
                    "supplemental_final_frozen_test": "Final test (supplemental)"}
    model_labels = {"logistic_regression": "Logistic regression (C=0.1)",
                    "decision_tree": "Selected depth-3 tree", "depth_3_tree": "Existing depth-3 tree",
                    "random_forest": "Random forest (depth 3)",
                    "frozen_depth_3_tree": "Frozen depth-3 tree", "always_predict_late": "Always late"}
    for row in report["rows"]:
        unit = "Item" if row["unit"] == "order_item" else "Order"
        count = f"{row['observations']:,}" if row["observations"] is not None else "—"
        lines.append(f"| {stage_labels[row['stage']]} | {unit} | {model_labels[row['model']]} | "
                     f"{count} | "
                     f"{value(row['average_precision'])} | {value(row['precision'], True)} | {value(row['recall'], True)} |")
    lines += ["", *["- " + note for note in report["notes"]], "",
        "The bounded historical search selected settings using only three expanding chronological folds within outer training, preserving whole orders. Validation and final test cover later, separate periods. The inner-fold mean is not a validation or final-test score. Item weighting gives multi-item orders more influence; the supplemental order unit weights each order once.", "",
        "## Why retain the tree", "",
        "The selected logistic regression, decision tree, and random forest tie exactly on validation AP. With the two approved inputs, shipping mode and scheduled duration define four observed profiles, and all selected candidates rank them First Class > Second Class > Same Day > Standard Class. Different score values therefore produce the same AP; the tie does not establish that the tree is more accurate.", "",
        "The earlier AutoML preference for logistic regression came from its predeclared family tie rule, not superior performance. The later frozen configuration records the team's explicit choice of the existing tree before final-test scoring. A depth cap of 3, minimum leaf size of 100 training items, and seed 42 give a compact, reproducible four-leaf structure. Its stable profile ranking across the compared candidates, readable decisions, and operational simplicity justify retaining it without an AP gain from the forest. This does not establish statistical stability under future distribution changes.", "",
        "Only `days_for_shipment_scheduled` and `shipping_mode` are predictors. IDs and timestamps preserve order groups and chronology. Outcome fields and identifying fields are excluded. Preprocessing and the tree were fitted on training only; validation was not added to training.", "",
        "## What the decisions mean", "",
        "The frozen rule alerts when late-class probability is **greater than or equal to 0.50**, including equality. This threshold was fixed before test scoring; it was not optimized for business costs. Earlier tree/policy documents describe an illustrative or undecided threshold because they precede the frozen configuration.", "",
        "| Observed shipping profile | Scheduled days | Training-derived tree late score | Alert at 0.50 |",
        "|---|---:|---:|---|",
    ]
    for p in report["shipping_profiles"]:
        lines.append(f"| {p['shipping_mode']} | {p['days_for_shipment_scheduled']} | "
                     f"{p['late_probability']:.6f} | {'Yes' if p['alerts_at_frozen_threshold'] else 'No'} |")
    item = next(r for r in report["rows"] if r["stage"] == "final_frozen_test" and r["model"] == "frozen_depth_3_tree")
    order = next(r for r in report["rows"] if r["stage"] == "supplemental_final_frozen_test")
    lines += ["",
        "Source: `reports/validation_alert_policies.json`, tree profile probabilities. These are training leaf proportions, not calibrated deployment risks or observed test late rates. The table describes observed profiles; it does not establish behavior for unknown or inconsistent future combinations. No causal claim about shipping choices follows from these associations.", "",
        f"On final test, the tree issued {item['total_alerts']:,} item alerts: {item['true_alerts']:,} true alerts and {item['false_alerts']:,} false alerts. It missed {item['missed_late']:,} late items and correctly left {item['correct_non_alerts']:,} non-late items unalerted. Precision was {value(item['precision'], True)} and recall {value(item['recall'], True)}, so {value(1 - item['recall'], True)} of late items were missed.", "",
        "![Frozen tree final-test confusion matrices for items and unique orders](../reports/figures/modeling/frozen_tree_alert_outcomes.png)", "",
        "**Figure:** Actual outcomes are rows; alert decisions are columns. Left: the frozen primary evaluation, one vote per item. Right: post-hoc supplemental reporting, one vote per order. Counts come directly from the respective tracked final-evaluation reports. Each panel uses its own count scale; compare rates in the table rather than cell colors across panels.", "",
        f"The supplemental order view gives {value(order['precision'], True)} precision and {value(order['recall'], True)} recall, with {order['missed_late']:,} of {order['true_alerts'] + order['missed_late']:,} late orders missed ({value(1 - order['recall'], True)}). The policy produces comparatively precise alerts while missing roughly 46% of late orders. Item counts must not be described as unique interventions.", "",
        "Always-late catches every late case by alerting on everything; the tree reduces false alerts at the cost of missed late deliveries. Lower thresholds historically added whole shipping profiles, increasing recall and alert volume together. No documented intervention budget or false-alert/miss costs establish that 0.50 is economically optimal. No new threshold, model, feature, calibration, or preprocessing choice is proposed here.", "",
        "The model cannot distinguish risk within a shipping profile. Scores are uncalibrated, earlier full-data EDA was disclosed, and these results do not demonstrate causality, statistical significance, or future deployment performance. The final test has already been examined; a future candidate requires a new protocol with an untouched test set.", "",
        "## Reproduce and hand off", "",
        "Run in the existing `cap3764-team4` environment from the repository root:", "",
        "```bash", "python -m src.model_interpretation", "pytest -q tests/test_model_interpretation.py", "pytest -q", "```", "",
        "The command writes only the new comparison JSON/CSV, this document, and the modeling figure. It reads tracked JSON; it imports no evaluation runners and performs no model fitting or prediction. JSON retains full stored precision and source pointers; CSV uses blanks for unavailable values. Tests check the displayed tables against sources and compare the generated document to this tracked copy.", "",
        "Sources (the JSON is authoritative; paired CSVs round floating values to 12 decimals):", "",
    ]
    lines += [f"- [{path}](../{path})" for path in SOURCES]
    lines += ["", "Paired source tables checked by tests: `reports/automl_leaderboard.csv`, `reports/validation_alert_policies.csv`, `reports/final_evaluation.csv`, and `reports/order_level_evaluation.csv`.", "",
        "Person 2 should reuse `reports/model_comparison.csv`, the figure above, and the following verified interpretation:", "",
        "> The depth-3 tree was retained because it gives the same validation ranking as logistic regression and random forest with a compact, interpretable decision structure. At the frozen 0.50 threshold it alerts for First Class and Second Class shipping. Its alerts are comparatively precise, but it misses roughly 46% of late orders, so it supports prioritization rather than comprehensive detection.", ""]
    return "\n".join(lines)


def write_figure(report: dict, path: Path) -> None:
    """Render tracked confusion counts with explicit units and orientation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [next(r for r in report["rows"] if r["stage"] == stage
                     and r["model"] == "frozen_depth_3_tree")
                for stage in ("final_frozen_test", "supplemental_final_frozen_test")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.suptitle("Frozen tree: precise alerts, substantial missed lateness", fontsize=17, y=0.97)
    for ax, row in zip(axes, selected):
        matrix = [[row["true_alerts"], row["missed_late"]],
                  [row["false_alerts"], row["correct_non_alerts"]]]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(map(max, matrix)))
        labels = [["True alerts (TP)", "Missed late (FN)"],
                  ["False alerts (FP)", "Correct non-alerts\n(TN)"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i][j]:,}\n{labels[i][j]}", ha="center", va="center",
                        fontsize=12, color="white" if matrix[i][j] > max(map(max, matrix)) * .55 else "#15283b")
        unit = "items (primary)" if row["unit"] == "order_item" else "unique orders (supplemental)"
        ax.set_title(f"{row['observations']:,} {unit}\nPrecision {row['precision']:.2%} | Recall {row['recall']:.2%}", fontsize=12, pad=13)
        ax.set_xticks([0, 1], ["Alert", "No alert"])
        ax.set_yticks([0, 1], ["Late", "Not late"])
        ax.set_xlabel("Frozen decision: score >= 0.50", labelpad=10)
        ax.set_ylabel("Actual outcome")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.5, .105, "Final test | Same training-only tree and fixed threshold in both panels", ha="center", fontsize=11)
    fig.text(.5, .045, "Sources: reports/final_evaluation.json and reports/order_level_evaluation.json\nEach panel has its own count scale. Unique-order results are post-hoc reporting only.", ha="center", fontsize=9, color="#45566a")
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.22, wspace=.28)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white", metadata={"Software": "model_interpretation"})
    plt.close(fig)


def main() -> None:
    report = build_report()
    (ROOT / "reports/model_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / "reports/model_comparison.csv").write_text(comparison_csv(report), encoding="utf-8")
    (ROOT / "docs/MODEL_INTERPRETATION.md").write_text(interpretation_markdown(report), encoding="utf-8")
    write_figure(report, ROOT / "reports/figures/modeling/frozen_tree_alert_outcomes.png")


if __name__ == "__main__":
    main()
