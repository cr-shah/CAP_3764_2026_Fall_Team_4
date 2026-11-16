"""Enumerate distinct validation alert policies for the recorded tree/logistic fits.

Run ``python -m src.alert_policy_evaluation``. No operating point is selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .automl_evaluation import DEFAULT_REPORT_PATH as AUTOML_REPORT_PATH, late_probabilities
from .baseline_evaluation import (
    DEFAULT_REPORT_PATH as BASELINE_REPORT_PATH,
    EXPECTED_RAW_SHA256, MODEL_FEATURES, ORDER_ID, ORDER_TIMESTAMP, TARGET_COLUMN,
    chronological_partitions, evaluate_always_late, select_model_features,
)
from .data_loader import REPOSITORY_ROOT, load_raw_data, raw_file_metadata
from .data_validation import validate_target
from .preprocessing import clean_supply_chain_data
from .tree_evaluation import (
    DEFAULT_REPORT_PATH as TREE_REPORT_PATH, build_tree_pipeline, verify_reference_contract,
)


DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "reports" / "validation_alert_policies.json"
DEFAULT_TABLE_PATH = REPOSITORY_ROOT / "reports" / "validation_alert_policies.csv"


def alert_policy_masks(scores: np.ndarray) -> list[np.ndarray]:
    """All distinct policies under score >= threshold, with ties kept together."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite one-dimensional array.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Late probabilities must lie in [0, 1].")
    levels = np.unique(scores)[::-1]
    return [np.zeros(scores.size, dtype=bool)] + [scores >= level for level in levels]


def enumerate_alert_policies(validation: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    """Use validation labels, profiles and probabilities only; no fitting here."""
    validate_target(validation)
    if set(validation[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Validation must contain both classes.")
    scores = np.asarray(scores, dtype=float)
    masks = alert_policy_masks(scores)
    if len(scores) != len(validation):
        raise ValueError("Score count does not match validation items.")
    profiles = select_model_features(validation).reset_index(drop=True)
    profiles["late_probability"] = scores
    if profiles.groupby(list(MODEL_FEATURES), dropna=False).late_probability.nunique().gt(1).any():
        raise ValueError("One shipping profile received inconsistent scores.")
    profiles = profiles.drop_duplicates(list(MODEL_FEATURES)).sort_values(
        ["late_probability", *MODEL_FEATURES], ascending=[False, True, True], ignore_index=True
    )
    target = validation[TARGET_COLUMN].to_numpy()
    positives, negatives = int((target == 1).sum()), int((target == 0).sum())
    levels = np.unique(scores)[::-1]
    rows = []
    for index, mask in enumerate(masks):
        total = int(mask.sum())
        true_alerts = int((mask & (target == 1)).sum())
        false_alerts = int((mask & (target == 0)).sum())
        if index == 0:
            threshold = float(np.nextafter(levels[0], np.inf))
            lower, upper = float(levels[0]), None
        else:
            threshold = float(levels[index - 1])
            lower = float(levels[index]) if index < len(levels) else None
            upper = threshold
        selected_profiles = profiles.loc[profiles.late_probability >= threshold, list(MODEL_FEATURES)]
        rows.append({
            "policy_id": f"P{index}", "partition": "validation", "unit": "order_item",
            "alerted_shipping_profiles": selected_profiles.to_dict(orient="records"),
            "threshold_interval": {"lower_exclusive": lower, "upper_inclusive": upper},
            "representative_threshold": threshold,
            "precision": true_alerts / total if total else None,
            "recall": true_alerts / positives,
            "false_alerts": false_alerts, "missed_late_items": positives - true_alerts,
            "total_alerts": total, "true_alerts": true_alerts,
            "correct_non_alerts": negatives - false_alerts, "alert_rate": total / len(target),
        })
    return {
        "average_precision": float(average_precision_score(target, scores)),
        "validation_items": len(target), "late_items": positives, "not_late_items": negatives,
        "profile_probabilities": profiles.to_dict(orient="records"), "policies": rows,
    }


def fit_recorded_configurations(
    training: pd.DataFrame, tree_parameters: dict[str, Any], logistic_parameters: dict[str, Any]
) -> dict[str, Pipeline]:
    """Recreate recorded fits on training only; no search or validation fitting."""
    validate_target(training)
    if set(training[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Training must contain both classes.")
    if tree_parameters["max_depth"] != 3 or logistic_parameters["C"] != 0.1:
        raise ValueError("Expected the depth-3 tree and selected logistic C=0.1 settings.")
    tree = build_tree_pipeline().set_params(tree=DecisionTreeClassifier(**tree_parameters))
    logistic = Pipeline([
        ("preprocessing", clone(tree.named_steps["preprocessing"])),
        ("model", LogisticRegression(**logistic_parameters)),
    ])
    sorted_training = training.sort_values([ORDER_TIMESTAMP, ORDER_ID], kind="stable").reset_index(drop=True)
    with threadpool_limits(limits=1):
        tree.fit(select_model_features(training).replace({pd.NA: np.nan}), training[TARGET_COLUMN])
        logistic.fit(select_model_features(sorted_training).replace({pd.NA: np.nan}), sorted_training[TARGET_COLUMN])
    return {"depth_3_tree": tree, "logistic_regression_C_0_1": logistic}


def evaluate_validation_policies(models: dict[str, Pipeline], validation: pd.DataFrame) -> dict[str, Any]:
    features = select_model_features(validation).replace({pd.NA: np.nan})
    results, model_masks = {}, []
    for name, model in models.items():
        scores = late_probabilities(model, features)
        results[name] = enumerate_alert_policies(validation, scores)
        model_masks.append(alert_policy_masks(scores))
    first = model_masks[0]
    equivalent = all(
        len(other) == len(first) and all(np.array_equal(a, b) for a, b in zip(first, other))
        for other in model_masks[1:]
    )
    return {"models": results, "same_alert_policies_on_every_validation_item": equivalent}


def run_alert_policy_evaluation(
    raw_path: str | Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    table_path: Path = DEFAULT_TABLE_PATH,
    baseline_report_path: Path = BASELINE_REPORT_PATH,
    tree_report_path: Path = TREE_REPORT_PATH,
    automl_report_path: Path = AUTOML_REPORT_PATH,
) -> dict[str, Any]:
    source = raw_file_metadata(raw_path)
    if source["sha256"] != EXPECTED_RAW_SHA256:
        raise ValueError("Raw source checksum differs from the agreed DataCo source.")
    baseline, tree_reference, automl_reference = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in [baseline_report_path, tree_report_path, automl_report_path]
    ]
    data, _ = clean_supply_chain_data(load_raw_data(raw_path))
    partitions = chronological_partitions(data)
    split = verify_reference_contract(partitions, str(source["sha256"]), baseline)
    for saved in [tree_reference, automl_reference]:
        if (saved["source"]["sha256"] != source["sha256"]
            or saved["model_features"] != list(MODEL_FEATURES) or saved["split"] != split):
            raise ValueError("Recorded model source, feature restrictions, or split differs.")
    training, validation = partitions.pop("train"), partitions.pop("validation")
    del partitions, data  # Final test cannot reach model fitting or policy analysis.
    selected_logistic = next(row for row in automl_reference["search_results"]
                             if row["model"] == "logistic_regression")
    if selected_logistic["selected_settings"] != {"model__C": 0.1}:
        raise ValueError("Recorded logistic selection is no longer C=0.1.")
    models = fit_recorded_configurations(
        training, tree_reference["model"]["parameters"], selected_logistic["refit_estimator_parameters"]
    )
    evaluated = evaluate_validation_policies(models, validation)
    for name, recorded_name in [("depth_3_tree", "depth_3_tree"), ("logistic_regression_C_0_1", "logistic_regression")]:
        expected = next(row["validation_average_precision"] for row in automl_reference["validation_leaderboard"]
                        if row["model"] == recorded_name)
        if abs(evaluated["models"][name]["average_precision"] - expected) > 1e-12:
            raise ValueError("Recorded validation AP was not reproduced.")
    reference_result = evaluate_always_late(validation[TARGET_COLUMN])
    if reference_result["class_counts"] != baseline["validation_reference"]["class_counts"]:
        raise ValueError("Validation class counts differ from the baseline.")
    report = {
        "source": source, "split": split, "model_features": list(MODEL_FEATURES),
        "target": TARGET_COLUMN, "positive_class": 1, "reference_contracts_verified": True,
        "fit_scope": "training only; exact recorded settings; no hyperparameter search",
        "recorded_configurations": {"depth_3_tree": tree_reference["model"]["parameters"],
                                    "logistic_regression_C_0_1": selected_logistic["refit_estimator_parameters"]},
        "decision_rule": "alert if late-class probability >= threshold; tied scores enter together",
        "interval_convention": "lower_exclusive < threshold <= upper_inclusive; null means unbounded",
        "no_alert_precision": "undefined; stored as null",
        **evaluated,
        "operating_point": None,
        "decision_basis": "No documented alert budget or false-alert/missed-late costs found in project materials; team choice required.",
        "final_test_evaluated": False,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    report_path, table_path = Path(report_path), Path(table_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    rows = []
    for name, result in report["models"].items():
        for policy in result["policies"]:
            row = {"model": name, **policy}
            for field in ["alerted_shipping_profiles", "threshold_interval"]:
                row[field] = json.dumps(row[field], sort_keys=True)
            rows.append(row)
    pd.DataFrame(rows).to_csv(table_path, index=False, lineterminator="\n", float_format="%.12f")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    args = parser.parse_args()
    report = run_alert_policy_evaluation(args.raw_path, args.report_path, args.table_path)
    print(json.dumps({"same_alert_policies": report["same_alert_policies_on_every_validation_item"],
                      "models": report["models"], "operating_point": None,
                      "final_test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
