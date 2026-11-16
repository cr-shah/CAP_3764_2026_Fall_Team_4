"""Evaluate the pre-recorded final tree and always-late reference, without tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from .baseline_evaluation import (
    EXPECTED_RAW_SHA256, MODEL_FEATURES, TARGET_COLUMN, chronological_partitions, select_model_features,
)
from .data_loader import REPOSITORY_ROOT, load_raw_data, raw_file_metadata
from .data_validation import validate_target
from .preprocessing import clean_supply_chain_data
from .tree_evaluation import build_tree_pipeline, fit_tree, verify_reference_contract


DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "docs" / "FINAL_MODEL_CONFIG.json"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "reports" / "final_evaluation.json"
DEFAULT_TABLE_PATH = REPOSITORY_ROOT / "reports" / "final_evaluation.csv"


def text_sha256(path: str | Path) -> str:
    """Ignore platform newline conversion when hashing tracked text artifacts."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def load_frozen_configuration(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config["features"] != list(MODEL_FEATURES) or config["alert_threshold"] != 0.5
        or config["fit_partition"] != "train" or config["refit_with_validation"] is not False
        or config["target"] != TARGET_COLUMN or config["positive_class"] != 1
        or config["estimator"] != "DecisionTreeClassifier" or not config["frozen_before_test_scoring"]):
        raise ValueError("Configuration differs from the authorized final protocol.")
    if config["estimator_parameters"] != build_tree_pipeline().named_steps["tree"].get_params():
        raise ValueError("Frozen tree parameters differ from the existing depth-3 tree.")
    for group in ["implementation_sha256", "reference_report_sha256"]:
        for name, expected in config[group].items():
            if text_sha256(REPOSITORY_ROOT / name) != expected:
                raise ValueError(f"Frozen {group} mismatch: {name}")
    return config, text_sha256(path)


def score_predictions(
    target: pd.Series, scores: np.ndarray, model_name: str, partition: str, threshold: float
) -> dict[str, Any]:
    validate_target(pd.DataFrame({TARGET_COLUMN: target}))
    values = np.asarray(target)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("Each evaluated partition must contain both classes.")
    if (scores.shape != values.shape or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Expected one finite late-class probability in [0, 1] per item.")
    alerts = scores >= threshold
    tp, fp = int((alerts & (values == 1)).sum()), int((alerts & (values == 0)).sum())
    fn, tn = int((~alerts & (values == 1)).sum()), int((~alerts & (values == 0)).sum())
    return {
        "model": model_name, "partition": partition, "unit": "order_item",
        "items": len(values), "late_items": int((values == 1).sum()),
        "late_prevalence": float((values == 1).mean()), "threshold": threshold,
        "average_precision": float(average_precision_score(values, scores)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn), "false_alerts": fp, "missed_late_items": fn,
        "total_alerts": tp + fp, "true_alerts": tp, "correct_non_alerts": tn,
        "alert_rate": float(alerts.mean()),
    }


def score_partition(model: Any, data: pd.DataFrame, name: str, threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = select_model_features(data).replace({pd.NA: np.nan})
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    scores = model.predict_proba(features)[:, positive_index]
    rows = [
        score_predictions(data[TARGET_COLUMN], scores, "frozen_depth_3_tree", name, threshold),
        score_predictions(data[TARGET_COLUMN], np.ones(len(data)), "always_predict_late", name, threshold),
    ]
    diagnostics = select_model_features(data).reset_index(drop=True)
    diagnostics["target"] = data[TARGET_COLUMN].to_numpy()
    diagnostics["score"] = scores
    grouped = diagnostics.groupby(list(MODEL_FEATURES), dropna=False, as_index=False).agg(
        items=("target", "size"), late_items=("target", "sum"), late_probability=("score", "first")
    )
    grouped["late_rate"] = grouped.late_items / grouped["items"]
    grouped["item_share"] = grouped["items"] / len(data)
    grouped["alerts"] = grouped.late_probability >= threshold
    grouped["partition"] = name
    return rows, grouped.to_dict(orient="records")


def verify_validation_results(rows: list[dict[str, Any]], reference: dict[str, Any]) -> None:
    for row, old_name in zip(rows, ["shallow_decision_tree", "always_predict_late"]):
        previous = next(value for value in reference["validation_comparison"] if value["model"] == old_name)
        for metric in ["average_precision", "precision", "recall"]:
            if not np.isclose(row[metric], previous[metric], rtol=0, atol=1e-12):
                raise ValueError(f"Validation {metric} does not reproduce the recorded result.")
        for field, confusion_key in [("false_alerts", "fp"), ("missed_late_items", "fn"),
                                     ("true_alerts", "tp"), ("correct_non_alerts", "tn")]:
            if row[field] != previous["confusion_counts"][confusion_key]:
                raise ValueError(f"Validation {field} differs from the recorded result.")


def run_final_evaluation(
    raw_path: str | Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    table_path: Path = DEFAULT_TABLE_PATH,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    # Verify the already-recorded decision before loading or scoring test data.
    config, config_hash = load_frozen_configuration(configuration_path)
    baseline = json.loads((REPOSITORY_ROOT / config["baseline_report_path"]).read_text(encoding="utf-8"))
    tree_reference = json.loads((REPOSITORY_ROOT / config["tree_report_path"]).read_text(encoding="utf-8"))
    source = raw_file_metadata(raw_path)
    if source["sha256"] != EXPECTED_RAW_SHA256 or source["sha256"] != config["source_sha256"]:
        raise ValueError("Source checksum differs from the frozen data source.")
    data, _ = clean_supply_chain_data(load_raw_data(raw_path))
    partitions = chronological_partitions(data)
    split = verify_reference_contract(partitions, str(source["sha256"]), baseline)
    if split != config["split"] or split != tree_reference["split"]:
        raise ValueError("Split differs from the frozen manifest/audit.")
    if tree_reference["model"]["parameters"] != config["estimator_parameters"]:
        raise ValueError("Reference estimator parameters differ from the frozen decision.")
    model = fit_tree(partitions["train"])
    if model.named_steps["tree"].get_params() != config["estimator_parameters"]:
        raise ValueError("Fitted tree parameters differ from the frozen decision.")
    if list(model.named_steps["preprocessing"].feature_names_in_) != config["features"]:
        raise ValueError("Fitted preprocessing inputs differ from the frozen feature list.")
    model_fingerprint = joblib.hash(model)
    threshold = config["alert_threshold"]
    validation_rows, validation_profiles = score_partition(model, partitions["validation"], "validation", threshold)
    verify_validation_results(validation_rows, tree_reference)
    if text_sha256(configuration_path) != config_hash:
        raise ValueError("Configuration changed before final-test scoring.")
    # First test prediction occurs only after freezing, fitting, and validation checks.
    test_rows, test_profiles = score_partition(model, partitions["test"], "test", threshold)
    if joblib.hash(model) != model_fingerprint or text_sha256(configuration_path) != config_hash:
        raise ValueError("Model or configuration changed during final evaluation.")
    deltas = {metric: test_rows[0][metric] - validation_rows[0][metric]
              for metric in ["average_precision", "precision", "recall", "late_prevalence", "alert_rate"]}
    report = {
        "configuration_id": config["configuration_id"], "configuration_sha256": config_hash,
        "configuration_recorded_before_test_scoring": True,
        "source": source, "split": split, "target": TARGET_COLUMN, "positive_class": 1,
        "model_features": config["features"], "frozen_estimator_parameters": config["estimator_parameters"],
        "alert_threshold": threshold, "fit_partition": "train", "validation_added_to_training": False,
        "training_fitted_model_fingerprint": model_fingerprint,
        "model_and_configuration_unchanged_after_scoring": True,
        "validation_reference_reproduced_before_test": True,
        "results": validation_rows + test_rows,
        "tree_test_minus_validation": deltas,
        "profile_diagnostics": validation_profiles + test_profiles,
        "final_test_evaluated": True,
        "post_test_model_feature_threshold_changes": "none; no tuning, selection, calibration, or refitting after results",
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    report_path, table_path = Path(report_path), Path(table_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    pd.DataFrame(report["results"]).to_csv(table_path, index=False, lineterminator="\n", float_format="%.12f")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    args = parser.parse_args()
    report = run_final_evaluation(args.raw_path, args.report_path, args.table_path)
    print(json.dumps({"configuration_sha256": report["configuration_sha256"],
                      "results": report["results"], "tree_test_minus_validation": report["tree_test_minus_validation"],
                      "final_test_evaluated": True}, indent=2))


if __name__ == "__main__":
    main()
