"""Fit one shallow tree on training data and compare validation AP with PR #8.

Run ``python -m src.tree_evaluation``. Final test is audited structurally only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from .baseline_evaluation import (
    DEFAULT_REPORT_PATH as REFERENCE_REPORT_PATH,
    EXPECTED_RAW_SHA256,
    MODEL_FEATURES,
    TARGET_COLUMN,
    audit_partitions,
    chronological_partitions,
    evaluate_always_late,
    select_model_features,
    split_manifest,
)
from .data_loader import REPOSITORY_ROOT, load_raw_data, raw_file_metadata
from .data_validation import validate_target
from .preprocessing import clean_supply_chain_data


DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "reports" / "tree_validation.json"
TREE_PARAMETERS = {"max_depth": 3, "min_samples_leaf": 100, "random_state": 42}
ALERT_THRESHOLD = 0.5


def build_tree_pipeline() -> Pipeline:
    """Return an unfitted pipeline; all learned transformations fit with the tree."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("scheduled_days", numeric, ["days_for_shipment_scheduled"]),
        ("shipping_service", categorical, ["shipping_mode"]),
    ], remainder="drop")
    return Pipeline([
        ("preprocessing", preprocessing),
        ("tree", DecisionTreeClassifier(**TREE_PARAMETERS)),
    ])


def fit_tree(training: pd.DataFrame) -> Pipeline:
    """Fit exactly once, using only approved training features and labels."""
    validate_target(training)
    if set(training[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Training must contain both classes.")
    features = select_model_features(training)
    # Deterministic missing-value representation, not a learned transformation.
    features = features.replace({pd.NA: np.nan})
    return build_tree_pipeline().fit(features, training[TARGET_COLUMN])


def compare_on_validation(
    training: pd.DataFrame, validation: pd.DataFrame
) -> tuple[Pipeline, list[dict[str, Any]]]:
    """Fit on training only, and compute both references on identical validation."""
    reference = evaluate_always_late(validation[TARGET_COLUMN])
    model = fit_tree(training)
    features = select_model_features(validation).replace({pd.NA: np.nan})
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    scores = model.predict_proba(features)[:, positive_index]
    predictions = (scores >= ALERT_THRESHOLD).astype(int)
    target = validation[TARGET_COLUMN]
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    tree_result = {
        "model": "shallow_decision_tree",
        "partition": "validation",
        "unit": "order_item",
        "primary_metric": "average_precision",
        "average_precision": float(average_precision_score(target, scores)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
        "alert_rate": float(predictions.mean()),
        "confusion_counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "alert_threshold": ALERT_THRESHOLD,
        "threshold_selected_on_validation": False,
        "decision_rule": "positive-class probability >= 0.5; illustrative fixed operating point",
    }
    return model, [reference, tree_result]


def verify_reference_contract(
    partitions: dict[str, pd.DataFrame], source_sha256: str, reference: dict[str, Any]
) -> dict[str, Any]:
    """Fail before fitting if source, features, or split differ from merged PR #8."""
    if source_sha256 != reference["source"]["sha256"]:
        raise ValueError("Source checksum differs from the reference report.")
    if list(MODEL_FEATURES) != reference["prospective_model_features"]:
        raise ValueError("Feature allowlist differs from the reference report.")
    manifest_text = split_manifest(partitions).to_csv(
        index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n"
    )
    digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if digest != reference["split"]["manifest_sha256"]:
        raise ValueError("Split manifest differs from the reference report.")
    audit = audit_partitions(partitions)
    for field, value in audit.items():
        if value != reference["split"][field]:
            raise ValueError(f"Split {field} differs from the reference report.")
    return {"manifest_sha256": digest, **audit}


def run_tree_evaluation(
    raw_path: str | Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    reference_report_path: Path = REFERENCE_REPORT_PATH,
) -> dict[str, Any]:
    """Reuse the merged split contract; never fit, transform, or score final test."""
    source = raw_file_metadata(raw_path)
    if source["sha256"] != EXPECTED_RAW_SHA256:
        raise ValueError("Raw source checksum differs from the agreed DataCo source.")
    reference = json.loads(Path(reference_report_path).read_text(encoding="utf-8"))
    data, _ = clean_supply_chain_data(load_raw_data(raw_path))
    partitions = chronological_partitions(data)
    split = verify_reference_contract(partitions, str(source["sha256"]), reference)
    # Deliberately remove final-test data before calling any fitting/scoring code.
    training = partitions.pop("train")
    validation = partitions.pop("validation")
    del partitions, data
    model, results = compare_on_validation(training, validation)
    original = reference["validation_reference"]
    reproduced = results[0]
    if (
        reproduced["class_counts"] != original["class_counts"]
        or not np.isclose(
            reproduced["average_precision"], original["average_precision"], rtol=0, atol=1e-12
        )
    ):
        raise ValueError("Validation reference result differs from merged PR #8.")
    preprocessing = model.named_steps["preprocessing"]
    tree = model.named_steps["tree"]
    report = {
        "source": source,
        "target": TARGET_COLUMN,
        "positive_class": 1,
        "prediction_time": "order_placement",
        "model_features": list(MODEL_FEATURES),
        "feature_policy": "same explicit allowlist and exclusions as PR #8",
        "split": split,
        "reference_contract_verified": True,
        "model": {
            "estimator": "DecisionTreeClassifier",
            "parameters": tree.get_params(),
            "fit_partition": "train",
            "hyperparameter_search": "none; one fixed baseline configuration",
            "achieved_depth": int(tree.get_depth()),
            "leaves": int(tree.get_n_leaves()),
            "rules": export_text(
                tree, feature_names=list(preprocessing.get_feature_names_out()),
                decimals=4, show_weights=True,
            ),
        },
        "preprocessing": {
            "fit_partition": "train",
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot; ignore unknown categories",
            "scaling": "none",
        },
        "validation_comparison": results,
        "validation_average_precision_gain": results[1]["average_precision"] - results[0]["average_precision"],
        "final_test_evaluated": False,
        "final_threshold_fixed": False,
        "runtime": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--reference-report-path", type=Path, default=REFERENCE_REPORT_PATH)
    args = parser.parse_args()
    report = run_tree_evaluation(args.raw_path, args.report_path, args.reference_report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
