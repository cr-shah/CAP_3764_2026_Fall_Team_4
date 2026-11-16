"""Bounded scikit-learn comparison with grouped chronological training folds.

Run ``python -m src.automl_evaluation``. Final test is never scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .baseline_evaluation import (
    DEFAULT_REPORT_PATH as BASELINE_REPORT_PATH,
    EXPECTED_RAW_SHA256, MODEL_FEATURES, ORDER_ID, ORDER_TIMESTAMP, TARGET_COLUMN,
    chronological_partitions, evaluate_always_late, select_model_features,
)
from .data_loader import REPOSITORY_ROOT, load_raw_data, raw_file_metadata
from .data_validation import validate_order_ids, validate_target
from .leakage_guardrails import assert_no_group_overlap
from .preprocessing import clean_supply_chain_data
from .tree_evaluation import (
    DEFAULT_REPORT_PATH as TREE_REPORT_PATH,
    build_tree_pipeline, fit_tree, verify_reference_contract,
)


DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "reports" / "automl_validation.json"
DEFAULT_LEADERBOARD_PATH = REPOSITORY_ROOT / "reports" / "automl_leaderboard.csv"
TIE_TOLERANCE = 1e-12


def candidate_specs() -> list[tuple[str, Any, dict[str, list[Any]]]]:
    """Predeclared family/grid order is also the deterministic CV tie preference."""
    return [
        ("logistic_regression", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
         {"model__C": [0.1, 1.0]}),
        ("decision_tree", DecisionTreeClassifier(min_samples_leaf=100, random_state=42),
         {"model__max_depth": [1, 3]}),
        ("random_forest", RandomForestClassifier(
            n_estimators=100, min_samples_leaf=100, max_features=None, n_jobs=1, random_state=42,
         ), {"model__max_depth": [2, 3]}),
    ]


def make_inner_folds(
    training: pd.DataFrame, n_splits: int = 3
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Expand across unique timestamp cohorts; return row-position CV indices.

    All orders/items at the same timestamp stay on the same side. Folds use
    training metadata only and do not assume one item per order or sorted rows.
    """
    validate_order_ids(training)
    dates = training[ORDER_TIMESTAMP]
    if not is_datetime64_any_dtype(dates) or dates.isna().any():
        raise ValueError("Training timestamps must be parsed without missing values.")
    if training.groupby(ORDER_ID)[ORDER_TIMESTAMP].nunique().gt(1).any():
        raise ValueError("Conflicting timestamps within a training order.")
    timestamps = pd.Index(dates.unique()).sort_values()
    if len(timestamps) <= n_splits:
        raise ValueError("Not enough distinct training timestamps for chronological folds.")
    folds, audits = [], []
    for number, (earlier, later) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(timestamps), 1):
        fit_positions = np.flatnonzero((dates <= timestamps[earlier[-1]]).to_numpy())
        score_positions = np.flatnonzero((
            (dates >= timestamps[later[0]]) & (dates <= timestamps[later[-1]])
        ).to_numpy())
        fit_part, score_part = training.iloc[fit_positions], training.iloc[score_positions]
        assert_no_group_overlap(fit_part[ORDER_ID], score_part[ORDER_ID])
        if not fit_part[ORDER_TIMESTAMP].max() < score_part[ORDER_TIMESTAMP].min():
            raise ValueError("Inner fold is not strictly chronological.")
        folds.append((fit_positions, score_positions))
        audit: dict[str, Any] = {"fold": number, "overlapping_orders": 0}
        for label, part in [("fit", fit_part), ("score", score_part)]:
            audit[label] = {
                "items": len(part), "orders": int(part[ORDER_ID].nunique()),
                "first_order_timestamp": part[ORDER_TIMESTAMP].min().isoformat(),
                "last_order_timestamp": part[ORDER_TIMESTAMP].max().isoformat(),
            }
        audits.append(audit)
    return folds, audits


def late_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Explicitly use probabilities for the late class, never hard class labels."""
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(features)[:, positive_index]


def probability_average_precision(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> float:
    return float(average_precision_score(target, late_probabilities(model, features)))


def first_best(values: list[float]) -> int:
    """Prefer first declared setting/family when CV AP ties within tolerance."""
    best = max(values)
    return next(i for i, value in enumerate(values) if best - value <= TIE_TOLERANCE)


def tune_on_training(
    training: pd.DataFrame,
    specs: list[tuple[str, Any, dict[str, list[Any]]]] | None = None,
) -> tuple[dict[str, Pipeline], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Select settings and the preferred family without accepting outer validation."""
    training = training.sort_values([ORDER_TIMESTAMP, ORDER_ID], kind="stable").reset_index(drop=True)
    validate_target(training)
    folds, audits = make_inner_folds(training)
    for fit_positions, score_positions in folds:
        for positions in [fit_positions, score_positions]:
            if set(training.iloc[positions][TARGET_COLUMN].unique()) != {0, 1}:
                raise ValueError("Every inner fit/score partition must contain both classes.")
    features = select_model_features(training).replace({pd.NA: np.nan})
    target = training[TARGET_COLUMN]
    models, summaries = {}, []
    with threadpool_limits(limits=1):
        for name, estimator, grid in (candidate_specs() if specs is None else specs):
            preprocessing = clone(build_tree_pipeline().named_steps["preprocessing"])
            pipeline = Pipeline([("preprocessing", preprocessing), ("model", clone(estimator))])
            search = GridSearchCV(
                pipeline, grid, scoring=probability_average_precision, cv=folds,
                refit=False, n_jobs=1, error_score="raise", return_train_score=False,
            ).fit(features, target)
            means = [float(value) for value in search.cv_results_["mean_test_score"]]
            selected = first_best(means)
            settings = search.cv_results_["params"][selected]
            models[name] = clone(pipeline).set_params(**settings).fit(features, target)
            configurations = []
            for index, params in enumerate(search.cv_results_["params"]):
                configurations.append({
                    "settings": params,
                    "fold_average_precision": [
                        float(search.cv_results_[f"split{fold}_test_score"][index])
                        for fold in range(len(folds))
                    ],
                    "mean_average_precision": means[index],
                    "std_average_precision": float(search.cv_results_["std_test_score"][index]),
                })
            summaries.append({
                "model": name, "selected_settings": settings,
                "selected_cv_average_precision": means[selected],
                "refit_estimator_parameters": models[name].named_steps["model"].get_params(),
                "configurations": configurations,
            })
    preferred = first_best([row["selected_cv_average_precision"] for row in summaries])
    return models, summaries, audits, summaries[preferred]["model"]


def evaluate_on_validation(models: dict[str, Pipeline], validation: pd.DataFrame) -> list[dict[str, Any]]:
    validate_target(validation)
    if set(validation[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Validation must contain both classes.")
    features = select_model_features(validation).replace({pd.NA: np.nan})
    return [
        {"model": name, "average_precision": probability_average_precision(model, features, validation[TARGET_COLUMN]),
         "distinct_scores": int(np.unique(late_probabilities(model, features)).size)}
        for name, model in models.items()
    ]


def run_automl_evaluation(
    raw_path: str | Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    leaderboard_path: Path = DEFAULT_LEADERBOARD_PATH,
    baseline_report_path: Path = BASELINE_REPORT_PATH,
    tree_report_path: Path = TREE_REPORT_PATH,
) -> dict[str, Any]:
    """Verify existing contracts, discard final test, tune, then compare validation."""
    source = raw_file_metadata(raw_path)
    if source["sha256"] != EXPECTED_RAW_SHA256:
        raise ValueError("Raw source checksum differs from the agreed DataCo source.")
    baseline = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    tree_reference = json.loads(Path(tree_report_path).read_text(encoding="utf-8"))
    data, _ = clean_supply_chain_data(load_raw_data(raw_path))
    partitions = chronological_partitions(data)
    split = verify_reference_contract(partitions, str(source["sha256"]), baseline)
    if (tree_reference["source"]["sha256"] != source["sha256"]
        or tree_reference["model_features"] != list(MODEL_FEATURES)
        or tree_reference["split"] != split):
        raise ValueError("Tree reference source, features, or split differs.")
    training, validation = partitions.pop("train"), partitions.pop("validation")
    del partitions, data  # Final test cannot reach search, refitting, or scoring.
    models, search_results, fold_audit, preferred = tune_on_training(training)
    baseline_result = evaluate_always_late(validation[TARGET_COLUMN])
    baseline_ap = baseline_result["average_precision"]
    if (baseline_result["class_counts"] != baseline["validation_reference"]["class_counts"]
        or abs(baseline_ap - baseline["validation_reference"]["average_precision"]) > TIE_TOLERANCE):
        raise ValueError("Always-late validation reference was not reproduced.")
    fixed_tree = fit_tree(training)
    fixed_tree_ap = probability_average_precision(
        fixed_tree, select_model_features(validation), validation[TARGET_COLUMN]
    )
    recorded_tree_ap = next(row["average_precision"] for row in tree_reference["validation_comparison"]
                            if row["model"] == "shallow_decision_tree")
    if abs(fixed_tree_ap - recorded_tree_ap) > TIE_TOLERANCE:
        raise ValueError("Fixed-tree validation reference was not reproduced.")
    evaluated = evaluate_on_validation(models, validation)
    rows = [
        {"model": "always_predict_late", "kind": "reference", "validation_average_precision": baseline_ap,
         "inner_cv_average_precision": None, "settings": {}, "distinct_scores": 1},
        {"model": "depth_3_tree", "kind": "reference", "validation_average_precision": fixed_tree_ap,
         "inner_cv_average_precision": None, "settings": tree_reference["model"]["parameters"],
         "distinct_scores": int(np.unique(late_probabilities(fixed_tree, select_model_features(validation))).size)},
    ]
    for result, search in zip(evaluated, search_results):
        rows.append({
            "model": result["model"], "kind": "tuned_candidate",
            "validation_average_precision": result["average_precision"],
            "inner_cv_average_precision": search["selected_cv_average_precision"],
            "settings": search["selected_settings"], "distinct_scores": result["distinct_scores"],
        })
    rows.sort(key=lambda row: -row["validation_average_precision"])
    rank, previous = 0, None
    for position, row in enumerate(rows, 1):
        ap = row["validation_average_precision"]
        if previous is None or abs(previous - ap) > TIE_TOLERANCE:
            rank = position
        row.update({"rank": rank, "partition": "validation", "unit": "order_item",
                    "gain_vs_always_late": ap - baseline_ap, "gain_vs_depth_3_tree": ap - fixed_tree_ap})
        previous = ap
    profiles = select_model_features(training).drop_duplicates().sort_values(list(MODEL_FEATURES)).reset_index(drop=True)
    profile_scores = profiles.copy()
    for name, model in models.items():
        profile_scores[name] = late_probabilities(model, profiles)
    report = {
        "source": source, "target": TARGET_COLUMN, "positive_class": 1,
        "prediction_time": "order_placement", "model_features": list(MODEL_FEATURES),
        "split": split, "reference_contracts_verified": True,
        "primary_metric": "average_precision_from_late_class_probabilities",
        "inner_cv": {"method": "3 expanding folds over unique training timestamp cohorts",
                     "folds": fold_audit, "mean_rule": "unweighted mean of per-item AP across folds"},
        "search_results": search_results,
        "selection": {"preferred_model_by_training_cv": preferred,
                      "tie_tolerance": TIE_TOLERANCE,
                      "tie_rule": "first declared grid setting; family order logistic, tree, forest",
                      "outer_validation_used_for_tuning": False},
        "validation_leaderboard": rows,
        "training_shipping_profiles_and_candidate_probabilities": profile_scores.to_dict(orient="records"),
        "validation_shipping_profile_count": len(select_model_features(validation).drop_duplicates()),
        "preprocessing": "clone of shared tree preprocessing; fitted inside each fold and training-only refit",
        "final_test_evaluated": False, "operating_threshold": None,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                    "search_jobs": 1, "fit_thread_limit": 1},
    }
    report_path, leaderboard_path = Path(report_path), Path(leaderboard_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    table = pd.DataFrame(rows)
    table["settings"] = table["settings"].map(lambda settings: json.dumps(settings, sort_keys=True))
    table.to_csv(leaderboard_path, index=False, lineterminator="\n", float_format="%.12f")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--leaderboard-path", type=Path, default=DEFAULT_LEADERBOARD_PATH)
    args = parser.parse_args()
    report = run_automl_evaluation(args.raw_path, args.report_path, args.leaderboard_path)
    print(json.dumps({"selection": report["selection"], "leaderboard": report["validation_leaderboard"],
                      "final_test_evaluated": report["final_test_evaluated"], "operating_threshold": None}, indent=2))


if __name__ == "__main__":
    main()
