"""Distinct threshold policies, tied scores, and reserved-test isolation."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import alert_policy_evaluation as policies
from src.automl_evaluation import probability_average_precision
from src.baseline_evaluation import (
    EXPECTED_RAW_SHA256, MODEL_FEATURES, ORDER_ID, ORDER_TIMESTAMP, TARGET_COLUMN,
    audit_partitions, chronological_partitions, evaluate_always_late, select_model_features, split_manifest,
)


@pytest.fixture
def validation():
    return pd.DataFrame({
        "shipping_mode": np.repeat(["A", "B", "C", "D"], 2),
        "days_for_shipment_scheduled": np.repeat([1, 2, 3, 4], 2),
        TARGET_COLUMN: [1, 0, 1, 1, 0, 1, 0, 0],
    })


def test_enumeration_includes_each_policy_once_and_reports_item_counts(validation):
    scores = np.repeat([0.9, 0.7, 0.4, 0.1], 2)
    rows = policies.enumerate_alert_policies(validation, scores)["policies"]
    assert [row["total_alerts"] for row in rows] == [0, 2, 4, 6, 8]
    assert rows[0]["precision"] is None
    assert rows[0]["recall"] == 0.0
    assert rows[0]["missed_late_items"] == 4
    assert rows[0]["alerted_shipping_profiles"] == []
    assert rows[2]["precision"] == 0.75
    assert rows[2]["recall"] == 0.75
    assert rows[2]["false_alerts"] == 1
    assert rows[2]["missed_late_items"] == 1
    assert [p["shipping_mode"] for p in rows[2]["alerted_shipping_profiles"]] == ["A", "B"]
    assert rows[-1]["recall"] == 1.0
    assert rows[-1]["false_alerts"] == 4
    for row in rows:
        assert row["unit"] == "order_item"
        assert row["true_alerts"] + row["false_alerts"] == row["total_alerts"]
        assert row["true_alerts"] + row["missed_late_items"] == 4


def test_threshold_bounds_use_inclusive_alert_rule_and_preserve_tied_profiles(validation):
    scores = np.repeat([1.0, 0.7, 0.7, 0.0], 2)
    rows = policies.enumerate_alert_policies(validation, scores)["policies"]
    assert [row["total_alerts"] for row in rows] == [0, 2, 6, 8]
    for row in rows:
        assert int((scores >= row["representative_threshold"]).sum()) == row["total_alerts"]
        lower = row["threshold_interval"]["lower_exclusive"]
        upper = row["threshold_interval"]["upper_inclusive"]
        if lower is not None:
            assert row["representative_threshold"] > lower
        if upper is not None:
            assert row["representative_threshold"] <= upper
            assert int((scores >= upper).sum()) == row["total_alerts"]
    assert rows[0]["representative_threshold"] > 1.0


def test_nearby_distinct_scores_are_not_merged_by_display_rounding(validation):
    scores = np.repeat([0.9, np.nextafter(0.5, 1.0), 0.5, 0.1], 2)
    assert len(policies.enumerate_alert_policies(validation, scores)["policies"]) == 5


@pytest.mark.parametrize("problem", ["nan", "out_of_range", "wrong_length", "inconsistent_profile"])
def test_invalid_predictions_fail(validation, problem):
    scores = np.repeat([0.9, 0.7, 0.4, 0.1], 2)
    if problem == "nan":
        scores[0] = np.nan
    elif problem == "out_of_range":
        scores[0] = 1.1
    elif problem == "wrong_length":
        scores = scores[:-1]
    else:
        scores[0] = 0.95
    with pytest.raises(ValueError):
        policies.enumerate_alert_policies(validation, scores)


def test_policy_equivalence_checks_actual_alerted_items_not_only_counts(validation):
    class ScoreModel:
        classes_ = np.array([0, 1])

        def __init__(self, scores):
            self.scores = np.repeat(scores, 2)

        def predict_proba(self, features):
            return np.column_stack([1 - self.scores, self.scores])

    first = ScoreModel([0.9, 0.7, 0.4, 0.1])
    equivalent = ScoreModel([0.95, 0.75, 0.45, 0.35])
    different = ScoreModel([0.7, 0.9, 0.4, 0.1])
    assert policies.evaluate_validation_policies({"a": first, "b": equivalent}, validation)[
        "same_alert_policies_on_every_validation_item"] is True
    assert policies.evaluate_validation_policies({"a": first, "b": different}, validation)[
        "same_alert_policies_on_every_validation_item"] is False


def test_runner_uses_fixed_training_fits_and_never_reads_test_predictions(monkeypatch, tmp_path):
    orders = np.repeat(np.arange(20), 20)
    items = pd.DataFrame({
        ORDER_ID: orders, ORDER_TIMESTAMP: pd.Timestamp("2020-01-01") + pd.to_timedelta(orders, unit="D"),
        TARGET_COLUMN: orders % 2,
        "days_for_shipment_scheduled": np.where(orders % 2, 1.0, 4.0),
        "shipping_mode": np.where(orders % 2, "First Class", "Standard Class"),
        "delivery_status": "forbidden",
    })
    parts = chronological_partitions(items)
    text = split_manifest(parts).to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n")
    split = {"manifest_sha256": hashlib.sha256(text.encode()).hexdigest(), **audit_partitions(parts)}
    source = {"sha256": EXPECTED_RAW_SHA256}
    tree_parameters = {"max_depth": 3, "min_samples_leaf": 100, "random_state": 42}
    logistic_parameters = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=42).get_params()
    models = policies.fit_recorded_configurations(parts["train"], tree_parameters, logistic_parameters)
    baseline = {"source": source, "prospective_model_features": list(MODEL_FEATURES), "split": split,
                "validation_reference": evaluate_always_late(parts["validation"][TARGET_COLUMN])}
    tree = {"source": source, "model_features": list(MODEL_FEATURES), "split": split,
            "model": {"parameters": tree_parameters}}
    automl = {"source": source, "model_features": list(MODEL_FEATURES), "split": split,
              "search_results": [{"model": "logistic_regression", "selected_settings": {"model__C": 0.1},
                                  "refit_estimator_parameters": logistic_parameters}],
              "validation_leaderboard": [
                  {"model": old_name, "validation_average_precision": probability_average_precision(
                      models[name], select_model_features(parts["validation"]), parts["validation"][TARGET_COLUMN])}
                  for name, old_name in [("depth_3_tree", "depth_3_tree"), ("logistic_regression_C_0_1", "logistic_regression")]
              ]}
    paths = [tmp_path / name for name in ["baseline.json", "tree.json", "automl.json"]]
    for path, report in zip(paths, [baseline, tree, automl]):
        path.write_text(json.dumps(report), encoding="utf-8")
    originals = [path.read_bytes() for path in paths]
    test_mask = items[ORDER_ID].isin(parts["test"][ORDER_ID])
    items.loc[test_mask, TARGET_COLUMN] = np.nan
    items.loc[test_mask, "days_for_shipment_scheduled"] = np.inf
    monkeypatch.setattr(policies, "raw_file_metadata", lambda path: source)
    monkeypatch.setattr(policies, "load_raw_data", lambda path: items)
    monkeypatch.setattr(policies, "clean_supply_chain_data", lambda data: (data, {}))
    original_fit = policies.fit_recorded_configurations

    def checked_fit(training, tree_params, logistic_params):
        assert set(training[ORDER_ID]) == set(parts["train"][ORDER_ID])
        assert tree_params == tree_parameters and logistic_params == logistic_parameters
        return original_fit(training, tree_params, logistic_params)

    monkeypatch.setattr(policies, "fit_recorded_configurations", checked_fit)
    report = policies.run_alert_policy_evaluation(
        report_path=tmp_path / "policies.json", table_path=tmp_path / "policies.csv",
        baseline_report_path=paths[0], tree_report_path=paths[1], automl_report_path=paths[2],
    )
    assert report["final_test_evaluated"] is False
    assert report["operating_point"] is None
    assert report["same_alert_policies_on_every_validation_item"] is True
    assert "class_counts" not in report["split"]["partitions"]["test"]
    assert [path.read_bytes() for path in paths] == originals
