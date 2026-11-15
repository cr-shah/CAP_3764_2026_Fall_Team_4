"""Grouped temporal CV, fold-local preprocessing, and held-out isolation checks."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

from src import automl_evaluation as automl
from src.baseline_evaluation import (
    EXPECTED_RAW_SHA256, MODEL_FEATURES, ORDER_ID, ORDER_TIMESTAMP, TARGET_COLUMN,
    audit_partitions, chronological_partitions, evaluate_always_late, select_model_features, split_manifest,
)
from src.tree_evaluation import fit_tree


@pytest.fixture
def items():
    orders = np.repeat(np.arange(32), 3)
    return pd.DataFrame({
        ORDER_ID: orders,
        ORDER_TIMESTAMP: pd.Timestamp("2020-01-01") + pd.to_timedelta(orders // 2, unit="D"),
        TARGET_COLUMN: orders % 2,
        "days_for_shipment_scheduled": np.where(orders % 2, 1.0, 4.0),
        "shipping_mode": np.where(orders % 2, "First Class", "Standard Class"),
        "delivery_status": "forbidden",
    })


def small_specs():
    return [("logistic_regression", LogisticRegression(max_iter=1000), {"model__C": [0.1]})]


def test_inner_folds_preserve_whole_orders_tied_dates_and_expanding_history(items):
    shuffled = items.sample(frac=1, random_state=11)
    folds, audit = automl.make_inner_folds(shuffled.drop(columns=TARGET_COLUMN))
    previous_fit, scored_orders = set(), set()
    for (fit_positions, score_positions), entry in zip(folds, audit):
        fit, score = shuffled.iloc[fit_positions], shuffled.iloc[score_positions]
        assert set(fit[ORDER_ID]).isdisjoint(score[ORDER_ID])
        assert set(fit[ORDER_TIMESTAMP]).isdisjoint(score[ORDER_TIMESTAMP])
        assert fit[ORDER_TIMESTAMP].max() < score[ORDER_TIMESTAMP].min()
        assert previous_fit.issubset(set(fit[ORDER_ID]))
        assert scored_orders.isdisjoint(set(score[ORDER_ID]))
        previous_fit = set(fit[ORDER_ID])
        scored_orders.update(score[ORDER_ID])
        for part in [fit, score]:
            for order in part[ORDER_ID].unique():
                assert len(part[part[ORDER_ID].eq(order)]) == len(items[items[ORDER_ID].eq(order)])
        assert entry["overlapping_orders"] == 0
    assert len(folds) == 3


@pytest.mark.parametrize("problem", ["missing_date", "conflicting_order", "one_timestamp"])
def test_inner_folds_reject_invalid_time_structure(items, problem):
    if problem == "missing_date":
        items.loc[0, ORDER_TIMESTAMP] = pd.NaT
    elif problem == "conflicting_order":
        items.loc[0, ORDER_TIMESTAMP] = pd.Timestamp("2019-12-01")
    else:
        items[ORDER_TIMESTAMP] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError):
        automl.make_inner_folds(items)


def test_search_fits_encoder_inside_each_fold_then_refits_on_training(items, monkeypatch):
    items.loc[items[ORDER_ID].ge(24), "shipping_mode"] = "LATER_ONLY"
    fitted_categories = []
    original_fit = OneHotEncoder.fit

    def record_fit(self, features, y=None):
        result = original_fit(self, features, y)
        fitted_categories.append((len(features), set(self.categories_[0])))
        return result

    monkeypatch.setattr(OneHotEncoder, "fit", record_fit)
    models, summaries, _, preferred = automl.tune_on_training(items, small_specs())
    folds, _ = automl.make_inner_folds(items)
    expected = [
        (len(fit_positions), set(items.iloc[fit_positions]["shipping_mode"]))
        for fit_positions, _ in folds
    ] + [(len(items), set(items["shipping_mode"]))]
    assert fitted_categories == expected
    assert "LATER_ONLY" not in fitted_categories[0][1]
    assert list(models[preferred].named_steps["preprocessing"].feature_names_in_) == list(MODEL_FEATURES)
    assert len(summaries[0]["configurations"][0]["fold_average_precision"]) == 3


def test_probability_scorer_uses_late_class_even_when_classes_are_reversed():
    scores = np.array([0.1, 0.4, 0.7, 0.9])

    class ProbabilityModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, features):
            return np.column_stack([scores, 1 - scores])

    target = pd.Series([0, 1, 0, 1])
    ap = automl.probability_average_precision(ProbabilityModel(), pd.DataFrame(index=range(4)), target)
    assert ap == pytest.approx(average_precision_score(target, scores))
    assert ap != pytest.approx(average_precision_score(target, scores >= 0.5))


def test_cv_ties_use_predeclared_order():
    assert automl.first_best([0.7, 0.7 + 1e-13, 0.69]) == 0
    assert automl.first_best([0.7, 0.71]) == 1


def test_single_class_inner_fold_fails_instead_of_silently_scoring(items):
    items.loc[items[ORDER_ID].lt(8), TARGET_COLUMN] = 1
    with pytest.raises(ValueError, match="both classes"):
        automl.tune_on_training(items, small_specs())


def test_runner_keeps_outer_validation_out_of_tuning_and_final_test_out_of_scoring(items, monkeypatch, tmp_path):
    parts = chronological_partitions(items)
    manifest = split_manifest(parts).to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n")
    split = {"manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(), **audit_partitions(parts)}
    source = {"sha256": EXPECTED_RAW_SHA256}
    baseline = {
        "source": source, "prospective_model_features": list(MODEL_FEATURES), "split": split,
        "validation_reference": evaluate_always_late(parts["validation"][TARGET_COLUMN]),
    }
    fixed_tree = fit_tree(parts["train"])
    tree = {
        "source": source, "model_features": list(MODEL_FEATURES), "split": split,
        "model": {"parameters": {"max_depth": 3, "min_samples_leaf": 100, "random_state": 42}},
        "validation_comparison": [{
            "model": "shallow_decision_tree",
            "average_precision": automl.probability_average_precision(
                fixed_tree, select_model_features(parts["validation"]), parts["validation"][TARGET_COLUMN]
            ),
        }],
    }
    baseline_path, tree_path = tmp_path / "baseline.json", tmp_path / "tree.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    originals = baseline_path.read_bytes(), tree_path.read_bytes()
    test_mask = items[ORDER_ID].isin(parts["test"][ORDER_ID])
    items.loc[test_mask, TARGET_COLUMN] = np.nan
    items.loc[test_mask, "days_for_shipment_scheduled"] = np.inf
    items.loc[test_mask, "shipping_mode"] = "TEST_ONLY"
    monkeypatch.setattr(automl, "raw_file_metadata", lambda path: source)
    monkeypatch.setattr(automl, "load_raw_data", lambda path: items)
    monkeypatch.setattr(automl, "clean_supply_chain_data", lambda data: (data, {}))
    monkeypatch.setattr(automl, "candidate_specs", small_specs)
    original_tune = automl.tune_on_training

    def checked_tune(training):
        assert set(training[ORDER_ID]) == set(parts["train"][ORDER_ID])
        return original_tune(training)

    monkeypatch.setattr(automl, "tune_on_training", checked_tune)
    report = automl.run_automl_evaluation(
        report_path=tmp_path / "report.json", leaderboard_path=tmp_path / "leaderboard.csv",
        baseline_report_path=baseline_path, tree_report_path=tree_path,
    )
    assert report["reference_contracts_verified"] is True
    assert report["final_test_evaluated"] is False
    assert report["operating_threshold"] is None
    assert report["selection"]["outer_validation_used_for_tuning"] is False
    assert all(row["partition"] == "validation" for row in report["validation_leaderboard"])
    assert "class_counts" not in report["split"]["partitions"]["test"]
    assert (baseline_path.read_bytes(), tree_path.read_bytes()) == originals
