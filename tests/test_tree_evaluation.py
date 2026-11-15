"""Checks for training-only preprocessing, split reuse, and test-set isolation."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import tree_evaluation
from src.baseline_evaluation import (
    EXPECTED_RAW_SHA256, MODEL_FEATURES, ORDER_TIMESTAMP, TARGET_COLUMN,
    audit_partitions, chronological_partitions, evaluate_always_late, split_manifest,
)
from src.tree_evaluation import compare_on_validation, verify_reference_contract


@pytest.fixture
def items():
    orders = np.repeat(np.arange(20), 20)
    return pd.DataFrame({
        "order_id": orders,
        ORDER_TIMESTAMP: pd.to_datetime("2020-01-01") + pd.to_timedelta(orders, unit="D"),
        TARGET_COLUMN: orders % 2,
        "days_for_shipment_scheduled": np.where(orders % 2, 1.0, 4.0),
        "shipping_mode": np.where(orders % 2, "First Class", "Standard Class"),
        "delivery_status": "forbidden outcome",
    })


def reference_for(parts):
    text = split_manifest(parts).to_csv(
        index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n"
    )
    return {
        "source": {"sha256": EXPECTED_RAW_SHA256},
        "prospective_model_features": list(MODEL_FEATURES),
        "split": {
            "manifest_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            **audit_partitions(parts),
        },
        "validation_reference": evaluate_always_late(parts["validation"][TARGET_COLUMN]),
    }


def test_preprocessing_learns_training_values_only_and_accepts_unseen_validation():
    training = pd.DataFrame({
        "days_for_shipment_scheduled": [1.0, 1.0, 3.0, np.nan] * 100,
        "shipping_mode": ["A", "A", "B", np.nan] * 100,
        TARGET_COLUMN: [0, 0, 1, 1] * 100,
        "order_id": np.arange(400),
        "delivery_status": ["must be excluded"] * 400,
    })
    validation = pd.DataFrame({
        "days_for_shipment_scheduled": [np.nan, 99.0],
        "shipping_mode": ["VALIDATION_ONLY", np.nan],
        TARGET_COLUMN: [0, 1],
    })
    model, results = compare_on_validation(training, validation)
    preprocessing = model.named_steps["preprocessing"]
    numeric = preprocessing.named_transformers_["scheduled_days"]
    categorical = preprocessing.named_transformers_["shipping_service"]
    assert numeric.named_steps["impute"].statistics_[0] == 1.0
    assert categorical.named_steps["impute"].statistics_[0] == "A"
    assert set(categorical.named_steps["encode"].categories_[0]) == {"A", "B"}
    assert list(preprocessing.feature_names_in_) == list(MODEL_FEATURES)
    assert np.isfinite(results[1]["average_precision"])


def test_ap_uses_probabilities_and_repeats_deterministically():
    training = pd.DataFrame({
        "days_for_shipment_scheduled": np.repeat([1.0, 2.0, 4.0], 100),
        "shipping_mode": np.repeat(["A", "B", "C"], 100),
        TARGET_COLUMN: [0] * 80 + [1] * 20 + [0] * 40 + [1] * 60 + [0] * 10 + [1] * 90,
    })
    validation = pd.DataFrame({
        "days_for_shipment_scheduled": np.repeat([1.0, 2.0, 4.0], 10),
        "shipping_mode": np.repeat(["A", "B", "C"], 10),
        TARGET_COLUMN: [0] * 8 + [1] * 2 + [0] * 4 + [1] * 6 + [0] * 1 + [1] * 9,
    })
    model, results = compare_on_validation(training, validation)
    _, repeated = compare_on_validation(training, validation)
    features = validation[list(MODEL_FEATURES)]
    scores = model.predict_proba(features)[:, list(model.classes_).index(1)]
    assert results[1]["average_precision"] == pytest.approx(
        average_precision_score(validation[TARGET_COLUMN], scores)
    )
    assert results[1]["average_precision"] != pytest.approx(
        average_precision_score(validation[TARGET_COLUMN], (scores >= 0.5).astype(int))
    )
    assert results == repeated
    assert results[1]["average_precision"] > results[0]["average_precision"]


@pytest.mark.parametrize("changed", ["source", "features", "manifest"])
def test_reference_contract_rejects_comparison_drift(items, changed):
    parts = chronological_partitions(items)
    reference = reference_for(parts)
    if changed == "source":
        reference["source"]["sha256"] = "wrong"
    elif changed == "features":
        reference["prospective_model_features"].append("delivery_status")
    else:
        reference["split"]["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="differs"):
        verify_reference_contract(parts, EXPECTED_RAW_SHA256, reference)


def test_runner_never_fits_transforms_or_scores_final_test(items, monkeypatch, tmp_path):
    parts = chronological_partitions(items)
    reference = reference_for(parts)
    # After shared cleaning/schema validation, test labels and predictors must
    # never be needed. Infinity would fail tree preprocessing or prediction.
    test_mask = items.order_id.ge(16)
    items.loc[test_mask, TARGET_COLUMN] = np.nan
    items.loc[test_mask, "days_for_shipment_scheduled"] = np.inf
    items.loc[test_mask, "shipping_mode"] = "TEST_ONLY"
    monkeypatch.setattr(tree_evaluation, "raw_file_metadata",
                        lambda path: {"sha256": EXPECTED_RAW_SHA256})
    monkeypatch.setattr(tree_evaluation, "load_raw_data", lambda path: items)
    monkeypatch.setattr(tree_evaluation, "clean_supply_chain_data", lambda data: (data, {}))
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    original_reference = reference_path.read_bytes()
    result = tree_evaluation.run_tree_evaluation(
        report_path=tmp_path / "tree.json", reference_report_path=reference_path,
    )
    assert result["final_test_evaluated"] is False
    assert result["reference_contract_verified"] is True
    assert result["validation_comparison"][0] == reference["validation_reference"]
    assert "class_counts" not in result["split"]["partitions"]["test"]
    assert reference_path.read_bytes() == original_reference


def test_single_class_training_is_rejected(items):
    parts = chronological_partitions(items)
    parts["train"][TARGET_COLUMN] = 1
    with pytest.raises(ValueError, match="Training must contain both classes"):
        compare_on_validation(parts["train"], parts["validation"])
