"""Frozen configuration, scoring arithmetic, and pre-test protocol checks."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import final_evaluation as final
from src.baseline_evaluation import (
    EXPECTED_RAW_SHA256, MODEL_FEATURES, ORDER_ID, ORDER_TIMESTAMP, TARGET_COLUMN,
    audit_partitions, chronological_partitions, split_manifest,
)
import hashlib


def test_per_item_arithmetic_and_threshold_equality():
    target = pd.Series([1, 0, 0, 1])
    tree = final.score_predictions(target, np.array([0.5, 0.49, 0.9, 0.1]), "tree", "test", 0.5)
    assert tree["unit"] == "order_item" and tree["partition"] == "test"
    assert tree["precision"] == 0.5 and tree["recall"] == 0.5
    assert tree["false_alerts"] == 1 and tree["missed_late_items"] == 1
    assert tree["total_alerts"] == 2
    always = final.score_predictions(target, np.ones(4), "always_predict_late", "test", 0.5)
    assert always["average_precision"] == 0.5 and always["precision"] == 0.5
    assert always["recall"] == 1 and always["false_alerts"] == 2
    assert always["missed_late_items"] == 0 and always["total_alerts"] == 4


def test_text_hash_is_portable_across_line_endings(tmp_path):
    a, b = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    a.write_bytes(b"first\nsecond\n")
    b.write_bytes(b"first\r\nsecond\r\n")
    assert final.text_sha256(a) == final.text_sha256(b)


@pytest.mark.parametrize("change", ["threshold", "features", "training", "parameters"])
def test_rejects_changes_to_frozen_protocol(tmp_path, change):
    config = json.loads(final.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if change == "threshold":
        config["alert_threshold"] = 0.6
    elif change == "features":
        config["features"].append("delivery_status")
    elif change == "training":
        config["refit_with_validation"] = True
    else:
        config["estimator_parameters"]["max_depth"] = 4
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration differs|tree parameters differ"):
        final.load_frozen_configuration(path)


@pytest.fixture
def frozen_case(tmp_path, monkeypatch):
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
    params = final.build_tree_pipeline().named_steps["tree"].get_params()
    baseline = {"source": {"sha256": EXPECTED_RAW_SHA256}, "prospective_model_features": list(MODEL_FEATURES), "split": split}
    reference_rows = []
    for name, ap, precision, fp, tn in [("shallow_decision_tree", 1.0, 1.0, 0, 40),
                                       ("always_predict_late", 0.5, 0.5, 40, 0)]:
        reference_rows.append({"model": name, "average_precision": ap, "precision": precision, "recall": 1.0,
                               "confusion_counts": {"tp": 40, "fp": fp, "fn": 0, "tn": tn}})
    tree_reference = {"split": split, "model": {"parameters": params}, "validation_comparison": reference_rows}
    baseline_path, tree_path = tmp_path / "baseline.json", tmp_path / "tree.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    tree_path.write_text(json.dumps(tree_reference), encoding="utf-8")
    config = copy.deepcopy(json.loads(final.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")))
    config.update({"split": split, "baseline_report_path": str(baseline_path), "tree_report_path": str(tree_path),
                   "reference_report_sha256": {str(p): final.text_sha256(p) for p in [baseline_path, tree_path]}})
    config_path = tmp_path / "frozen.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    # Reverse only held-out outcomes: a model that never trained on them must fail
    # on these labels even though its validation predictions are perfect.
    mask = items[ORDER_ID].isin(parts["test"][ORDER_ID])
    items.loc[mask, TARGET_COLUMN] = 1 - items.loc[mask, TARGET_COLUMN]
    monkeypatch.setattr(final, "raw_file_metadata", lambda path: {"sha256": EXPECTED_RAW_SHA256})
    monkeypatch.setattr(final, "load_raw_data", lambda path: items)
    monkeypatch.setattr(final, "clean_supply_chain_data", lambda data: (data, {}))
    return items, parts, config_path


def test_final_test_is_scored_only_after_freeze_fit_and_validation_checks(frozen_case, monkeypatch, tmp_path):
    _, parts, config_path = frozen_case
    original_config = config_path.read_bytes()
    events = []
    original_fit, original_verify, original_score = final.fit_tree, final.verify_validation_results, final.score_partition

    def checked_fit(training):
        assert config_path.read_bytes() == original_config
        assert set(training[ORDER_ID]) == set(parts["train"][ORDER_ID])
        events.append("fit_train_only")
        return original_fit(training)

    def checked_verify(rows, reference):
        original_verify(rows, reference)
        events.append("validation_verified")

    def checked_score(model, data, name, threshold):
        assert config_path.read_bytes() == original_config
        assert threshold == 0.5
        if name == "test":
            assert events == ["fit_train_only", "score_validation", "validation_verified"]
        events.append("score_" + name)
        return original_score(model, data, name, threshold)

    monkeypatch.setattr(final, "fit_tree", checked_fit)
    monkeypatch.setattr(final, "verify_validation_results", checked_verify)
    monkeypatch.setattr(final, "score_partition", checked_score)
    result = final.run_final_evaluation(configuration_path=config_path,
                                        report_path=tmp_path / "result.json", table_path=tmp_path / "result.csv")
    assert events == ["fit_train_only", "score_validation", "validation_verified", "score_test"]
    test_tree, test_always = result["results"][2:]
    assert test_tree["false_alerts"] == 40 and test_tree["missed_late_items"] == 40
    assert test_tree["precision"] == 0 and test_tree["recall"] == 0
    assert test_tree["items"] == test_always["items"] == 80
    assert result["model_and_configuration_unchanged_after_scoring"] is True
    assert result["validation_added_to_training"] is False
    assert result["final_test_evaluated"] is True
    assert config_path.read_bytes() == original_config


def test_validation_mismatch_blocks_test_scoring(frozen_case, monkeypatch, tmp_path):
    _, _, config_path = frozen_case
    calls = []
    original_score = final.score_partition

    def mismatched_score(model, data, name, threshold):
        calls.append(name)
        rows, profiles = original_score(model, data, name, threshold)
        rows[0]["recall"] = 0.25
        return rows, profiles

    monkeypatch.setattr(final, "score_partition", mismatched_score)
    with pytest.raises(ValueError, match="Validation recall"):
        final.run_final_evaluation(configuration_path=config_path,
                                   report_path=tmp_path / "result.json", table_path=tmp_path / "result.csv")
    assert calls == ["validation"]


def test_code_or_reference_changes_invalidate_freeze(tmp_path):
    config = json.loads(final.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["implementation_sha256"]["src/preprocessing.py"] = "incorrect hash"
    path = tmp_path / "bad_hash.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen implementation_sha256 mismatch"):
        final.load_frozen_configuration(path)
