"""Behavioral checks for grouped time splits and validation-only evaluation."""

import pandas as pd
import pytest

from src.baseline_evaluation import (
    EXPECTED_RAW_SHA256,
    ORDER_TIMESTAMP,
    audit_partitions,
    chronological_partitions,
    evaluate_always_late,
    select_model_features,
    split_manifest,
)
from src import baseline_evaluation


@pytest.fixture
def items():
    return pd.DataFrame({
        "order_id": [order for order in range(10) for _ in range(2)],
        ORDER_TIMESTAMP: pd.to_datetime([
            f"2020-01-{order + 1:02d}" for order in range(10) for _ in range(2)
        ]),
        "late_delivery_risk": [0, 1] * 10,
        "days_for_shipment_scheduled": [2] * 20,
        "shipping_mode": ["Second Class"] * 20,
    })


def test_split_is_chronological_disjoint_complete_and_order_independent(items):
    parts = chronological_partitions(items)
    assert [len(part) for part in parts.values()] == [12, 4, 4]
    assert sorted(idx for part in parts.values() for idx in part.index) == list(items.index)
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(parts[left].order_id).isdisjoint(parts[right].order_id)
        assert parts[left][ORDER_TIMESTAMP].max() < parts[right][ORDER_TIMESTAMP].min()
    shuffled = chronological_partitions(items.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(split_manifest(parts), split_manifest(shuffled))


def test_timestamp_ties_at_both_boundaries_stay_together(items):
    items.loc[items.order_id.eq(5), ORDER_TIMESTAMP] = pd.Timestamp("2020-01-07")
    items.loc[items.order_id.eq(7), ORDER_TIMESTAMP] = pd.Timestamp("2020-01-09")
    parts = chronological_partitions(items)
    assert set(parts["train"].order_id) == set(range(5))
    assert set(parts["validation"].order_id) == {5, 6}
    assert set(parts["test"].order_id) == {7, 8, 9}


def test_split_requires_no_target_and_manifest_contains_no_labels(items):
    parts = chronological_partitions(items.drop(columns="late_delivery_risk"))
    manifest = split_manifest(parts)
    assert list(manifest.columns) == ["order_id", ORDER_TIMESTAMP, "partition"]
    assert len(manifest) == items.order_id.nunique()


@pytest.mark.parametrize("column", ["order_id", ORDER_TIMESTAMP])
def test_split_rejects_missing_structural_values(items, column):
    items.loc[0, column] = None
    with pytest.raises(ValueError, match="missing"):
        chronological_partitions(items)


def test_split_rejects_conflicting_order_timestamps(items):
    items.loc[0, ORDER_TIMESTAMP] = pd.Timestamp("2020-02-01")
    with pytest.raises(ValueError, match="Conflicting"):
        chronological_partitions(items)


def test_split_rejects_degenerate_timestamp_cohorts(items):
    items[ORDER_TIMESTAMP] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="empty"):
        chronological_partitions(items)


def test_audit_rejects_accidental_order_overlap(items):
    parts = chronological_partitions(items)
    parts["validation"] = pd.concat([parts["validation"], parts["train"].iloc[:1]])
    with pytest.raises(ValueError, match="Order-group leakage"):
        audit_partitions(parts)


def test_feature_allowlist_discards_labels_ids_dates_and_post_order_fields(items):
    for field in ["delivery_status", "days_for_shipping_real", "shipping_date_dateorders",
                  "order_status", "customer_email", "unknown_future_feature"]:
        items[field] = "must never reach a model"
    features = select_model_features(items)
    assert list(features.columns) == ["days_for_shipment_scheduled", "shipping_mode"]


def test_always_late_exposes_alert_costs_and_prevalence_reference():
    result = evaluate_always_late(pd.Series([0, 0, 0, 1]))
    assert result["average_precision"] == pytest.approx(0.25)
    assert result["precision"] == pytest.approx(0.25)
    assert result["recall"] == 1.0
    assert result["alert_rate"] == 1.0
    assert result["confusion_counts"] == {"tn": 0, "fp": 3, "fn": 0, "tp": 1}


def test_reference_rejects_invalid_or_single_class_labels():
    with pytest.raises(ValueError, match="outside"):
        evaluate_always_late(pd.Series([0, 2]))
    with pytest.raises(ValueError, match="both classes"):
        evaluate_always_late(pd.Series([1, 1]))


def test_scoring_does_not_read_final_test_labels_after_cleaning(items, monkeypatch, tmp_path):
    # After shared schema checks, missing test labels must not affect scoring.
    items.loc[items.order_id.ge(8), "late_delivery_risk"] = float("nan")
    monkeypatch.setattr(baseline_evaluation, "raw_file_metadata",
                        lambda path: {"sha256": EXPECTED_RAW_SHA256})
    monkeypatch.setattr(baseline_evaluation, "load_raw_data", lambda path: items)
    monkeypatch.setattr(baseline_evaluation, "clean_supply_chain_data",
                        lambda data: (data, {}))
    result = baseline_evaluation.run_baseline(
        report_path=tmp_path / "report.json", manifest_path=tmp_path / "split.csv"
    )
    assert result["final_test_evaluated"] is False
    assert result["validation_reference"]["average_precision"] == pytest.approx(0.5)
    assert "class_counts" not in result["split"]["partitions"]["test"]
