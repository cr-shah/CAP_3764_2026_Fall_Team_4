"""Tests for supplemental unique-order evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.order_level_evaluation import (
    aggregate_orders,
    report_path_label,
    score_orders,
    verify_tracked_item_results,
)


def repeated_order_items() -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.DataFrame(
        {
            "order_id": [101, 101, 202, 202],
            "late_delivery_risk": [1, 1, 0, 0],
            "days_for_shipment_scheduled": [2, 2, 4, 4],
            "shipping_mode": ["Second Class", "Second Class", "Standard Class", "Standard Class"],
            "order_date_dateorders": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
            ),
        }
    )
    return data, np.array([0.8, 0.8, 0.3, 0.3])


def test_aggregate_orders_weights_each_order_once() -> None:
    data, scores = repeated_order_items()

    orders = aggregate_orders(data, scores)

    assert orders["order_id"].tolist() == [101, 202]
    assert orders["late_delivery_risk"].tolist() == [1, 0]
    assert orders["score"].tolist() == [0.8, 0.3]


@pytest.mark.parametrize(
    "column,replacement",
    [
        ("late_delivery_risk", 0),
        ("days_for_shipment_scheduled", 99),
        ("shipping_mode", "First Class"),
        ("order_date_dateorders", pd.Timestamp("2020-02-01")),
    ],
)
def test_aggregate_orders_rejects_inconsistent_order_fields(
    column: str, replacement: object
) -> None:
    data, scores = repeated_order_items()
    data.loc[1, column] = replacement

    with pytest.raises(ValueError, match=column):
        aggregate_orders(data, scores)


def test_aggregate_orders_rejects_inconsistent_scores() -> None:
    data, scores = repeated_order_items()
    scores[1] = 0.7

    with pytest.raises(ValueError, match="score"):
        aggregate_orders(data, scores)


def test_aggregate_orders_requires_nonmissing_order_id() -> None:
    data, scores = repeated_order_items()
    data.loc[0, "order_id"] = np.nan

    with pytest.raises(ValueError, match="order_id"):
        aggregate_orders(data, scores)


def test_score_orders_uses_one_vote_per_order_and_inclusive_threshold() -> None:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "late_delivery_risk": [1, 0, 1, 0],
            "score": [0.9, 0.5, 0.4, 0.1],
        }
    )

    result = score_orders(orders, partition="validation", threshold=0.5)

    assert result["model"] == "frozen_depth_3_tree"
    assert result["partition"] == "validation"
    assert result["unit"] == "unique_order"
    assert result["post_hoc_supplemental"] is True
    assert result["orders"] == 4
    assert result["late_orders"] == 2
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_alerts"] == 1
    assert result["missed_late_orders"] == 1
    assert result["total_alerts"] == 2
    assert result["true_alerts"] == 1
    assert result["correct_non_alerts"] == 1
    assert result["alert_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "column,values",
    [
        ("late_delivery_risk", [1, 1]),
        ("late_delivery_risk", [0, 2]),
        ("score", [0.1, np.nan]),
        ("score", [0.1, 1.1]),
    ],
)
def test_score_orders_rejects_invalid_targets_or_scores(column: str, values: list[float]) -> None:
    orders = pd.DataFrame(
        {"order_id": [1, 2], "late_delivery_risk": [0, 1], "score": [0.1, 0.9]}
    )
    orders[column] = values

    with pytest.raises(ValueError):
        score_orders(orders, partition="test", threshold=0.5)


def tracked_contract() -> tuple[dict, list[dict]]:
    rows = [
        {
            "model": "frozen_depth_3_tree",
            "partition": partition,
            "unit": "order_item",
            "items": 10,
            "late_items": 6,
            "late_prevalence": 0.6,
            "threshold": 0.5,
            "average_precision": 0.75,
            "precision": 0.8,
            "recall": 2 / 3,
            "false_alerts": 1,
            "missed_late_items": 2,
            "total_alerts": 5,
            "true_alerts": 4,
            "correct_non_alerts": 3,
            "alert_rate": 0.5,
        }
        for partition in ("validation", "test")
    ]
    tracked = {
        "configuration_id": "frozen-tree",
        "configuration_sha256": "config-hash",
        "training_fitted_model_fingerprint": "model-hash",
        "validation_reference_reproduced_before_test": True,
        "model_and_configuration_unchanged_after_scoring": True,
        "results": rows,
    }
    return tracked, rows


def test_tracked_item_contract_accepts_exact_reproduction() -> None:
    tracked, rows = tracked_contract()

    verify_tracked_item_results(
        {"configuration_id": "frozen-tree"},
        "config-hash",
        "model-hash",
        rows,
        tracked,
    )


def test_tracked_item_contract_rejects_changed_configuration() -> None:
    tracked, rows = tracked_contract()

    with pytest.raises(ValueError, match="configuration"):
        verify_tracked_item_results(
            {"configuration_id": "changed"}, "config-hash", "model-hash", rows, tracked
        )


def test_tracked_item_contract_rejects_changed_fingerprint() -> None:
    tracked, rows = tracked_contract()

    with pytest.raises(ValueError, match="fingerprint"):
        verify_tracked_item_results(
            {"configuration_id": "frozen-tree"},
            "config-hash",
            "different-model",
            rows,
            tracked,
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("unit", "unique_order"),
        ("items", 11),
        ("late_prevalence", 0.61),
        ("threshold", 0.6),
        ("average_precision", 0.76),
        ("total_alerts", 6),
        ("true_alerts", 5),
        ("alert_rate", 0.6),
    ],
)
def test_tracked_item_contract_rejects_changed_reproduced_result(
    field: str, replacement: object
) -> None:
    tracked, rows = tracked_contract()
    changed = [dict(row) for row in rows]
    changed[0][field] = replacement

    with pytest.raises(ValueError, match="validation"):
        verify_tracked_item_results(
            {"configuration_id": "frozen-tree"},
            "config-hash",
            "model-hash",
            changed,
            tracked,
        )


def test_tracked_item_contract_requires_validation_and_test_rows() -> None:
    tracked, rows = tracked_contract()

    with pytest.raises(ValueError, match="partition set"):
        verify_tracked_item_results(
            {"configuration_id": "frozen-tree"},
            "config-hash",
            "model-hash",
            rows[:1],
            tracked,
        )


@pytest.mark.parametrize("corruption", ["duplicate", "extra"])
def test_tracked_item_contract_rejects_duplicate_or_extra_partitions(corruption: str) -> None:
    tracked, rows = tracked_contract()
    added = dict(rows[0])
    if corruption == "extra":
        added["partition"] = "future"
    tracked["results"] = [added, *rows]

    with pytest.raises(ValueError, match="Tracked item-level partition set"):
        verify_tracked_item_results(
            {"configuration_id": "frozen-tree"},
            "config-hash",
            "model-hash",
            rows,
            tracked,
        )


def test_report_path_label_falls_back_for_external_contract(tmp_path) -> None:
    external = tmp_path / "final.json"

    assert report_path_label(external) == str(external.resolve())
