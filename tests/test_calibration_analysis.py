"""Tests for validation-only calibration and profile error analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration_analysis import (
    build_profile_table,
    calculate_calibration_metrics,
    run_calibration_analysis,
)
from src.data_loader import REPOSITORY_ROOT


def synthetic_profiles() -> list[dict]:
    return [
        {
            "days_for_shipment_scheduled": 1,
            "shipping_mode": "First Class",
            "items": 10,
            "late_items": 6,
            "late_probability": 0.8,
            "late_rate": 0.6,
            "item_share": 0.5,
            "alerts": True,
            "partition": "validation",
        },
        {
            "days_for_shipment_scheduled": 4,
            "shipping_mode": "Standard Class",
            "items": 10,
            "late_items": 2,
            "late_probability": 0.2,
            "late_rate": 0.2,
            "item_share": 0.5,
            "alerts": False,
            "partition": "validation",
        },
    ]


def test_profile_table_calculates_calibration_and_confusion_counts() -> None:
    table = build_profile_table(synthetic_profiles(), threshold=0.5)

    first = table.iloc[0]
    assert first["observed_late_rate"] == pytest.approx(0.6)
    assert first["calibration_gap"] == pytest.approx(-0.2)
    assert first["profile_brier_score"] == pytest.approx(0.28)
    assert first["true_alerts"] == 6
    assert first["false_alerts"] == 4
    assert first["missed_late_items"] == 0

    standard = table.iloc[1]
    assert standard["calibration_gap"] == pytest.approx(0.0)
    assert standard["profile_brier_score"] == pytest.approx(0.16)
    assert standard["true_alerts"] == 0
    assert standard["false_alerts"] == 0
    assert standard["missed_late_items"] == 2
    assert standard["correct_non_alerts"] == 8


def test_calibration_metrics_match_hand_calculation() -> None:
    metrics = calculate_calibration_metrics(
        build_profile_table(synthetic_profiles(), threshold=0.5)
    )

    assert metrics["items"] == 20
    assert metrics["late_items"] == 8
    assert metrics["observed_late_prevalence"] == pytest.approx(0.4)
    assert metrics["mean_predicted_probability"] == pytest.approx(0.5)
    assert metrics["mean_probability_minus_prevalence"] == pytest.approx(0.1)
    assert metrics["brier_score"] == pytest.approx(0.22)
    assert metrics["prevalence_baseline_brier_score"] == pytest.approx(0.24)
    assert metrics["brier_skill_score"] == pytest.approx(1 - 0.22 / 0.24)
    assert metrics["expected_calibration_error"] == pytest.approx(0.1)
    assert metrics["maximum_calibration_error"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("items", 0, "items"),
        ("late_items", 11, "late_items"),
        ("late_probability", 1.1, "probability"),
        ("late_rate", 0.9, "late_rate"),
        ("item_share", 0.4, "item_share"),
        ("alerts", False, "alerts"),
    ],
)
def test_profile_table_rejects_inconsistent_tracked_values(
    field: str, value: object, match: str
) -> None:
    profiles = synthetic_profiles()
    profiles[0][field] = value

    with pytest.raises(ValueError, match=match):
        build_profile_table(profiles, threshold=0.5)


def write_runner_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    configuration = {
        "configuration_id": "frozen-tree",
        "alert_threshold": 0.5,
        "features": ["days_for_shipment_scheduled", "shipping_mode"],
    }
    configuration_path = tmp_path / "config.json"
    configuration_text = json.dumps(configuration, indent=2) + "\n"
    configuration_path.write_text(configuration_text, encoding="utf-8")
    configuration_hash = hashlib.sha256(configuration_text.encode()).hexdigest()

    validation_result = {
        "model": "frozen_depth_3_tree",
        "partition": "validation",
        "unit": "order_item",
        "items": 20,
        "late_items": 8,
        "late_prevalence": 0.4,
        "threshold": 0.5,
        "false_alerts": 4,
        "missed_late_items": 2,
        "total_alerts": 10,
        "true_alerts": 6,
        "correct_non_alerts": 8,
        "alert_rate": 0.5,
    }
    final_report = {
        "configuration_id": "frozen-tree",
        "configuration_sha256": configuration_hash,
        "configuration_recorded_before_test_scoring": True,
        "training_fitted_model_fingerprint": "model-hash",
        "model_and_configuration_unchanged_after_scoring": True,
        "validation_reference_reproduced_before_test": True,
        "alert_threshold": 0.5,
        "model_features": ["days_for_shipment_scheduled", "shipping_mode"],
        "results": [validation_result],
        "profile_diagnostics": synthetic_profiles(),
        "post_test_model_feature_threshold_changes": "none",
    }
    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(final_report), encoding="utf-8")

    order_report = {
        "configuration_id": "frozen-tree",
        "configuration_sha256": configuration_hash,
        "training_fitted_model_fingerprint": "model-hash",
        "alert_threshold": 0.5,
        "tracked_item_level_results_reproduced": True,
        "post_hoc_supplemental": True,
        "permitted_use": "reporting only; no tuning",
        "results": [
            {
                "model": "frozen_depth_3_tree",
                "partition": "validation",
                "unit": "unique_order",
                "orders": 10,
                "late_orders": 4,
                "false_alerts": 2,
                "missed_late_orders": 1,
                "total_alerts": 5,
                "true_alerts": 3,
                "correct_non_alerts": 4,
            },
            {
                "model": "frozen_depth_3_tree",
                "partition": "test",
                "unit": "unique_order",
                "orders": 10,
                "late_orders": 5,
                "false_alerts": 1,
                "missed_late_orders": 2,
                "total_alerts": 4,
                "true_alerts": 3,
                "correct_non_alerts": 4,
            },
        ],
    }
    order_path = tmp_path / "orders.json"
    order_path.write_text(json.dumps(order_report), encoding="utf-8")
    return configuration_path, final_path, order_path


def test_runner_writes_reproducible_reports_and_figures(tmp_path: Path) -> None:
    configuration_path, final_path, order_path = write_runner_inputs(tmp_path)
    report_path = tmp_path / "calibration.json"
    table_path = tmp_path / "calibration.csv"
    reliability_path = tmp_path / "reliability.png"
    errors_path = tmp_path / "errors.png"

    report = run_calibration_analysis(
        final_report_path=final_path,
        order_report_path=order_path,
        configuration_path=configuration_path,
        report_path=report_path,
        table_path=table_path,
        reliability_figure_path=reliability_path,
        error_figure_path=errors_path,
    )

    assert report["analysis_partition"] == "validation"
    assert report["calibration_fit_performed"] is False
    assert report["frozen_model_or_threshold_changed"] is False
    assert report["calibration_metrics"]["brier_score"] == pytest.approx(0.22)
    assert report_path.is_file()
    assert reliability_path.stat().st_size > 0
    assert errors_path.stat().st_size > 0
    table = pd.read_csv(table_path)
    assert table["shipping_mode"].tolist() == ["First Class", "Standard Class"]


def test_runner_rejects_mismatched_order_contract(tmp_path: Path) -> None:
    configuration_path, final_path, order_path = write_runner_inputs(tmp_path)
    order_report = json.loads(order_path.read_text())
    order_report["configuration_sha256"] = "changed"
    order_path.write_text(json.dumps(order_report), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration"):
        run_calibration_analysis(
            final_report_path=final_path,
            order_report_path=order_path,
            configuration_path=configuration_path,
            report_path=tmp_path / "out.json",
            table_path=tmp_path / "out.csv",
            reliability_figure_path=tmp_path / "reliability.png",
            error_figure_path=tmp_path / "errors.png",
        )


def test_tracked_calibration_artifacts_are_internally_consistent() -> None:
    report_path = REPOSITORY_ROOT / "reports" / "calibration_analysis.json"
    table_path = REPOSITORY_ROOT / "reports" / "calibration_analysis.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)

    assert report["analysis_partition"] == "validation"
    assert report["runtime"] == {
        "python": "3.11.18",
        "pandas": "2.2.3",
        "numpy": "2.1.3",
        "matplotlib": "3.9.4",
    }
    assert report["calibration_fit_performed"] is False
    assert report["threshold_selection_performed"] is False
    assert report["frozen_model_or_threshold_changed"] is False
    assert table["items"].sum() == report["calibration_metrics"]["items"]
    assert table["late_items"].sum() == report["calibration_metrics"]["late_items"]
    assert table["shipping_mode"].tolist() == [
        row["shipping_mode"] for row in report["profile_results"]
    ]
    for figure in report["figures"].values():
        assert (REPOSITORY_ROOT / figure).stat().st_size > 0
