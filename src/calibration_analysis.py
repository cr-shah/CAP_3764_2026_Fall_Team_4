"""Analyze frozen-model calibration and errors without fitting or retuning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import REPOSITORY_ROOT
from .final_evaluation import DEFAULT_CONFIG_PATH, text_sha256


DEFAULT_FINAL_REPORT_PATH = REPOSITORY_ROOT / "reports" / "final_evaluation.json"
DEFAULT_ORDER_REPORT_PATH = REPOSITORY_ROOT / "reports" / "order_level_evaluation.json"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "reports" / "calibration_analysis.json"
DEFAULT_TABLE_PATH = REPOSITORY_ROOT / "reports" / "calibration_analysis.csv"
DEFAULT_RELIABILITY_FIGURE_PATH = (
    REPOSITORY_ROOT / "reports" / "figures" / "calibration" / "validation_reliability.png"
)
DEFAULT_ERROR_FIGURE_PATH = (
    REPOSITORY_ROOT / "reports" / "figures" / "calibration" / "validation_error_burden.png"
)
MODEL_NAME = "frozen_depth_3_tree"
VALIDATION_PARTITION = "validation"
EXPECTED_FEATURES = ["days_for_shipment_scheduled", "shipping_mode"]


def _path_label(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _wilson_interval(positives: int, total: int, z: float = 1.96) -> tuple[float, float]:
    rate = positives / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def build_profile_table(profiles: list[dict[str, Any]], threshold: float) -> pd.DataFrame:
    """Validate tracked validation profiles and derive calibration/error columns."""
    if not profiles:
        raise ValueError("At least one validation profile is required.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The frozen threshold must be finite and in [0, 1].")

    required = {
        "days_for_shipment_scheduled",
        "shipping_mode",
        "items",
        "late_items",
        "late_probability",
        "late_rate",
        "item_share",
        "alerts",
        "partition",
    }
    normalized: list[dict[str, Any]] = []
    seen_profiles: set[tuple[Any, Any]] = set()
    total_items = 0

    for source in profiles:
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"Profile is missing required fields: {missing}")
        if source["partition"] != VALIDATION_PARTITION:
            raise ValueError("Calibration analysis is restricted to the validation partition.")

        items = int(source["items"])
        late_items = int(source["late_items"])
        if items != source["items"] or items <= 0:
            raise ValueError("Profile items must be a positive integer.")
        if late_items != source["late_items"] or not 0 <= late_items <= items:
            raise ValueError("Profile late_items must be an integer between zero and items.")

        probability = float(source["late_probability"])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Profile late probability must be finite and in [0, 1].")
        observed = late_items / items
        if not np.isclose(float(source["late_rate"]), observed, rtol=0, atol=1e-12):
            raise ValueError("Tracked late_rate differs from late_items / items.")
        expected_alert = probability >= threshold
        if not isinstance(source["alerts"], bool) or source["alerts"] != expected_alert:
            raise ValueError("Tracked alerts differ from the frozen inclusive threshold rule.")

        profile_key = (source["days_for_shipment_scheduled"], source["shipping_mode"])
        if profile_key in seen_profiles:
            raise ValueError(f"Duplicate shipping profile: {profile_key}")
        seen_profiles.add(profile_key)
        total_items += items
        normalized.append(
            {
                "partition": VALIDATION_PARTITION,
                "shipping_mode": str(source["shipping_mode"]),
                "days_for_shipment_scheduled": source["days_for_shipment_scheduled"],
                "items": items,
                "late_items": late_items,
                "predicted_late_probability": probability,
                "observed_late_rate": observed,
                "tracked_item_share": float(source["item_share"]),
                "alerts": expected_alert,
            }
        )

    rows: list[dict[str, Any]] = []
    for row in normalized:
        expected_share = row["items"] / total_items
        if not np.isclose(row["tracked_item_share"], expected_share, rtol=0, atol=1e-12):
            raise ValueError("Tracked item_share differs from profile items / total items.")
        probability = row["predicted_late_probability"]
        observed = row["observed_late_rate"]
        late_items = row["late_items"]
        items = row["items"]
        non_late_items = items - late_items
        alerts = row["alerts"]
        lower, upper = _wilson_interval(late_items, items)
        rows.append(
            {
                "partition": VALIDATION_PARTITION,
                "shipping_mode": row["shipping_mode"],
                "days_for_shipment_scheduled": row["days_for_shipment_scheduled"],
                "items": items,
                "late_items": late_items,
                "item_share": expected_share,
                "predicted_late_probability": probability,
                "observed_late_rate": observed,
                "calibration_gap": observed - probability,
                "absolute_calibration_gap": abs(observed - probability),
                "observed_rate_ci95_lower": lower,
                "observed_rate_ci95_upper": upper,
                "profile_brier_score": (
                    late_items * (1 - probability) ** 2
                    + non_late_items * probability**2
                )
                / items,
                "alerts": alerts,
                "true_alerts": late_items if alerts else 0,
                "false_alerts": non_late_items if alerts else 0,
                "missed_late_items": late_items if not alerts else 0,
                "correct_non_alerts": non_late_items if not alerts else 0,
            }
        )
    return pd.DataFrame(rows)


def calculate_calibration_metrics(profile_table: pd.DataFrame) -> dict[str, Any]:
    """Calculate weighted validation calibration metrics from profile aggregates."""
    if profile_table.empty:
        raise ValueError("Profile table cannot be empty.")
    items = int(profile_table["items"].sum())
    late_items = int(profile_table["late_items"].sum())
    weights = profile_table["items"].to_numpy(dtype=float) / items
    prevalence = late_items / items
    mean_probability = float(
        np.sum(weights * profile_table["predicted_late_probability"].to_numpy(dtype=float))
    )
    brier = float(np.sum(weights * profile_table["profile_brier_score"].to_numpy(dtype=float)))
    baseline_brier = prevalence * (1 - prevalence)
    return {
        "profiles": len(profile_table),
        "items": items,
        "late_items": late_items,
        "observed_late_prevalence": prevalence,
        "mean_predicted_probability": mean_probability,
        "mean_probability_minus_prevalence": mean_probability - prevalence,
        "brier_score": brier,
        "prevalence_baseline_brier_score": baseline_brier,
        "brier_skill_score": 1 - brier / baseline_brier if baseline_brier else None,
        "expected_calibration_error": float(
            np.sum(weights * profile_table["absolute_calibration_gap"].to_numpy(dtype=float))
        ),
        "maximum_calibration_error": float(profile_table["absolute_calibration_gap"].max()),
    }


def _single_result(
    report: dict[str, Any], partition: str, unit: str
) -> dict[str, Any]:
    rows = [
        row
        for row in report.get("results", [])
        if row.get("model") == MODEL_NAME
        and row.get("partition") == partition
        and row.get("unit") == unit
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {partition} {unit} frozen-tree result.")
    return rows[0]


def _verify_contract(
    configuration: dict[str, Any],
    configuration_hash: str,
    final_report: dict[str, Any],
    order_report: dict[str, Any],
    profile_table: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    configuration_id = configuration.get("configuration_id")
    threshold = configuration.get("alert_threshold")
    if configuration.get("features") != EXPECTED_FEATURES or threshold != 0.5:
        raise ValueError("Frozen configuration differs from the approved feature/threshold protocol.")
    for name, report in (("final", final_report), ("order", order_report)):
        if (
            report.get("configuration_id") != configuration_id
            or report.get("configuration_sha256") != configuration_hash
        ):
            raise ValueError(f"{name.title()} report configuration differs from the frozen record.")
        if report.get("alert_threshold") != threshold:
            raise ValueError(f"{name.title()} report threshold differs from the frozen record.")
    if final_report.get("model_features") != EXPECTED_FEATURES:
        raise ValueError("Final report feature list differs from the frozen record.")
    if (
        order_report.get("training_fitted_model_fingerprint")
        != final_report.get("training_fitted_model_fingerprint")
    ):
        raise ValueError("Item- and order-level reports have different model fingerprints.")
    if not final_report.get("configuration_recorded_before_test_scoring"):
        raise ValueError("Final report lacks the pre-test configuration guarantee.")
    if not final_report.get("validation_reference_reproduced_before_test"):
        raise ValueError("Final report lacks the validation reproduction guarantee.")
    if not final_report.get("model_and_configuration_unchanged_after_scoring"):
        raise ValueError("Final report lacks the unchanged-model guarantee.")
    if not order_report.get("tracked_item_level_results_reproduced"):
        raise ValueError("Order report did not reproduce the tracked item-level results.")
    if not order_report.get("post_hoc_supplemental"):
        raise ValueError("Order report is not marked post-hoc supplemental.")

    validation = _single_result(final_report, VALIDATION_PARTITION, "order_item")
    sums = {
        "items": int(profile_table["items"].sum()),
        "late_items": int(profile_table["late_items"].sum()),
        "false_alerts": int(profile_table["false_alerts"].sum()),
        "missed_late_items": int(profile_table["missed_late_items"].sum()),
        "total_alerts": int((profile_table["true_alerts"] + profile_table["false_alerts"]).sum()),
        "true_alerts": int(profile_table["true_alerts"].sum()),
        "correct_non_alerts": int(profile_table["correct_non_alerts"].sum()),
    }
    for field, value in sums.items():
        if validation.get(field) != value:
            raise ValueError(f"Profile-derived validation {field} differs from the final report.")
    if not np.isclose(
        validation.get("alert_rate"), sums["total_alerts"] / sums["items"], rtol=0, atol=1e-12
    ):
        raise ValueError("Profile-derived validation alert_rate differs from the final report.")

    order_context: dict[str, dict[str, Any]] = {}
    for partition in ("validation", "test"):
        row = _single_result(order_report, partition, "unique_order")
        confusion_total = (
            row["true_alerts"]
            + row["false_alerts"]
            + row["missed_late_orders"]
            + row["correct_non_alerts"]
        )
        if row["orders"] != confusion_total:
            raise ValueError(f"{partition.title()} unique-order confusion counts do not sum to orders.")
        order_context[partition] = {
            field: row[field]
            for field in (
                "orders",
                "late_orders",
                "false_alerts",
                "missed_late_orders",
                "total_alerts",
                "true_alerts",
                "correct_non_alerts",
            )
        }
    return validation, order_context


def _save_reliability_figure(profile_table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#667085", label="Perfect calibration")
    x = profile_table["predicted_late_probability"].to_numpy(dtype=float)
    y = profile_table["observed_late_rate"].to_numpy(dtype=float)
    lower = y - profile_table["observed_rate_ci95_lower"].to_numpy(dtype=float)
    upper = profile_table["observed_rate_ci95_upper"].to_numpy(dtype=float) - y
    sizes = 180 + 900 * profile_table["item_share"].to_numpy(dtype=float)
    axis.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#175CD3", capsize=4)
    axis.scatter(x, y, s=sizes, color="#1570EF", edgecolor="white", linewidth=1.5, zorder=3)
    label_offsets = {
        "First Class": (-112, -8),
        "Second Class": (8, 8),
        "Same Day": (8, 8),
        "Standard Class": (8, -2),
    }
    for _, row in profile_table.iterrows():
        axis.annotate(
            f"{row['shipping_mode']}\n(n={int(row['items']):,})",
            (row["predicted_late_probability"], row["observed_late_rate"]),
            xytext=label_offsets.get(row["shipping_mode"], (8, 8)),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="Frozen model predicted late probability",
        ylabel="Observed validation late rate",
        title="Validation reliability by shipping profile",
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight", metadata={"Software": "CAP 3764 Team 4"})
    plt.close(figure)


def _save_error_figure(profile_table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    false_alert_rows = profile_table.loc[profile_table["false_alerts"] > 0]
    missed_rows = profile_table.loc[profile_table["missed_late_items"] > 0]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    panels = (
        (axes[0], false_alert_rows, "false_alerts", "False alerts in alerted profiles", "#F79009"),
        (
            axes[1],
            missed_rows,
            "missed_late_items",
            "Missed late items in non-alerted profiles",
            "#D92D20",
        ),
    )
    for axis, rows, field, title, color in panels:
        bars = axis.barh(rows["shipping_mode"], rows[field], color=color)
        axis.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=10)
        axis.set(title=title, xlabel="Validation order-item count")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.margins(x=0.16)
    figure.suptitle("Where errors occur at the frozen 0.50 threshold", fontsize=15)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight", metadata={"Software": "CAP 3764 Team 4"})
    plt.close(figure)


def run_calibration_analysis(
    final_report_path: Path = DEFAULT_FINAL_REPORT_PATH,
    order_report_path: Path = DEFAULT_ORDER_REPORT_PATH,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    table_path: Path = DEFAULT_TABLE_PATH,
    reliability_figure_path: Path = DEFAULT_RELIABILITY_FIGURE_PATH,
    error_figure_path: Path = DEFAULT_ERROR_FIGURE_PATH,
) -> dict[str, Any]:
    """Verify frozen reports and write validation-only calibration diagnostics."""
    configuration_path = Path(configuration_path)
    final_report_path = Path(final_report_path)
    order_report_path = Path(order_report_path)
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    order_report = json.loads(order_report_path.read_text(encoding="utf-8"))
    configuration_hash = text_sha256(configuration_path)
    profiles = [
        row
        for row in final_report.get("profile_diagnostics", [])
        if row.get("partition") == VALIDATION_PARTITION
    ]
    profile_table = build_profile_table(profiles, float(configuration["alert_threshold"]))
    validation, order_context = _verify_contract(
        configuration, configuration_hash, final_report, order_report, profile_table
    )
    metrics = calculate_calibration_metrics(profile_table)

    reliability_figure_path = Path(reliability_figure_path)
    error_figure_path = Path(error_figure_path)
    _save_reliability_figure(profile_table, reliability_figure_path)
    _save_error_figure(profile_table, error_figure_path)
    report = {
        "configuration_id": configuration["configuration_id"],
        "configuration_sha256": configuration_hash,
        "training_fitted_model_fingerprint": final_report["training_fitted_model_fingerprint"],
        "source_reports": {
            "final_item_level": _path_label(final_report_path),
            "supplemental_unique_order": _path_label(order_report_path),
        },
        "analysis_partition": VALIDATION_PARTITION,
        "unit_for_calibration": "order_item",
        "calibration_method": "four observed frozen-score shipping profiles; item-weighted",
        "confidence_intervals": "95% Wilson intervals for observed profile late rates",
        "alert_threshold": configuration["alert_threshold"],
        "calibration_fit_performed": False,
        "threshold_selection_performed": False,
        "frozen_model_or_threshold_changed": False,
        "calibration_metrics": metrics,
        "validation_item_level_operating_context": {
            field: validation[field]
            for field in (
                "items",
                "late_items",
                "false_alerts",
                "missed_late_items",
                "total_alerts",
                "true_alerts",
                "correct_non_alerts",
                "alert_rate",
            )
        },
        "unique_order_operational_context": order_context,
        "profile_results": profile_table.to_dict(orient="records"),
        "figures": {
            "reliability": _path_label(reliability_figure_path),
            "error_burden": _path_label(error_figure_path),
        },
        "permitted_use": (
            "validation-only diagnostic reporting; no calibration fit, model selection, "
            "feature change, or threshold change"
        ),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    report_path = Path(report_path)
    table_path = Path(table_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    profile_table.to_csv(table_path, index=False, lineterminator="\n", float_format="%.12f")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-report-path", type=Path, default=DEFAULT_FINAL_REPORT_PATH)
    parser.add_argument("--order-report-path", type=Path, default=DEFAULT_ORDER_REPORT_PATH)
    parser.add_argument("--configuration-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument(
        "--reliability-figure-path", type=Path, default=DEFAULT_RELIABILITY_FIGURE_PATH
    )
    parser.add_argument("--error-figure-path", type=Path, default=DEFAULT_ERROR_FIGURE_PATH)
    arguments = parser.parse_args()
    report = run_calibration_analysis(
        final_report_path=arguments.final_report_path,
        order_report_path=arguments.order_report_path,
        configuration_path=arguments.configuration_path,
        report_path=arguments.report_path,
        table_path=arguments.table_path,
        reliability_figure_path=arguments.reliability_figure_path,
        error_figure_path=arguments.error_figure_path,
    )
    print(
        json.dumps(
            {
                "analysis_partition": report["analysis_partition"],
                "calibration_metrics": report["calibration_metrics"],
                "frozen_model_or_threshold_changed": False,
                "figures": report["figures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
