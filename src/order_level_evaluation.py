"""Report post-hoc metrics with each unique order weighted once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from .baseline_evaluation import (
    EXPECTED_RAW_SHA256,
    MODEL_FEATURES,
    ORDER_ID,
    ORDER_TIMESTAMP,
    TARGET_COLUMN,
    chronological_partitions,
    select_model_features,
)
from .data_loader import REPOSITORY_ROOT, load_raw_data, raw_file_metadata
from .final_evaluation import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_PATH as TRACKED_FINAL_REPORT_PATH,
    load_frozen_configuration,
    score_partition,
    text_sha256,
    verify_validation_results,
)
from .preprocessing import clean_supply_chain_data
from .tree_evaluation import fit_tree, verify_reference_contract


ORDER_INVARIANTS = (TARGET_COLUMN, *MODEL_FEATURES, ORDER_TIMESTAMP, "score")
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "reports" / "order_level_evaluation.json"
DEFAULT_TABLE_PATH = REPOSITORY_ROOT / "reports" / "order_level_evaluation.csv"


def report_path_label(path: str | Path) -> str:
    """Prefer a repository-relative artifact label, with an external-path fallback."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def aggregate_orders(data: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Collapse item rows after proving all evaluated fields agree per order."""
    required = {ORDER_ID, TARGET_COLUMN, ORDER_TIMESTAMP, *MODEL_FEATURES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Order aggregation is missing required columns: {missing}")
    if data[ORDER_ID].isna().any():
        raise ValueError("order_id must be nonmissing for unique-order evaluation.")

    values = np.asarray(scores, dtype=float)
    if values.shape != (len(data),):
        raise ValueError("Expected one model score per order item.")
    working = data.loc[:, [ORDER_ID, TARGET_COLUMN, *MODEL_FEATURES, ORDER_TIMESTAMP]].copy()
    working["score"] = values

    variation = working.groupby(ORDER_ID, sort=False, dropna=False)[list(ORDER_INVARIANTS)].nunique(
        dropna=False
    )
    inconsistent = [column for column in ORDER_INVARIANTS if (variation[column] != 1).any()]
    if inconsistent:
        raise ValueError(f"Each order must have one consistent value for: {inconsistent}")

    return working.drop_duplicates(ORDER_ID, keep="first").reset_index(drop=True)


def score_orders(
    orders: pd.DataFrame, partition: str, threshold: float
) -> dict[str, Any]:
    """Compute the frozen decision metrics with every unique order weighted once."""
    required = {ORDER_ID, TARGET_COLUMN, "score"}
    missing = sorted(required - set(orders.columns))
    if missing:
        raise ValueError(f"Order scoring is missing required columns: {missing}")
    if orders[ORDER_ID].isna().any() or orders[ORDER_ID].duplicated().any():
        raise ValueError("Order scoring requires one nonmissing row per order_id.")

    target = orders[TARGET_COLUMN].to_numpy()
    scores = orders["score"].to_numpy(dtype=float)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Each evaluated order partition must contain both target classes.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Order scores must be finite probabilities in [0, 1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The alert threshold must be a finite value in [0, 1].")

    alerts = scores >= threshold
    late = target == 1
    tp = int((alerts & late).sum())
    fp = int((alerts & ~late).sum())
    fn = int((~alerts & late).sum())
    tn = int((~alerts & ~late).sum())
    return {
        "model": "frozen_depth_3_tree",
        "partition": partition,
        "unit": "unique_order",
        "post_hoc_supplemental": True,
        "orders": len(orders),
        "late_orders": int(late.sum()),
        "late_prevalence": float(late.mean()),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(target, scores)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn),
        "false_alerts": fp,
        "missed_late_orders": fn,
        "total_alerts": tp + fp,
        "true_alerts": tp,
        "correct_non_alerts": tn,
        "alert_rate": float(alerts.mean()),
    }


def verify_tracked_item_results(
    config: dict[str, Any],
    config_hash: str,
    model_fingerprint: str,
    reproduced_rows: list[dict[str, Any]],
    tracked_report: dict[str, Any],
) -> None:
    """Require exact reproduction of the frozen item-level evaluation contract."""
    if (
        tracked_report.get("configuration_id") != config.get("configuration_id")
        or tracked_report.get("configuration_sha256") != config_hash
    ):
        raise ValueError("Frozen configuration does not match the tracked final report.")
    if tracked_report.get("training_fitted_model_fingerprint") != model_fingerprint:
        raise ValueError("Fitted model fingerprint differs from the tracked final report.")
    if not tracked_report.get("validation_reference_reproduced_before_test"):
        raise ValueError("Tracked validation reproduction guarantee is absent.")
    if not tracked_report.get("model_and_configuration_unchanged_after_scoring"):
        raise ValueError("Tracked final report does not guarantee an unchanged model.")

    expected_partitions = {"validation", "test"}
    reproduced_partitions = {row.get("partition") for row in reproduced_rows}
    if reproduced_partitions != expected_partitions or len(reproduced_rows) != 2:
        raise ValueError("Reproduced item-level partition set must be validation and test.")

    exact_fields = (
        "model",
        "partition",
        "unit",
        "items",
        "late_items",
        "false_alerts",
        "missed_late_items",
        "total_alerts",
        "true_alerts",
        "correct_non_alerts",
    )
    metric_fields = (
        "late_prevalence",
        "threshold",
        "average_precision",
        "precision",
        "recall",
        "alert_rate",
    )
    tracked_frozen_rows = [
        row
        for row in tracked_report.get("results", [])
        if row.get("model") == "frozen_depth_3_tree"
    ]
    tracked_partitions = [row.get("partition") for row in tracked_frozen_rows]
    if len(tracked_partitions) != 2 or set(tracked_partitions) != expected_partitions:
        raise ValueError("Tracked item-level partition set must be unique validation and test rows.")
    tracked_rows = {
        (row["model"], row["partition"]): row for row in tracked_frozen_rows
    }
    for current in reproduced_rows:
        key = (current.get("model"), current.get("partition"))
        previous = tracked_rows.get(key)
        partition = current.get("partition", "unknown")
        if previous is None:
            raise ValueError(f"Tracked {partition} item-level result is missing.")
        if set(current) != set(previous):
            raise ValueError(f"Reproduced {partition} result fields differ from the tracked result.")
        for field in exact_fields:
            if current.get(field) != previous.get(field):
                raise ValueError(f"Reproduced {partition} {field} differs from the tracked result.")
        for field in metric_fields:
            if not np.isclose(current.get(field), previous.get(field), rtol=0, atol=1e-12):
                raise ValueError(f"Reproduced {partition} {field} differs from the tracked result.")


def _positive_class_scores(model: Any, data: pd.DataFrame) -> np.ndarray:
    features = select_model_features(data).replace({pd.NA: np.nan})
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(features)[:, positive_index]


def run_order_level_evaluation(
    raw_path: str | Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    table_path: Path = DEFAULT_TABLE_PATH,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
    tracked_final_path: Path = TRACKED_FINAL_REPORT_PATH,
) -> dict[str, Any]:
    """Reproduce frozen item metrics, then report unique-order metrics separately."""
    config, config_hash = load_frozen_configuration(Path(configuration_path))
    tracked_path = Path(tracked_final_path).resolve()
    tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
    baseline = json.loads(
        (REPOSITORY_ROOT / config["baseline_report_path"]).read_text(encoding="utf-8")
    )
    tree_reference = json.loads(
        (REPOSITORY_ROOT / config["tree_report_path"]).read_text(encoding="utf-8")
    )
    source = raw_file_metadata(raw_path)
    if source["sha256"] != EXPECTED_RAW_SHA256 or source["sha256"] != config["source_sha256"]:
        raise ValueError("Source checksum differs from the frozen data source.")

    data, _ = clean_supply_chain_data(load_raw_data(raw_path))
    partitions = chronological_partitions(data)
    split = verify_reference_contract(partitions, str(source["sha256"]), baseline)
    if split != config["split"] or split != tree_reference["split"] or split != tracked["split"]:
        raise ValueError("Split differs from the frozen configuration or tracked final report.")

    model = fit_tree(partitions["train"])
    fingerprint = joblib.hash(model)
    threshold = config["alert_threshold"]
    validation_rows, _ = score_partition(
        model, partitions["validation"], "validation", threshold
    )
    verify_validation_results(validation_rows, tree_reference)
    if text_sha256(configuration_path) != config_hash or joblib.hash(model) != fingerprint:
        raise ValueError("Model or configuration changed before test reproduction.")

    test_rows, _ = score_partition(model, partitions["test"], "test", threshold)
    reproduced_rows = [validation_rows[0], test_rows[0]]
    if text_sha256(configuration_path) != config_hash or joblib.hash(model) != fingerprint:
        raise ValueError("Model or configuration changed during item-level reproduction.")
    verify_tracked_item_results(config, config_hash, fingerprint, reproduced_rows, tracked)

    results = []
    for partition in ("validation", "test"):
        item_data = partitions[partition]
        scores = _positive_class_scores(model, item_data)
        results.append(score_orders(aggregate_orders(item_data, scores), partition, threshold))

    report = {
        "configuration_id": config["configuration_id"],
        "configuration_sha256": config_hash,
        "source": source,
        "split": split,
        "target": TARGET_COLUMN,
        "model_features": list(MODEL_FEATURES),
        "alert_threshold": threshold,
        "training_fitted_model_fingerprint": fingerprint,
        "tracked_item_level_report": report_path_label(tracked_path),
        "tracked_item_level_results_reproduced": True,
        "results": results,
        "post_hoc_supplemental": True,
        "permitted_use": "reporting only; no model, feature, preprocessing, or threshold changes",
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    report_path = Path(report_path)
    table_path = Path(table_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    pd.DataFrame(results).to_csv(
        table_path, index=False, lineterminator="\n", float_format="%.12f"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--configuration-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tracked-final-path", type=Path, default=TRACKED_FINAL_REPORT_PATH)
    arguments = parser.parse_args()
    report = run_order_level_evaluation(
        arguments.raw_path,
        arguments.report_path,
        arguments.table_path,
        arguments.configuration_path,
        arguments.tracked_final_path,
    )
    print(
        json.dumps(
            {
                "tracked_item_level_results_reproduced": report[
                    "tracked_item_level_results_reproduced"
                ],
                "post_hoc_supplemental": True,
                "results": report["results"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
