"""Chronological order partitions and a validation-only always-late reference.

Run from the repository root with ``python -m src.baseline_evaluation``.
The final test partition is audited structurally, never scored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data_loader import REPOSITORY_ROOT, load_raw_data, raw_file_metadata
from .data_validation import TARGET_COLUMN, validate_order_ids, validate_target
from .leakage_guardrails import assert_no_group_overlap, validate_model_features
from .preprocessing import clean_supply_chain_data


ORDER_ID = "order_id"
ORDER_TIMESTAMP = "order_date_dateorders"
MODEL_FEATURES = ("days_for_shipment_scheduled", "shipping_mode")
PARTITION_NAMES = ("train", "validation", "test")
EXPECTED_RAW_SHA256 = "994b3c8d24049cc46bf20e8161fedece9a9b95376cc0dc929921d76cf6b9bc0d"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "reports" / "always_late_validation.json"
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "data" / "processed" / "baseline_split.csv"


def select_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Use an explicit allowlist; never infer features by dropping the target."""
    validate_model_features(MODEL_FEATURES)
    return data.loc[:, list(MODEL_FEATURES)].copy()


def audit_partitions(partitions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Require disjoint orders and strict time ordering without reading labels."""
    metadata: dict[str, Any] = {}
    for name in PARTITION_NAMES:
        part = partitions[name]
        if part.empty:
            raise ValueError(f"The {name} partition is empty; check timestamp ties.")
        metadata[name] = {
            "rows": len(part),
            "orders": int(part[ORDER_ID].nunique()),
            "first_order_timestamp": part[ORDER_TIMESTAMP].min().isoformat(),
            "last_order_timestamp": part[ORDER_TIMESTAMP].max().isoformat(),
        }

    overlaps = {}
    for left, right in combinations(PARTITION_NAMES, 2):
        assert_no_group_overlap(partitions[left][ORDER_ID], partitions[right][ORDER_ID])
        overlaps[f"{left}_{right}"] = 0
        if not (
            partitions[left][ORDER_TIMESTAMP].max()
            < partitions[right][ORDER_TIMESTAMP].min()
        ):
            raise ValueError(f"Partitions {left} and {right} are not strictly chronological.")
    return {"partitions": metadata, "overlapping_order_counts": overlaps}


def chronological_partitions(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split at order-rank 60% and 80% timestamps, moving ties to later sets.

    Only order ID and timestamp determine membership. All rows are retained.
    Invalid/missing dates or conflicting timestamps within an order fail loudly.
    No shuffle, label stratification, or random seed is involved.
    """
    validate_order_ids(data)
    dates = data[ORDER_TIMESTAMP]
    if not is_datetime64_any_dtype(dates) or dates.isna().any():
        raise ValueError("Order timestamps must be parsed datetimes without missing values.")
    if data.groupby(ORDER_ID)[ORDER_TIMESTAMP].nunique().gt(1).any():
        raise ValueError("Conflicting order timestamps within an order.")

    orders = data[[ORDER_ID, ORDER_TIMESTAMP]].drop_duplicates(ORDER_ID)
    orders = orders.sort_values([ORDER_TIMESTAMP, ORDER_ID])
    if len(orders) < 5:
        raise ValueError("At least five unique orders are required for a 60/20/20 split.")
    validation_start = orders.iloc[len(orders) * 3 // 5][ORDER_TIMESTAMP]
    test_start = orders.iloc[len(orders) * 4 // 5][ORDER_TIMESTAMP]
    partitions = {
        "train": data.loc[dates < validation_start].copy(),
        "validation": data.loc[(dates >= validation_start) & (dates < test_start)].copy(),
        "test": data.loc[dates >= test_start].copy(),
    }
    audit_partitions(partitions)
    if sum(len(part) for part in partitions.values()) != len(data):
        raise ValueError("Partitioning did not retain every input row exactly once.")
    return partitions


def split_manifest(partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one structural record per order, with no labels or predictors."""
    return pd.concat(
        [
            partitions[name][[ORDER_ID, ORDER_TIMESTAMP]]
            .drop_duplicates(ORDER_ID)
            .assign(partition=name)
            for name in PARTITION_NAMES
        ],
        ignore_index=True,
    ).sort_values([ORDER_TIMESTAMP, ORDER_ID], ignore_index=True)


def evaluate_always_late(validation_target: pd.Series) -> dict[str, Any]:
    """Score the fixed rule on validation labels; there is no fitting or tuning."""
    validate_target(pd.DataFrame({TARGET_COLUMN: validation_target}))
    if set(validation_target.unique()) != {0, 1}:
        raise ValueError("Validation must contain both classes for this comparison.")
    scores = np.ones(len(validation_target), dtype=float)
    predictions = np.ones(len(validation_target), dtype=int)
    tn, fp, fn, tp = confusion_matrix(validation_target, predictions, labels=[0, 1]).ravel()
    return {
        "model": "always_predict_late",
        "partition": "validation",
        "unit": "order_item",
        "primary_metric": "average_precision",
        "class_counts": {
            "not_late_0": int(validation_target.eq(0).sum()),
            "late_1": int(validation_target.eq(1).sum()),
        },
        "late_prevalence": float(validation_target.mean()),
        "average_precision": float(average_precision_score(validation_target, scores)),
        "precision": float(precision_score(validation_target, predictions)),
        "recall": float(recall_score(validation_target, predictions)),
        "f1": float(f1_score(validation_target, predictions)),
        "accuracy": float(accuracy_score(validation_target, predictions)),
        "alert_rate": float(predictions.mean()),
        "confusion_counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "constant_score": 1.0,
        "decision_rule": "alert on every item; no threshold selected or tuned",
    }


def run_baseline(
    raw_path: str | Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Reuse Submission 1 cleaning, audit the split, and score validation only."""
    source = raw_file_metadata(raw_path)
    if source["sha256"] != EXPECTED_RAW_SHA256:
        raise ValueError("Raw source checksum differs from the agreed DataCo source.")
    data, _ = clean_supply_chain_data(load_raw_data(raw_path))
    partitions = chronological_partitions(data)
    # Guard the prospective modeling contract without using test predictors/labels.
    select_model_features(partitions["train"])
    select_model_features(partitions["validation"])
    manifest = split_manifest(partitions)
    manifest_text = manifest.to_csv(
        index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n"
    )
    report = {
        "source": source,
        "prediction_time": "order_placement",
        "target": TARGET_COLUMN,
        "positive_class": 1,
        "order_id": ORDER_ID,
        "order_timestamp": ORDER_TIMESTAMP,
        "prospective_model_features": list(MODEL_FEATURES),
        "excluded_cleaned_columns": sorted(set(data.columns) - set(MODEL_FEATURES)),
        "reference_uses_features": False,
        "split": {
            "rule": "chronological whole orders; 60/20/20 by order count; boundary ties go later",
            "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
            **audit_partitions(partitions),
        },
        "validation_reference": evaluate_always_late(partitions["validation"][TARGET_COLUMN]),
        "final_test_evaluated": False,
        "learned_preprocessing": "none; fixed rule requires no fitting",
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    manifest_path = Path(manifest_path)
    report_path = Path(report_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    report = run_baseline(args.raw_path, args.report_path, args.manifest_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
