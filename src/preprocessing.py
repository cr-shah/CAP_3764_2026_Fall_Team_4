"""Deterministic, non-learned preprocessing for the DataCo dataset."""

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .data_loader import REPOSITORY_ROOT, load_raw_data
from .data_validation import DATE_COLUMNS, validate_cleaned_data, validate_schema


DEFAULT_PROCESSED_DATA_PATH = (
    REPOSITORY_ROOT / "data" / "processed" / "dataco_clean.csv"
)

DIRECT_PII_REASONS = {
    "customer_email": "Direct contact information is unnecessary for delivery-risk analysis.",
    "customer_password": "Credential-like data is unnecessary and must not be modeled.",
    "customer_fname": "A customer's given name is direct PII and has no justified analytical role.",
    "customer_lname": "A customer's surname is direct PII and has no justified analytical role.",
    "customer_street": "A street address is direct location PII and is unnecessarily granular.",
}


def normalize_column_name(column: object) -> str:
    """Convert one source column name to deterministic snake_case."""
    normalized = str(column).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError(f"Column name {column!r} normalizes to an empty name.")
    return normalized


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized names, rejecting normalization collisions."""
    normalized = [normalize_column_name(column) for column in data.columns]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        collision_details = {
            name: [str(source) for source, result in zip(data.columns, normalized) if result == name]
            for name in duplicates
        }
        raise ValueError(
            "Column-name normalization would create duplicate names: "
            f"{collision_details}"
        )

    cleaned = data.copy(deep=True)
    cleaned.columns = normalized
    return cleaned


def normalize_string_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from string values without statistical filling."""
    cleaned = data.copy(deep=True)

    def strip_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped if stripped else pd.NA

    for column in cleaned.columns:
        if is_object_dtype(cleaned[column]) or is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].map(strip_value)
    return cleaned


def parse_dates(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Convert both source dates, recording success, failure, and missing counts."""
    cleaned = data.copy(deep=True)
    metadata: dict[str, dict[str, int]] = {}

    for column in DATE_COLUMNS:
        if column not in cleaned.columns:
            raise ValueError(f"Cannot parse missing date column: {column}")
        source = cleaned[column]
        parsed = pd.to_datetime(source, errors="coerce")
        missing_input = int(source.isna().sum())
        failed = int((source.notna() & parsed.isna()).sum())
        successful = int((source.notna() & parsed.notna()).sum())
        cleaned[column] = parsed
        metadata[column] = {
            "successful_conversions": successful,
            "failed_conversions": failed,
            "missing_input_values": missing_input,
        }

    return cleaned, metadata


def summarize_missingness(data: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages, highest percentage first."""
    row_count = len(data)
    counts = data.isna().sum()
    percentages = counts.div(row_count).mul(100) if row_count else counts.astype(float)
    summary = pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": [int(counts[column]) for column in data.columns],
            "missing_percentage": [float(percentages[column]) for column in data.columns],
        }
    )
    return summary.sort_values(
        ["missing_percentage", "column"], ascending=[False, True], ignore_index=True
    )


def remove_exact_duplicates(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows while preserving repeated order IDs."""
    duplicate_count = int(data.duplicated().sum())
    return data.drop_duplicates().reset_index(drop=True), duplicate_count


def remove_unusable_columns(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remove only explicitly supported deterministic unusable fields."""
    cleaned = data.copy(deep=True)
    removed: dict[str, str] = {}
    column = "product_description"
    if column in cleaned.columns and cleaned[column].isna().all():
        cleaned = cleaned.drop(columns=[column])
        removed[column] = (
            "The field is 100% missing in this source file and contains no information."
        )
    return cleaned, removed


def remove_direct_pii(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remove documented direct PII without removing useful geographic categories."""
    cleaned = data.copy(deep=True)
    present = [column for column in DIRECT_PII_REASONS if column in cleaned.columns]
    removed = {column: DIRECT_PII_REASONS[column] for column in present}
    return cleaned.drop(columns=present), removed


def summarize_numeric_ranges(data: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Report numeric minima/maxima without deleting unusual observations."""
    summary: dict[str, dict[str, float | None]] = {}
    for column in data.select_dtypes(include="number").columns:
        series = data[column].dropna()
        summary[column] = {
            "minimum": None if series.empty else float(series.min()),
            "maximum": None if series.empty else float(series.max()),
        }
    return summary


def clean_supply_chain_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic cleaning without fitting any learned preprocessing.

    The caller's dataframe is never mutated. Legitimate missing values and
    repeated order IDs are retained for future train-only pipeline decisions.
    """
    original_shape = (int(data.shape[0]), int(data.shape[1]))
    original_columns = [str(column) for column in data.columns]

    cleaned = standardize_column_names(data)
    normalized_columns = list(cleaned.columns)
    column_mapping = dict(zip(original_columns, normalized_columns))
    validate_schema(cleaned)

    cleaned = normalize_string_whitespace(cleaned)
    missing_before = summarize_missingness(cleaned)
    cleaned, date_metadata = parse_dates(cleaned)
    cleaned, exact_duplicates_removed = remove_exact_duplicates(cleaned)
    cleaned, unusable_removed = remove_unusable_columns(cleaned)
    cleaned, pii_removed = remove_direct_pii(cleaned)

    final_validation = validate_cleaned_data(cleaned)
    missing_after = summarize_missingness(cleaned)
    removed_columns = {**unusable_removed, **pii_removed}

    metadata: dict[str, Any] = {
        "original_shape": original_shape,
        "cleaned_shape": (int(cleaned.shape[0]), int(cleaned.shape[1])),
        "column_mapping": column_mapping,
        "date_conversion": date_metadata,
        "exact_duplicates_removed": exact_duplicates_removed,
        "unique_order_ids": final_validation["unique_order_ids"],
        "repeated_order_id_rows": final_validation["repeated_order_id_rows"],
        "removed_columns": removed_columns,
        "missingness_before_removal": missing_before.to_dict(orient="records"),
        "missingness_after_cleaning": missing_after.to_dict(orient="records"),
        "numeric_ranges": summarize_numeric_ranges(cleaned),
        "outlier_policy": (
            "Outliers are retained during deterministic preprocessing unless they "
            "are proven to be invalid data. Statistical outlier treatment, if needed, "
            "must be justified later."
        ),
    }
    return cleaned, metadata


def save_processed_data(
    data: pd.DataFrame,
    path: str | PathLike[str] | None = None,
) -> Path:
    """Save cleaned data with stable date formatting and return the output path."""
    destination = DEFAULT_PROCESSED_DATA_PATH if path is None else Path(path)
    if not destination.is_absolute():
        destination = REPOSITORY_ROOT / destination
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        destination,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    return destination


def run_preprocessing(
    raw_path: str | PathLike[str] | None = None,
    output_path: str | PathLike[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Load, deterministically clean, and locally save the DataCo dataset."""
    raw = load_raw_data(raw_path)
    cleaned, metadata = clean_supply_chain_data(raw)
    destination = save_processed_data(cleaned, output_path)
    return cleaned, metadata, destination


if __name__ == "__main__":
    cleaned_data, processing_metadata, output = run_preprocessing()
    print(f"Processed data written to: {output}")
    print(f"Raw shape: {processing_metadata['original_shape']}")
    print(f"Cleaned shape: {processing_metadata['cleaned_shape']}")
    print(
        "Exact duplicates removed: "
        f"{processing_metadata['exact_duplicates_removed']}"
    )
    print(f"Unique order IDs: {processing_metadata['unique_order_ids']}")
