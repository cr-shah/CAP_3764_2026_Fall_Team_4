"""Schema and value validation for normalized DataCo data."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


TARGET_COLUMN = "late_delivery_risk"
ORDER_ID_COLUMN = "order_id"
DATE_COLUMNS = ("order_date_dateorders", "shipping_date_dateorders")
SHIPPING_COLUMNS = (
    "days_for_shipping_real",
    "days_for_shipment_scheduled",
    "shipping_mode",
)
REQUIRED_COLUMNS = frozenset(
    {
        TARGET_COLUMN,
        ORDER_ID_COLUMN,
        "delivery_status",
        "order_status",
        *DATE_COLUMNS,
        *SHIPPING_COLUMNS,
    }
)
REQUIRED_NUMERIC_COLUMNS = (
    TARGET_COLUMN,
    "days_for_shipping_real",
    "days_for_shipment_scheduled",
)


class DataValidationError(ValueError):
    """Raised when DataCo data violates an expected structural invariant."""


def validate_not_empty(data: pd.DataFrame) -> None:
    """Require at least one row and one column."""
    if data.empty or len(data.columns) == 0:
        raise DataValidationError("The DataCo dataframe is empty.")


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str] | frozenset[str] = REQUIRED_COLUMNS,
) -> None:
    """Require all important normalized columns."""
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise DataValidationError(f"Missing required DataCo columns: {missing}")


def validate_target(data: pd.DataFrame) -> None:
    """Require a non-missing binary late-delivery target containing only 0/1."""
    if TARGET_COLUMN not in data.columns:
        raise DataValidationError(f"Missing target column: {TARGET_COLUMN}")

    target = data[TARGET_COLUMN]
    if target.isna().any():
        raise DataValidationError("late_delivery_risk contains missing values.")
    if not is_numeric_dtype(target):
        raise DataValidationError("late_delivery_risk must have a numeric binary dtype.")

    unexpected = sorted(set(target.unique()) - {0, 1})
    if unexpected:
        raise DataValidationError(
            f"late_delivery_risk contains values outside {{0, 1}}: {unexpected}"
        )


def validate_order_ids(data: pd.DataFrame) -> dict[str, int]:
    """Validate order IDs while explicitly allowing legitimate repeated IDs."""
    if ORDER_ID_COLUMN not in data.columns:
        raise DataValidationError(f"Missing structural identifier: {ORDER_ID_COLUMN}")
    if data[ORDER_ID_COLUMN].isna().any():
        raise DataValidationError("order_id contains missing values.")

    return {
        "unique_order_ids": int(data[ORDER_ID_COLUMN].nunique()),
        "repeated_order_id_rows": int(data[ORDER_ID_COLUMN].duplicated().sum()),
    }


def validate_numeric_columns(data: pd.DataFrame) -> None:
    """Require basic numeric fields and non-negative shipping-day values."""
    for column in REQUIRED_NUMERIC_COLUMNS:
        if column not in data.columns:
            raise DataValidationError(f"Missing required numeric column: {column}")
        if not is_numeric_dtype(data[column]):
            raise DataValidationError(f"{column} must have a numeric dtype.")

    for column in ("days_for_shipping_real", "days_for_shipment_scheduled"):
        non_missing = data[column].dropna()
        if (non_missing < 0).any():
            raise DataValidationError(f"{column} contains negative shipping-day values.")


def validate_parsed_dates(data: pd.DataFrame) -> None:
    """Require both date columns to have pandas datetime dtypes."""
    for column in DATE_COLUMNS:
        if column not in data.columns:
            raise DataValidationError(f"Missing required date column: {column}")
        if not is_datetime64_any_dtype(data[column]):
            raise DataValidationError(f"{column} has not been converted to datetime.")


def validate_schema(data: pd.DataFrame) -> dict[str, Any]:
    """Validate the normalized source schema and return safe structural facts."""
    validate_not_empty(data)
    validate_required_columns(data)
    validate_target(data)
    validate_numeric_columns(data)
    order_metadata = validate_order_ids(data)
    return {
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        **order_metadata,
    }


def validate_cleaned_data(data: pd.DataFrame) -> dict[str, Any]:
    """Validate final deterministic-cleaning invariants."""
    metadata = validate_schema(data)
    validate_parsed_dates(data)
    return metadata
