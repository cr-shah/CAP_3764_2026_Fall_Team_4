"""Tests for reusable DataCo schema validation."""

import pandas as pd
import pytest

from src.data_validation import (
    DataValidationError,
    validate_order_ids,
    validate_schema,
    validate_target,
)


def valid_normalized_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "late_delivery_risk": [0, 1],
            "order_id": [100, 100],
            "delivery_status": ["Shipping on time", "Late delivery"],
            "order_status": ["COMPLETE", "COMPLETE"],
            "order_date_dateorders": ["1/1/2018 00:00", "1/1/2018 00:00"],
            "shipping_date_dateorders": ["1/3/2018 00:00", "1/4/2018 00:00"],
            "days_for_shipping_real": [2, 3],
            "days_for_shipment_scheduled": [2, 2],
            "shipping_mode": ["Second Class", "Second Class"],
        }
    )


def test_valid_schema_accepts_repeated_order_ids() -> None:
    metadata = validate_schema(valid_normalized_data())

    assert metadata["unique_order_ids"] == 1
    assert metadata["repeated_order_id_rows"] == 1


def test_schema_validation_reports_missing_columns() -> None:
    data = valid_normalized_data().drop(columns=["shipping_mode"])

    with pytest.raises(DataValidationError, match="shipping_mode"):
        validate_schema(data)


def test_target_rejects_nonbinary_values() -> None:
    data = valid_normalized_data()
    data.loc[1, "late_delivery_risk"] = 2

    with pytest.raises(DataValidationError, match="outside"):
        validate_target(data)


def test_target_rejects_missing_values() -> None:
    data = valid_normalized_data()
    data.loc[1, "late_delivery_risk"] = None

    with pytest.raises(DataValidationError, match="missing"):
        validate_target(data)


def test_order_id_is_required() -> None:
    data = valid_normalized_data().drop(columns=["order_id"])

    with pytest.raises(DataValidationError, match="order_id"):
        validate_order_ids(data)
