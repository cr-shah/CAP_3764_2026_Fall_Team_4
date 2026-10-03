"""Tests for deterministic DataCo preprocessing."""

import pandas as pd
import pytest

from src.preprocessing import (
    clean_supply_chain_data,
    parse_dates,
    remove_direct_pii,
    remove_exact_duplicates,
    standardize_column_names,
    summarize_missingness,
)


def source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Late_delivery_risk": [0, 1],
            "Order Id": [100, 100],
            "Delivery Status": ["Shipping on time", "Late delivery"],
            "Order Status": ["COMPLETE", "COMPLETE"],
            "order date (DateOrders)": ["1/1/2018 00:00", "1/1/2018 00:00"],
            "shipping date (DateOrders)": ["1/3/2018 00:00", "1/4/2018 00:00"],
            "Days for shipping (real)": [2, 3],
            "Days for shipment (scheduled)": [2, 2],
            "Shipping Mode": ["Second Class", "Second Class"],
            "Product Description": [None, None],
            "Customer Email": ["masked", "masked"],
            "Customer Password": ["masked", "masked"],
            "Customer Fname": ["Person A", "Person A"],
            "Customer Lname": ["Example", "Example"],
            "Customer Street": ["Street A", "Street A"],
            "Customer Country": ["EE. UU.", "EE. UU."],
        }
    )


def test_column_name_normalization_matches_policy_examples() -> None:
    data = pd.DataFrame(
        columns=[
            "Late_delivery_risk",
            "Days for shipping (real)",
            "Days for shipment (scheduled)",
            "order date (DateOrders)",
            "shipping date (DateOrders)",
        ]
    )

    result = standardize_column_names(data)

    assert list(result.columns) == [
        "late_delivery_risk",
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
        "order_date_dateorders",
        "shipping_date_dateorders",
    ]


def test_column_name_collision_raises() -> None:
    data = pd.DataFrame([[1, 2]], columns=["A B", "A-B"])

    with pytest.raises(ValueError, match="duplicate names"):
        standardize_column_names(data)


def test_exact_duplicate_removal_only_removes_identical_rows() -> None:
    data = pd.DataFrame({"order_id": [1, 1, 1], "item": ["A", "A", "B"]})

    cleaned, removed = remove_exact_duplicates(data)

    assert removed == 1
    assert len(cleaned) == 2
    assert cleaned["order_id"].tolist() == [1, 1]


def test_missingness_summary_is_sorted() -> None:
    data = pd.DataFrame({"complete": [1, 2], "half_missing": [None, 1]})

    result = summarize_missingness(data)

    assert result.iloc[0].to_dict() == {
        "column": "half_missing",
        "missing_count": 1,
        "missing_percentage": 50.0,
    }


def test_date_conversion_records_failures_without_dropping_rows() -> None:
    data = pd.DataFrame(
        {
            "order_date_dateorders": ["1/1/2018 00:00", "not-a-date"],
            "shipping_date_dateorders": ["1/2/2018 00:00", None],
        }
    )

    converted, metadata = parse_dates(data)

    assert len(converted) == 2
    assert metadata["order_date_dateorders"]["successful_conversions"] == 1
    assert metadata["order_date_dateorders"]["failed_conversions"] == 1
    assert metadata["shipping_date_dateorders"]["missing_input_values"] == 1
    assert pd.api.types.is_datetime64_any_dtype(converted["order_date_dateorders"])


def test_direct_pii_is_removed_but_geography_is_retained() -> None:
    normalized = standardize_column_names(source_rows())

    cleaned, removed = remove_direct_pii(normalized)

    assert set(removed) == {
        "customer_email",
        "customer_password",
        "customer_fname",
        "customer_lname",
        "customer_street",
    }
    assert "customer_country" in cleaned.columns


def test_full_cleaning_retains_repeated_order_items_and_target() -> None:
    source = source_rows()

    cleaned, metadata = clean_supply_chain_data(source)

    assert len(cleaned) == 2
    assert metadata["exact_duplicates_removed"] == 0
    assert metadata["unique_order_ids"] == 1
    assert metadata["repeated_order_id_rows"] == 1
    assert "order_id" in cleaned.columns
    assert "late_delivery_risk" in cleaned.columns
    assert "product_description" not in cleaned.columns
    assert "customer_fname" not in cleaned.columns
    assert "Order Id" in source.columns
