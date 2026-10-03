"""Explicit feature policies that protect future modeling from leakage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


TARGET_COLUMNS = frozenset({"late_delivery_risk"})

EDA_ONLY_COLUMNS = frozenset(
    {
        "delivery_status",
        "days_for_shipping_real",
        "shipping_date_dateorders",
    }
)

IDENTIFIER_COLUMNS = frozenset(
    {
        "category_id",
        "customer_id",
        "department_id",
        "order_customer_id",
        "order_id",
        "order_item_cardprod_id",
        "order_item_id",
        "product_card_id",
        "product_category_id",
    }
)

PREDICTION_SAFE_CANDIDATES = frozenset(
    {
        "days_for_shipment_scheduled",
        "shipping_mode",
    }
)

REVIEW_REQUIRED_COLUMNS = frozenset(
    {
        "benefit_per_order",
        "category_name",
        "customer_city",
        "customer_country",
        "customer_segment",
        "customer_state",
        "customer_zipcode",
        "department_name",
        "latitude",
        "longitude",
        "market",
        "order_city",
        "order_country",
        "order_date_dateorders",
        "order_item_discount",
        "order_item_discount_rate",
        "order_item_product_price",
        "order_item_profit_ratio",
        "order_item_quantity",
        "order_item_total",
        "order_profit_per_order",
        "order_region",
        "order_state",
        "order_status",
        "order_zipcode",
        "product_image",
        "product_name",
        "product_price",
        "product_status",
        "sales",
        "sales_per_customer",
        "type",
    }
)

REMOVED_UNUSABLE_OR_PII_COLUMNS = frozenset(
    {
        "customer_email",
        "customer_fname",
        "customer_lname",
        "customer_password",
        "customer_street",
        "product_description",
    }
)

FORBIDDEN_MODEL_COLUMNS = frozenset(
    TARGET_COLUMNS | EDA_ONLY_COLUMNS | IDENTIFIER_COLUMNS
)

CLASSIFIED_COLUMNS = frozenset(
    FORBIDDEN_MODEL_COLUMNS
    | PREDICTION_SAFE_CANDIDATES
    | REVIEW_REQUIRED_COLUMNS
    | REMOVED_UNUSABLE_OR_PII_COLUMNS
)


def validate_model_features(
    columns: Iterable[str],
    *,
    allow_review_required: bool = False,
) -> None:
    """Reject target, post-outcome, identifier, or unreviewed model features.

    Review-required fields can be allowed only through an explicit opt-in after
    the team defines the prediction timestamp and documents availability.
    """
    selected = set(columns)
    forbidden = sorted(selected & FORBIDDEN_MODEL_COLUMNS)
    if forbidden:
        raise ValueError(f"Leakage-risk columns detected: {forbidden}")

    removed = sorted(selected & REMOVED_UNUSABLE_OR_PII_COLUMNS)
    if removed:
        raise ValueError(f"Removed unusable/PII columns cannot be modeled: {removed}")

    review_required = sorted(selected & REVIEW_REQUIRED_COLUMNS)
    if review_required and not allow_review_required:
        raise ValueError(
            "Review-required columns need an explicit timing decision: "
            f"{review_required}"
        )

    unclassified = sorted(selected - CLASSIFIED_COLUMNS)
    if unclassified:
        raise ValueError(
            "Unclassified columns require feature-policy review before modeling: "
            f"{unclassified}"
        )


def assert_no_group_overlap(
    train_order_ids: Iterable[Any],
    test_order_ids: Iterable[Any],
) -> None:
    """Require order-level isolation between future train and test partitions."""
    train = set(pd.Series(list(train_order_ids)).dropna())
    test = set(pd.Series(list(test_order_ids)).dropna())
    overlap = train & test
    if overlap:
        raise ValueError(
            "Order-group leakage detected: "
            f"{len(overlap)} order_id value(s) occur in both train and test."
        )
