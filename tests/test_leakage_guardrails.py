"""Tests for model-feature and grouped-split guardrails."""

import pytest

from src.leakage_guardrails import (
    assert_no_group_overlap,
    validate_model_features,
)


@pytest.mark.parametrize(
    "column",
    [
        "late_delivery_risk",
        "delivery_status",
        "days_for_shipping_real",
        "shipping_date_dateorders",
        "order_id",
    ],
)
def test_forbidden_model_columns_raise(column: str) -> None:
    with pytest.raises(ValueError, match="Leakage-risk columns detected"):
        validate_model_features([column])


def test_prediction_safe_candidates_pass() -> None:
    validate_model_features(["days_for_shipment_scheduled", "shipping_mode"])


def test_review_required_column_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="Review-required"):
        validate_model_features(["order_status"])

    validate_model_features(["order_status"], allow_review_required=True)


def test_unclassified_column_raises() -> None:
    with pytest.raises(ValueError, match="Unclassified"):
        validate_model_features(["future_derived_feature"])


def test_group_overlap_raises_without_printing_ids() -> None:
    with pytest.raises(ValueError, match="1 order_id"):
        assert_no_group_overlap([100, 101], [101, 102])


def test_disjoint_groups_pass() -> None:
    assert_no_group_overlap([100, 101], [102, 103])
