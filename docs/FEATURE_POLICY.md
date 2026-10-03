# Feature and Leakage Policy

## Prediction question

Using information available at or near order/fulfillment decision time, estimate
whether an order is at risk of late delivery.

This timing definition is the controlling rule. A variable can be useful for
descriptive analysis without being eligible as a predictor.

## Target

- `late_delivery_risk`

The target must never be included in model features.

## Prediction-safe candidates

- `days_for_shipment_scheduled`
- `shipping_mode`

These fields describe the planned service and are potentially available in
advance. They encode closely related information, so future modeling should
review their redundancy. This classification does not guarantee usefulness.

## EDA-only and forbidden predictors

| Column | Reason |
|---|---|
| `delivery_status` | Effectively reveals the late-delivery outcome. |
| `days_for_shipping_real` | Actual duration is known only after shipment/delivery activity. |
| `shipping_date_dateorders` | Post-order/post-shipment information for an order-time prediction. |

These variables may support descriptive or data-quality work, but cannot be
used as predictors for the stated prediction question.

## Identifiers / grouping variables

These are `RETAINED_FOR_STRUCTURE_NOT_MODELING`:

- `category_id`
- `customer_id`
- `department_id`
- `order_customer_id`
- `order_id`
- `order_item_cardprod_id`
- `order_item_id`
- `product_card_id`
- `product_category_id`

They are forbidden as ordinary predictive features. `order_id` must be used to
keep all items from an order in the same future split.

## Review-required variables

All remaining analytical fields are review-required unless explicitly promoted
through a documented team decision. They include `order_status`, order-time and
location fields, customer/product categories, and financial/item fields.

`order_status` is specifically review-required because its eligibility depends
on when the prediction is generated. Post-decision status updates would leak
future information. Financial and profit fields likewise require confirmation
that their values are known at the prediction timestamp.

The authoritative complete set is `REVIEW_REQUIRED_COLUMNS` in
`src/leakage_guardrails.py`. New or derived columns are unclassified by default
and cause the automated guardrail to fail until the policy is updated.

## Removed fields

Direct PII (`customer_email`, `customer_password`, `customer_fname`,
`customer_lname`, and `customer_street`) and the all-missing
`product_description` field are removed during deterministic preprocessing and
cannot be model inputs.

## Automated enforcement

`validate_model_features(columns)` raises an error for target, outcome,
post-shipment, identifier, removed, review-required, or unclassified fields.
Review-required fields need an explicit opt-in after their timing has been
documented. Forbidden fields cannot be overridden.

`assert_no_group_overlap(train_order_ids, test_order_ids)` raises an error when
an order appears in both future partitions. A random row-level split is unsafe
because two items from the same order could otherwise enter different sets.

## Future splitting and learned preprocessing

No train/test split is created in Phase 1 preprocessing. A chronological split
may eventually be preferable for evaluating generalization to future orders,
but it must also preserve whole `order_id` groups.

Scalers, statistical imputers, encoders, feature selection, PCA, SMOTE, and any
other learned transformation must be fitted on training data only and then
applied to held-out data through a model pipeline.
