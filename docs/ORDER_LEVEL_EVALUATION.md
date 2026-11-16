# Supplemental Unique-Order Evaluation

The deployed question is whether an order will be late, while the frozen final report weights every order-item row. This supplemental analysis gives every `order_id` one vote. It does not replace or alter the frozen item-level evaluation.

Before aggregation, the runner reproduces the tracked validation and test item-level metrics and verifies the frozen configuration hash, fitted-model fingerprint, split manifest, source checksum, and validation reference. It then requires each order's target, approved features, order timestamp, and model score to be constant across its item rows. Any disagreement fails the run.

## Results

| Partition | Unique orders | Late orders | Average precision | Precision at 0.50 | Recall at 0.50 | False alerts | Missed late orders |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 13,150 | 7,172 | 0.745603 | 84.72% | 53.49% | 692 | 3,336 |
| Final test | 13,151 | 7,228 | 0.749332 | 84.76% | 53.72% | 698 | 3,345 |

The order-weighted and item-weighted results are close, so repeated line items do not materially change the conclusion: the frozen depth-3 tree produces comparatively precise alerts but misses roughly 46% of late orders at the already-fixed threshold.

These results are post-hoc and supplemental. The final test has already been examined, so these numbers must not be used to change the model, features, preprocessing, or threshold. Any future candidate must be specified and evaluated under a new untouched-test protocol.

## Reproduce

Place the canonical ignored DataCo CSV at `data/raw/DataCoSupplyChainDataset.csv`, then run:

```bash
python -m src.project_preflight
python -m src.order_level_evaluation
```

Machine-readable outputs are in [`reports/order_level_evaluation.json`](../reports/order_level_evaluation.json) and [`reports/order_level_evaluation.csv`](../reports/order_level_evaluation.csv).
