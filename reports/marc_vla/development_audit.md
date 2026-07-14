# MARC-VLA Development Audit

Date: `2026-07-15`

Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`

Final decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

- closed-loop experiment happened: `False`
- training happened: `False`
- confirmatory-test tuning happened: `False`
- scoreable development records: `1600`
- train records: `1200`
- validation records: `400`
- reserved records not used: `1200`
- selected task count: `40`
- duplicate sample keys: `0`
- duplicate frame keys: `0`
- train disagreement positive fraction: `0.4`
- validation disagreement positive fraction: `0.44`
- gate probe margin: `0.04749999999999999`
- full-vs-L1 target mean L2: `0.026160557411240078`
- full-vs-no-gate target mean L2: `0.026160557411240078`
- full-vs-static target mean L2: `0.046550089904003`
- base action L2 validation: `0.08630366897708504`
- mean action L2 validation: `1.1880880851062106`
- preexisting LoRA action L2 validation: `0.08069799087861679`
- initial action delta p95: `0.0`
- base action validity: `1.0`

Disagreement thresholds:

```json
{
  "disagreement_l2_quantile_0_60": 0.07210158833092721
}
```

Gate probe summary:

```json
{
  "accuracy": 0.6075,
  "accuracy_margin": 0.04749999999999999,
  "first_gradient_norm": 0.19063152422255056,
  "majority_accuracy": 0.56,
  "mean_probability": 0.49962456718428894,
  "predicted_positive_fraction": 0.5075,
  "train_loss_final": 0.6347582846350418,
  "train_loss_initial": 0.6931471805599453,
  "valid": 1.0,
  "validation_loss": 0.667472274075055
}
```

Split manifest:

```json
{
  "confirmatory_reserved_splits": [
    "test"
  ],
  "reserved_record_count": 1200,
  "split_overlap": {
    "train_reserved": 0,
    "train_validation": 0,
    "validation_reserved": 0
  },
  "train_record_count": 1200,
  "train_splits": [
    "train"
  ],
  "validation_record_count": 400,
  "validation_splits": [
    "val"
  ]
}
```

Hard stop reasons:
- none

Next step: Run the bounded six-configuration MARC validation search.
