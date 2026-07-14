# DAGR-VLA Development Audit

Date: `2026-07-14`

Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`

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
- validation any-route fraction: `0.865`
- full-vs-shared target mean L2: `0.01581734353223074`
- full-vs-static target mean L2: `0.04481165090139941`
- initial action delta p95: `0.0`
- base action validity: `1.0`

Route thresholds:

```json
{
  "gripper_material": 0.02,
  "rotation": 0.011194201244082214,
  "translation": 0.05906000471441304
}
```

Train route label summary:

```json
{
  "gripper": {
    "negative_count": 776,
    "positive_count": 424,
    "positive_fraction": 0.35333333333333333,
    "total": 1200
  },
  "rotation": {
    "negative_count": 600,
    "positive_count": 600,
    "positive_fraction": 0.5,
    "total": 1200
  },
  "translation": {
    "negative_count": 600,
    "positive_count": 600,
    "positive_fraction": 0.5,
    "total": 1200
  }
}
```

Validation route label summary:

```json
{
  "gripper": {
    "negative_count": 191,
    "positive_count": 209,
    "positive_fraction": 0.5225,
    "total": 400
  },
  "rotation": {
    "negative_count": 190,
    "positive_count": 210,
    "positive_fraction": 0.525,
    "total": 400
  },
  "translation": {
    "negative_count": 183,
    "positive_count": 217,
    "positive_fraction": 0.5425,
    "total": 400
  }
}
```

Route probe summary:

```json
{
  "gripper": {
    "accuracy": 0.7825,
    "accuracy_margin": 0.26,
    "first_gradient_norm": 0.31321341806915504,
    "majority_accuracy": 0.5225,
    "mean_probability": 0.46815138153903846,
    "predicted_positive_fraction": 0.48,
    "train_loss_final": 0.49064355435241536,
    "train_loss_initial": 0.6931471805599453,
    "valid": 1.0,
    "validation_loss": 0.5106209429531919
  },
  "rotation": {
    "accuracy": 0.5975,
    "accuracy_margin": 0.07250000000000001,
    "first_gradient_norm": 0.20459572060344042,
    "majority_accuracy": 0.525,
    "mean_probability": 0.5007057975309973,
    "predicted_positive_fraction": 0.4775,
    "train_loss_final": 0.6303396886776109,
    "train_loss_initial": 0.6931471805599453,
    "valid": 1.0,
    "validation_loss": 0.6658261434429491
  },
  "translation": {
    "accuracy": 0.58,
    "accuracy_margin": 0.03749999999999998,
    "first_gradient_norm": 0.1733240991731677,
    "majority_accuracy": 0.5425,
    "mean_probability": 0.5081227153818817,
    "predicted_positive_fraction": 0.5125,
    "train_loss_final": 0.6414795675618605,
    "train_loss_initial": 0.6931471805599453,
    "valid": 1.0,
    "validation_loss": 0.678456769789048
  }
}
```

Route-label Jaccard on validation:

```json
{
  "rotation_gripper": 0.40604026845637586,
  "translation_gripper": 0.36977491961414793,
  "translation_rotation": 0.45733788395904434
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

Next step: Run the bounded six-configuration DAGR validation search.
