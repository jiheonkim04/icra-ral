# PESA-VLA Development Audit

Date: `2026-07-15`

Proposal hash: `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`

Final decision: `DESIGN_FAILURE`

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
- train query positive fraction: `0.3858333333333333`
- validation query positive fraction: `0.4`
- query probe margin: `-0.07750000000000001`
- standard LoRA headroom L1 validation: `0.0065395455599999985`
- validation spectral active-rank mean: `2.3475`
- validation spectral active-fraction mean: `0.3353571428571428`
- full-vs-prior proxy mean L2: `0.017141366547216257`
- full-vs-ablation mean L2: `0.05027918595705619`
- full-vs-simple killer mean L2: `0.017141366547216257`
- selected simple killer: `clean_retention_lora_proxy`
- gradient norm ratio: `38.207967806463`
- initial action delta p95: `0.0`
- base action validity: `1.0`

Query thresholds:

```json
{
  "query_improvement_l1_threshold": 0.01,
  "spectral_eta": 0.85
}
```

Query probe summary:

```json
{
  "accuracy": 0.5225,
  "accuracy_margin": -0.07750000000000001,
  "first_gradient_norm": 0.12687499279580958,
  "majority_accuracy": 0.6,
  "mean_probability": 0.49941829294265944,
  "predicted_positive_fraction": 0.5375,
  "train_loss_final": 0.6652477821047845,
  "train_loss_initial": 0.6931471805599453,
  "valid": 1.0,
  "validation_loss": 0.7218456889479673
}
```

Spectral summary:

```json
{
  "active_fraction_mean": 0.3353571428571428,
  "active_fraction_p05": 0.14285714285714285,
  "active_fraction_p95": 0.5714285714285714,
  "active_rank_max": 6,
  "active_rank_mean": 2.3475,
  "active_rank_min": 1,
  "distinct_task_active_rank_profiles": 10,
  "entropy_mean": 0.47955830103721053,
  "entropy_p95": 0.7405529476558655,
  "task_active_rank_mean": {
    "0": 1.9,
    "1": 2.3,
    "10": 2.4,
    "11": 2.5,
    "12": 1.9,
    "13": 2.6,
    "14": 2.7,
    "15": 2.5,
    "16": 2.5,
    "17": 2.5,
    "18": 2.4,
    "19": 2.1,
    "2": 2.3,
    "20": 2.1,
    "21": 2.6,
    "22": 2.2,
    "23": 2.1,
    "24": 2.1,
    "25": 2.4,
    "26": 2.5,
    "27": 2.5,
    "28": 2.3,
    "29": 2.0,
    "3": 2.7,
    "30": 2.3,
    "31": 2.5,
    "32": 2.1,
    "33": 2.4,
    "34": 2.7,
    "35": 2.2,
    "36": 2.0,
    "37": 2.5,
    "38": 2.3,
    "39": 2.3,
    "4": 2.4,
    "5": 2.5,
    "6": 2.3,
    "7": 2.3,
    "8": 2.8,
    "9": 2.2
  },
  "total": 400
}
```

Gradient audit:

```json
{
  "batch_size": 128,
  "gradient_norm_ratio_largest_to_smallest": 38.207967806463,
  "gradient_norms": {
    "adaptation": 0.07641816212378542,
    "query": 0.3505483782527514,
    "spectral": 0.009174745436041092,
    "trunk": 0.16330673780871016
  },
  "loss_terms": {
    "adapt": 0.09365329891443253,
    "delta": 0.00019276903185527772,
    "emit": 0.0023096411023288965,
    "query": 0.8539631366729736,
    "spectral_mse": 0.36815327405929565,
    "total": 0.9683529734611511
  },
  "valid": 1.0
}
```

Split manifest:

```json
{
  "confirmatory_reserved_splits": [
    "test"
  ],
  "reserved_record_count": 1200,
  "reset_overlap": {
    "train_reserved": 0,
    "train_validation": 0,
    "validation_reserved": 0
  },
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
- `query probe accuracy margin below minimum: -0.077500`

Next step: Do not train or roll out PESA; classify the Stage 0 failure and continue to the next method cycle.
