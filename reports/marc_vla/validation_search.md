# MARC-VLA Validation Search

Date: `2026-07-15`

Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`

Final decision: `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING`

- closed-loop experiment happened: `False`
- lightweight validation training happened: `True`
- confirmatory-test tuning happened: `False`
- audit final decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- search budget: `6 configs: correction alpha in {0.05, 0.10, 0.20} x gate architecture in {linear, mlp}`
- tried configs: `6`
- selected config: `marc_a020_gate_mlp`
- selected correction alpha: `0.2`
- selected gate architecture: `mlp`
- selected score: `0.5457964262366295`
- selected delta L2 p95: `0.011818917468190193`
- selected clean delta L2 p95: `0.010853752493858337`
- selected action validity: `1.0`
- selected MARC action L2: `0.08665236806523112`
- selected L1 proxy action L2: `0.08763420091414227`

Score weights:

```json
{
  "action_validity": 0.1,
  "clean_action_retention_and_bounded_deltas": 0.2,
  "compute_overhead": 0.05,
  "full_versus_no_gate_and_static_distinction": 0.15,
  "gate_predictability_above_majority": 0.25,
  "l1_proxy_validity_and_full_proxy_distinction": 0.25
}
```

Selected config:

```json
{
  "checkpoint_path": "reports\\marc_vla\\validation_checkpoints\\marc_a020_gate_mlp.pt",
  "checkpoint_reload_max_abs_diff": 0.0,
  "config_id": "marc_a020_gate_mlp",
  "correction_alpha": 0.2,
  "final_decision": "VALIDATION_CONFIG_PASS",
  "first_gradient_norms": {
    "anchor_residual": 0.0012295743917574429,
    "gate": 0.6870964448112337,
    "trunk": 0.17996795163687104
  },
  "gate_architecture": "mlp",
  "gate_metrics": {
    "accuracy": 0.6125,
    "accuracy_margin": 0.05249999999999999,
    "majority_accuracy": 0.56,
    "mean_probability": 0.40397125482559204,
    "predicted_positive_fraction": 0.3325
  },
  "hard_stop_reasons": [],
  "initial_delta_p95": 0.0,
  "loss_final_train": {
    "anchor": 0.0012159690959379077,
    "clean": 1.4354229278978892e-05,
    "delta": 3.426726834732108e-05,
    "gate": 0.5820536613464355,
    "total": 0.5832744836807251
  },
  "loss_initial": {
    "anchor": 0.0012350187171250582,
    "clean": 0.0,
    "delta": 0.0,
    "gate": 2.29272723197937,
    "total": 2.293962240219116
  },
  "loss_validation": {
    "anchor": 0.0016342008020728827,
    "clean": 1.6174186384887435e-05,
    "delta": 3.758813545573503e-05,
    "gate": 0.6703832149505615,
    "total": 0.6720227599143982
  },
  "proposal_hash": "D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A",
  "score_terms": {
    "action_validity": 1.0,
    "clean_retention_and_bounded_delta": 0.9457312375307083,
    "compute_overhead": 0.95,
    "full_ablation_static_distinction": 0.19475044682621956,
    "gate_predictability": 0.5249999999999999,
    "l1_proxy_validity_and_full_proxy_distinction": 0.5973752234131098,
    "total": 0.5457964262366295
  },
  "validation_metrics": {
    "action_validity": 1.0,
    "clean_delta_l2_p95": 0.010853752493858337,
    "delta_l2_mean": 0.005052071996033192,
    "delta_l2_p95": 0.011818917468190193,
    "full_vs_l1_proxy_mean_l2": 0.007010325323790312,
    "full_vs_no_gate_mean_l2": 0.007010325323790312,
    "full_vs_static_mean_l2": 0.0019475044682621956,
    "l1_proxy_action_l2": 0.08763420091414227,
    "l1_proxy_validity": 1.0,
    "marc_full_action_l2": 0.08665236806523112
  }
}
```

Tried configurations:

| config | decision | alpha | arch | proxy | gate | clean | distinction | validity | compute | total |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `marc_a005_gate_linear` | `VALIDATION_CONFIG_STOP` | 0.05 | `linear` | 0.939223600551486 | 0.0 | 0.9948376783868298 | 0.878447201102972 | 1.0 | 1.0 | 0.7003464161185547 |
| `marc_a010_gate_linear` | `VALIDATION_CONFIG_STOP` | 0.1 | `linear` | 0.9408373031765223 | 0.0 | 0.9950318288756534 | 0.8816746063530445 | 1.0 | 1.0 | 0.7016762083163485 |
| `marc_a020_gate_linear` | `VALIDATION_CONFIG_STOP` | 0.2 | `linear` | 0.9400393925607204 | 0.0 | 0.9952964042313397 | 0.8800787851214409 | 1.0 | 1.0 | 0.7010907948948443 |
| `marc_a005_gate_mlp` | `VALIDATION_CONFIG_STOP` | 0.05 | `mlp` | 0.6016892725601792 | 0.14999999999999902 | 0.954125969670713 | 0.20337854512035847 | 1.0 | 0.95 | 0.45717661198228576 |
| `marc_a010_gate_mlp` | `VALIDATION_CONFIG_PASS` | 0.1 | `mlp` | 0.6090936828404665 | 0.2749999999999997 | 0.9438977101817727 | 0.218187365680933 | 1.0 | 0.95 | 0.49230448830872764 |
| `marc_a020_gate_mlp` | `VALIDATION_CONFIG_PASS` | 0.2 | `mlp` | 0.5973752234131098 | 0.5249999999999999 | 0.9457312375307083 | 0.19475044682621956 | 1.0 | 0.95 | 0.5457964262366295 |

Next step: Freeze the selected MARC config and train disk-reloadable policy identities for the five-policy comparison before Stage A.
