# MTF-VLA Validation Search

Date: `2026-07-14`

Proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`

Final decision: `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING`

- closed-loop experiment happened: `False`
- training happened: `False`
- confirmatory-test tuning happened: `False`
- search budget: `6 configs: retained ratio in {0.20, 0.30}, retention coefficient in {0.25, 0.50, 1.00}`
- tried configs: `6`
- selected config: `mtf_r20_ret100`
- selected score: `0.6436633752294507`
- selected retained ratio: `0.2`
- selected retention coefficient: `1.0`
- selected train records: `1200`
- selected MTF high frames: `176`
- selected MTF retention frames: `391`

Score weights:

```json
{
  "action_validity_and_bounded_deltas": 0.1,
  "clean_retention": 0.25,
  "compute_overhead": 0.1,
  "mechanism_activation_and_score_health": 0.2,
  "validation_closed_loop_or_closest_feasible_proxy": 0.35
}
```

Selected config:

```json
{
  "audit_final_decision": "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH",
  "config_id": "mtf_r20_ret100",
  "final_decision": "VALIDATION_CONFIG_PASS",
  "gripper_transition_fraction": 0.341875,
  "hard_stop_reasons": [],
  "high_low_score_gap": 0.5857019297500093,
  "high_train_frames": 176,
  "retained_high_frame_ratio": 0.2,
  "retention_coefficient": 1.0,
  "retention_train_frames": 391,
  "score_terms": {
    "action_validity_and_bounded_delta": 0.9303362502855659,
    "base_action_l2_global_validation": 0.08630366897708504,
    "base_action_l2_high_validation": 0.09724746549783211,
    "base_action_l2_low_validation": 0.08069325967443643,
    "base_action_l2_uniform_validation": 0.10202290164275447,
    "clean_retention": 1.0,
    "compute_overhead": 1.0,
    "lora_minus_base_delta_p95_validation": 0.1393274994288682,
    "mechanism_activation": 0.8353119687777819,
    "total": 0.6436633752294507,
    "validation_closed_loop_proxy": 0.09590673270096481
  },
  "train_records": 1200,
  "uniform_overlap_fraction": 0.2175,
  "validation_records": 400
}
```

Tried configurations:

| config | decision | ratio | retention | train high | train retention | proxy | clean | mechanism | validity | compute | total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mtf_r20_ret025` | `VALIDATION_CONFIG_PASS` | 0.2 | 0.25 | 176 | 391 | 0.09590673270096481 | 0.25 | 0.8353119687777819 | 0.9303362502855659 | 1.0 | 0.4561633752294507 |
| `mtf_r20_ret050` | `VALIDATION_CONFIG_PASS` | 0.2 | 0.5 | 176 | 391 | 0.09590673270096481 | 0.5 | 0.8353119687777819 | 0.9303362502855659 | 1.0 | 0.5186633752294507 |
| `mtf_r20_ret100` | `VALIDATION_CONFIG_PASS` | 0.2 | 1.0 | 176 | 391 | 0.09590673270096481 | 1.0 | 0.8353119687777819 | 0.9303362502855659 | 1.0 | 0.6436633752294507 |
| `mtf_r30_ret025` | `VALIDATION_CONFIG_PASS` | 0.3 | 0.25 | 319 | 558 | 0.11179614326010309 | 0.25 | 0.8393338132196049 | 0.9303362502855659 | 0.8 | 0.44252903781351366 |
| `mtf_r30_ret050` | `VALIDATION_CONFIG_PASS` | 0.3 | 0.5 | 319 | 558 | 0.11179614326010309 | 0.5 | 0.8393338132196049 | 0.9303362502855659 | 0.8 | 0.5050290378135136 |
| `mtf_r30_ret100` | `VALIDATION_CONFIG_PASS` | 0.3 | 1.0 | 319 | 558 | 0.11179614326010309 | 1.0 | 0.8393338132196049 | 0.9303362502855659 | 0.8 | 0.6300290378135136 |

Checkpoint status:

- no adapter checkpoint was trained in this validation-search step;
- selected training manifest is frozen for the next adapter-training step;
- Stage A must not start before disk-reloadable checkpoints exist.

Next step: Freeze this config and train disk-reloadable MTF, no-retention, FrameSkip-proxy, and uniform adapter checkpoints before Stage A.
