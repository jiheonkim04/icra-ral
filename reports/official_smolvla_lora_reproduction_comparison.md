# Official SmolVLA LoRA Reproduction Comparison

Date: 2026-07-10 KST

Final decision: `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`
Tolerance: `0.002`
Tolerance pass: `False`
Static-mix conclusion preserved: `True`

## Per-Seed Comparison

| seed | metric | old action L2 | regenerated action L2 | abs diff | pass | static ranking preserved |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 11 | `rank4_lora` | 0.084128699 | 0.087213856 | 0.003085157 | `False` | `True` |
| 11 | `validation_selected_action_space_static_mix` | 0.077354597 | 0.078911385 | 0.001556788 | `True` | `True` |
| 22 | `rank4_lora` | 0.090162398 | 0.088713382 | 0.001449016 | `True` | `True` |
| 22 | `validation_selected_action_space_static_mix` | 0.080789904 | 0.080982228 | 0.000192324 | `True` | `True` |
| 33 | `rank4_lora` | 0.090426934 | 0.085934428 | 0.004492506 | `False` | `True` |
| 33 | `validation_selected_action_space_static_mix` | 0.083704791 | 0.078716617 | 0.004988174 | `False` | `True` |

## Aggregate Comparison

| metric | old mean | old std | regenerated mean | regenerated std | mean abs diff | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rank4_lora` | 0.088239344 | 0.00290867 | 0.087287222 | 0.001135689 | 0.000952122 | `True` |
| `validation_selected_action_space_static_mix` | 0.080616431 | 0.002595356 | 0.079536743 | 0.0010252 | 0.001079688 | `True` |
| `frame_oracle_upper_bound` | 0.069117204 | 0.002049401 | 0.069590253 | 0.001577148 | 0.000473049 | `True` |
| `task_oracle_upper_bound` | 0.081138309 | 0.001955707 | 0.080959383 | 0.001502216 | 0.000178926 | `True` |
