# DAGR-VLA Stage B Result

Date: `2026-07-14 KST`

Final decision: `DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`

- planned episodes: `200`
- completed episodes: `200`
- closed-loop experiment happened: `True`
- confirmatory-test tuning happened: `False`
- elapsed seconds: `4580.403`

## Policy Summary

| policy | successes | total | task-balanced success | exceptions | activation | delta L2 | latency mean s | peak VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 28 | 40 | 0.7 | 0 | 0.0 | 0.0 | 0.008024 | 926.645 |
| `dam_static_component_proxy` | 5 | 40 | 0.125 | 0 | 1.0 | 0.297116974 | 0.009636 | 927.666 |
| `dagr_full` | 18 | 40 | 0.45 | 0 | 0.999952 | 0.057304793 | 0.009427 | 927.667 |
| `dagr_no_dynamic_route_ablation` | 16 | 40 | 0.4 | 0 | 1.0 | 0.06611561 | 0.009015 | 927.667 |
| `gripper_transition_heuristic` | 24 | 40 | 0.6 | 0 | 0.016455 | 0.000822732 | 0.007731 | 927.656 |

## Paired Versus DAGR Full

| baseline | pairs | wins | losses | ties | delta | CI 95% |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `frozen_smolvla` | 40 | 1 | 11 | 28 | -0.25 | [-0.4, -0.1] |
| `dam_static_component_proxy` | 40 | 14 | 1 | 25 | 0.325 | [0.175, 0.475] |
| `dagr_no_dynamic_route_ablation` | 40 | 9 | 7 | 24 | 0.05 | [-0.15, 0.25] |
| `gripper_transition_heuristic` | 40 | 2 | 8 | 30 | -0.15 | [-0.3, 0.0] |

Next step: Archive or pivot under the preregistered governance; do not retune DAGR from Stage B outcomes.
