# DAGR-VLA Stage A Result

Date: `2026-07-14 KST`

Final decision: `DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

- planned episodes: `50`
- completed episodes: `50`
- closed-loop experiment happened: `True`
- confirmatory-test tuning happened: `False`
- elapsed seconds: `1098.191`

## Policy Summary

| policy | successes | total | task-balanced success | exceptions | activation | delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 8 | 10 | 0.8 | 0 | 0.0 | 0.0 |
| `dam_static_component_proxy` | 2 | 10 | 0.2 | 0 | 1.0 | 0.299515091 |
| `dagr_full` | 6 | 10 | 0.6 | 0 | 1.0 | 0.056203212 |
| `dagr_no_dynamic_route_ablation` | 5 | 10 | 0.5 | 0 | 1.0 | 0.06388505 |
| `gripper_transition_heuristic` | 7 | 10 | 0.7 | 0 | 0.016867 | 0.000843327 |

## Paired Versus DAGR Full

| baseline | pairs | wins | losses | ties | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 10 | 0 | 2 | 8 | -0.2 |
| `dam_static_component_proxy` | 10 | 4 | 0 | 6 | 0.4 |
| `dagr_no_dynamic_route_ablation` | 10 | 2 | 1 | 7 | 0.1 |
| `gripper_transition_heuristic` | 10 | 1 | 2 | 7 | -0.1 |

Next step: Run Stage B on a frozen expansion manifest.
