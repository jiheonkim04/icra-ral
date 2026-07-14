# RAC-VLA STAGE-A Result

Date: `2026-07-14`

Final decision: `STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

- closed-loop experiment happened: `True`
- completed episodes: `50` / `50`
- exception count: `0`
- strongest baseline: `reflective_history_proxy`
- hidden shift: `x_attenuate`

| Variant | Successes | Total | Task-Balanced Success | Mean Gate | Mean Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_smolvla_shifted` | 0 | 10 | 0.0 | 0.0 | 0.0 |
| `reflective_history_proxy` | 1 | 10 | 0.1 | 0.709626 | 0.002875 |
| `rac_full` | 0 | 10 | 0.0 | 0.064011 | 0.000299 |
| `rac_no_consequence_ablation` | 0 | 10 | 0.0 | 0.000357 | 0.0 |
| `online_diagonal_inverse_gain` | 1 | 10 | 0.1 | 0.722445 | 0.00269 |

Next step: Run Stage B.
