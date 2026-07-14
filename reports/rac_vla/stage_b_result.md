# RAC-VLA STAGE-B Result

Date: `2026-07-14`

Final decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`

- closed-loop experiment happened: `True`
- completed episodes: `200` / `200`
- exception count: `0`
- strongest baseline: `rac_no_consequence_ablation`
- hidden shift: `x_attenuate`

| Variant | Successes | Total | Task-Balanced Success | Mean Gate | Mean Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_smolvla_shifted` | 1 | 40 | 0.025 | 0.0 | 0.0 |
| `reflective_history_proxy` | 1 | 40 | 0.025 | 0.707419 | 0.002809 |
| `rac_full` | 1 | 40 | 0.025 | 0.064427 | 0.000315 |
| `rac_no_consequence_ablation` | 2 | 40 | 0.05 | 0.000268 | 0.0 |
| `online_diagonal_inverse_gain` | 2 | 40 | 0.05 | 0.697809 | 0.002515 |

Next step: Archive, expand, or scale according to governance.
