# FANG-VLA STAGE-B Result

Date: `2026-07-14`

Final decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`

- closed-loop experiment happened: `True`
- completed episodes: `200` / `200`
- exception count: `0`
- strongest baseline: `base_smolvla`

| Variant | Successes | Total | Task-Balanced Success | Mean Gate | Mean Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_smolvla` | 16 | 40 | 0.4 | 0.0 | 0.0 |
| `afil_local_proxy` | 15 | 40 | 0.375 | 0.470062 | 0.016527 |
| `fang_full` | 11 | 40 | 0.275 | 0.086914 | 0.008217 |
| `fang_no_failure_ablation` | 11 | 40 | 0.275 | 0.097046 | 0.008521 |
| `nearest_success_replay` | 14 | 40 | 0.35 | 1.0 | 0.017719 |

Next step: Archive, expand, or scale according to governance.
