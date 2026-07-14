# FANG-VLA STAGE-A Result

Date: `2026-07-14`

Final decision: `STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

- closed-loop experiment happened: `True`
- completed episodes: `50` / `50`
- exception count: `0`
- strongest baseline: `base_smolvla`

| Variant | Successes | Total | Task-Balanced Success | Mean Gate | Mean Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_smolvla` | 3 | 10 | 0.3 | 0.0 | 0.0 |
| `afil_local_proxy` | 3 | 10 | 0.3 | 0.496001 | 0.017292 |
| `fang_full` | 3 | 10 | 0.3 | 0.095963 | 0.008186 |
| `fang_no_failure_ablation` | 3 | 10 | 0.3 | 0.101396 | 0.008356 |
| `nearest_success_replay` | 3 | 10 | 0.3 | 1.0 | 0.018354 |

Next step: Run Stage B.
