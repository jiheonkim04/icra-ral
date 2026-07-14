# EAC-VLA Stage B Result

Date: `2026-07-15`

Final decision: `EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`

- planned episodes: `200`
- completed episodes: `200`
- infrastructure failures: `0`
- confirmatory-test tuning happened: `False`

## Policy Success

| Policy | Successes | Total | Rate | Task-balanced | Avg policy calls/step | Commitments |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `frozen_smolvla_fixed_queue` | `30` | `40` | `75.0%` | `0.75` | `0.022919` | `{'50': 178}` |
| `aac_entropy_proxy` | `30` | `40` | `75.0%` | `0.75` | `0.091915` | `{'2': 170, '50': 162, '8': 74}` |
| `eac_full` | `29` | `40` | `72.5%` | `0.725` | `0.251983` | `{'1': 807, '4': 199, '50': 148}` |
| `eac_no_calibration_no_hysteresis_ablation` | `30` | `40` | `75.0%` | `0.75` | `0.075413` | `{'1': 100, '4': 64, '50': 167}` |
| `fixed_short_replan_baseline` | `29` | `40` | `72.5%` | `0.725` | `1.0` | `{'1': 8697}` |

## Paired Versus EAC Full

```json
{
  "aac_entropy_proxy": {
    "eac_full_losses": 4,
    "eac_full_wins": 3,
    "paired_bootstrap_ci": [
      -0.15,
      0.1
    ],
    "paired_delta_eac_minus_policy": -0.025,
    "ties": 33
  },
  "eac_no_calibration_no_hysteresis_ablation": {
    "eac_full_losses": 5,
    "eac_full_wins": 4,
    "paired_bootstrap_ci": [
      -0.175,
      0.125
    ],
    "paired_delta_eac_minus_policy": -0.025,
    "ties": 31
  },
  "fixed_short_replan_baseline": {
    "eac_full_losses": 5,
    "eac_full_wins": 5,
    "paired_bootstrap_ci": [
      -0.15,
      0.15
    ],
    "paired_delta_eac_minus_policy": 0.0,
    "ties": 30
  },
  "frozen_smolvla_fixed_queue": {
    "eac_full_losses": 5,
    "eac_full_wins": 4,
    "paired_bootstrap_ci": [
      -0.175,
      0.125
    ],
    "paired_delta_eac_minus_policy": -0.025,
    "ties": 31
  }
}
```

EAC Stage B uses the frozen all-task manifest; no Stage B outcome may retune this EAC configuration.
