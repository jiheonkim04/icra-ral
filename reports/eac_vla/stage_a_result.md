# EAC-VLA Stage A Result

Date: `2026-07-15`

Final decision: `EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

- planned episodes: `50`
- completed episodes: `50`
- infrastructure failures: `0`
- confirmatory-test tuning happened: `False`

## Policy Success

| Policy | Successes | Total | Rate | Task-balanced | Avg policy calls/step | Commitments |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `frozen_smolvla_fixed_queue` | `7` | `10` | `70.0%` | `0.7` | `0.021696` | `{'50': 42}` |
| `aac_entropy_proxy` | `9` | `10` | `90.0%` | `0.9` | `0.132209` | `{'2': 83, '50': 29, '8': 18}` |
| `eac_full` | `8` | `10` | `80.0%` | `0.8` | `0.207184` | `{'1': 150, '4': 25, '50': 33}` |
| `eac_no_calibration_no_hysteresis_ablation` | `7` | `10` | `70.0%` | `0.7` | `0.08404` | `{'1': 27, '4': 26, '50': 43}` |
| `fixed_short_replan_baseline` | `7` | `10` | `70.0%` | `0.7` | `1.0` | `{'1': 1805}` |

## Paired Versus EAC Full

```json
{
  "aac_entropy_proxy": {
    "eac_full_losses": 1,
    "eac_full_wins": 0,
    "paired_delta_eac_minus_policy": -0.1,
    "ties": 9
  },
  "eac_no_calibration_no_hysteresis_ablation": {
    "eac_full_losses": 0,
    "eac_full_wins": 1,
    "paired_delta_eac_minus_policy": 0.1,
    "ties": 9
  },
  "fixed_short_replan_baseline": {
    "eac_full_losses": 1,
    "eac_full_wins": 2,
    "paired_delta_eac_minus_policy": 0.1,
    "ties": 7
  },
  "frozen_smolvla_fixed_queue": {
    "eac_full_losses": 1,
    "eac_full_wins": 2,
    "paired_delta_eac_minus_policy": 0.1,
    "ties": 7
  }
}
```

The EAC scheduler preserves frozen SmolVLA action values and changes only queue commitment length; action-value modification would be an implementation failure.
