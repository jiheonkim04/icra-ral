# Official Closed-Loop Seed Robustness

Date: 2026-07-11 KST

```json
{
  "lora_policy_success_rates": {
    "rank4_lora_seed_11": 0.74,
    "rank4_lora_seed_22": 0.68,
    "rank4_lora_seed_33": 0.66
  },
  "lora_success_rate_std": 0.033993,
  "max_task_success_spread": 0.6,
  "mean_task_success_spread": 0.18
}
```

## Decision

LoRA training-seed variation is visible but not the primary gap. The three LoRA seeds span `66.0%` to `74.0%` aggregate success, with standard deviation `0.033993`. That spread is too small to justify selecting a seed after closed-loop outcomes, and it does not explain the repeated all-policy failures.

## Paired Outcome Versus Frozen Base

| Policy | Reset-level wins | Reset-level ties | Reset-level losses | Task-level wins | Task-level ties | Task-level losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank4_lora_seed_11` | `9` | `82` | `9` | `3` | `13` | `4` |
| `rank4_lora_seed_22` | `8` | `78` | `14` | `3` | `11` | `6` |
| `rank4_lora_seed_33` | `4` | `84` | `12` | `1` | `12` | `7` |

Seed `11` ties frozen_base at aggregate success, but the paired reset/task tables do not support a best-seed claim. All seeds remain reported separately.
