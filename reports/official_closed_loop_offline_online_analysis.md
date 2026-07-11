# Official Closed-Loop Offline-Online Analysis

Date: 2026-07-11 KST

```json
{
  "offline_l2": {
    "frozen_base": 0.085579125,
    "rank4_lora_seed_11": 0.086743582,
    "rank4_lora_seed_22": 0.086474081,
    "rank4_lora_seed_33": 0.086918872
  },
  "pearson_l2_vs_success": -0.569086,
  "spearman_l2_vs_success": -0.632456,
  "success_rate": {
    "frozen_base": 0.74,
    "rank4_lora_seed_11": 0.74,
    "rank4_lora_seed_22": 0.68,
    "rank4_lora_seed_33": 0.66
  }
}
```

Closed-loop task success is the primary evidence. Offline action L2 is reported only as a diagnostic comparison and is not used to select a LoRA seed.

## Policy Selection Interpretation

Lower offline action L2 is not selection-safe for the LoRA seeds:

| Policy | Offline action L2 | Closed-loop success |
| --- | ---: | ---: |
| `frozen_base` | `0.085579125` | `74.0%` |
| `rank4_lora_seed_11` | `0.086743582` | `74.0%` |
| `rank4_lora_seed_22` | `0.086474081` | `68.0%` |
| `rank4_lora_seed_33` | `0.086918872` | `66.0%` |

The all-policy Pearson/Spearman diagnostics are negative because the worst offline-L2 policy is also the weakest closed-loop policy, but the LoRA-only ordering is not reliable: seed `22` has lower offline L2 than seed `11`, while seed `11` has higher closed-loop success. Offline L2 therefore remains diagnostic-only and does not identify an intervention target.
