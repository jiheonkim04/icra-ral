# MTF-VLA Adapter Training

Date: `2026-07-14 KST`

Final decision: `MTF_ADAPTER_TRAINING_PLAN_READY`

- dry run: `True`
- training happened: `False`
- closed-loop experiment happened: `False`
- confirmatory-test identities used: `False`
- Stage A allowed: `False`
- config: `mtf_r20_ret100`
- seed: `101`
- steps: `100`

Retention target implementation:

```json
{
  "known_limitation": "full frozen-base action chunks are unavailable in the stable prediction artifact, so retention overrides only the current action before the official preprocessor builds the native loss target",
  "not_used": "KL between deterministic actions",
  "scope": "current 7D action on retention frames",
  "uses_frozen_base_action": true
}
```

Jobs:

| variant | events | demo | retention | tasks | episodes | checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mtf_full` | 567 | 176 | 391 | 40 | 80 | `runs\mtf_vla_checkpoints\mtf_r20_ret100\mtf_full\seed_101` |
| `mtf_no_retention_ablation` | 176 | 176 | 0 | 40 | 71 | `runs\mtf_vla_checkpoints\mtf_r20_ret100\mtf_no_retention_ablation\seed_101` |
| `frameskip_proxy_lora` | 176 | 176 | 0 | 40 | 71 | `runs\mtf_vla_checkpoints\mtf_r20_ret100\frameskip_proxy_lora\seed_101` |
| `uniform_retained_ratio_lora` | 240 | 240 | 0 | 40 | 78 | `runs\mtf_vla_checkpoints\mtf_r20_ret100\uniform_retained_ratio_lora\seed_101` |

Hard stop reasons:
- none

Next step: Run the MTF adapter trainer to produce disk-reloadable checkpoints for all four trainable Stage A policies.
