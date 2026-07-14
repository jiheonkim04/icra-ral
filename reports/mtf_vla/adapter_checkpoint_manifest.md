# MTF-VLA Adapter Checkpoint Manifest

Date: `2026-07-14 KST`

Final decision: `MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY`

- training happened: `True`
- closed-loop experiment happened: `False`
- confirmatory-test identities used: `False`
- Stage A allowed after matched-manifest freeze: `True`
- checkpoint root: `runs/mtf_vla_checkpoints/mtf_r20_ret100`

| variant | disk reload | events | action L2 | task-balanced L2 | adapter-base p95 | adapter sha256 | checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `frameskip_proxy_lora` | `True` | 240 | 0.08255313 | 0.08255313 | 0.130987061 | `2E7CF4E4170145979CDCDA9B3348622BB1F25FAD6075F5C7ABFF1BFF14B1C188` | `runs\mtf_vla_checkpoints\mtf_r20_ret100\frameskip_proxy_lora\seed_101` |
| `mtf_full` | `True` | 567 | 0.082590885 | 0.082590886 | 0.127396702 | `7CA3A765E864BB6708E8B344BA35AF06F19107EC5FFDA132DA9DEBB9D301871B` | `runs\mtf_vla_checkpoints\mtf_r20_ret100\mtf_full\seed_101` |
| `mtf_no_retention_ablation` | `True` | 176 | 0.082867367 | 0.082867367 | 0.129667163 | `23D47A0568A8A88BA7B1B3E164E59DF675CD2DD7388AE36319439CDCB17659FB` | `runs\mtf_vla_checkpoints\mtf_r20_ret100\mtf_no_retention_ablation\seed_101` |
| `uniform_retained_ratio_lora` | `True` | 240 | 0.082396918 | 0.082396918 | 0.132568751 | `9DA654A1FD52EBABF3C014D779DF191B13C880792350CCAB43710FA99BBBF7A5` | `runs\mtf_vla_checkpoints\mtf_r20_ret100\uniform_retained_ratio_lora\seed_101` |

Next step: freeze a matched Stage A rollout manifest before any rollout.
