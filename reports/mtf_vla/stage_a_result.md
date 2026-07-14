# MTF-VLA Stage A Result

Date: `2026-07-14 KST`

Final decision: `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

- planned episodes: `50`
- completed episodes: `50`
- closed-loop experiment happened: `True`
- confirmatory-test tuning happened: `False`
- elapsed seconds: `1018.181`

## Policy Summary

| policy | successes | total | task-balanced success | exceptions |
| --- | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 8 | 10 | 0.8 | 0 |
| `frameskip_proxy_lora` | 8 | 10 | 0.8 | 0 |
| `uniform_retained_ratio_lora` | 8 | 10 | 0.8 | 0 |
| `mtf_no_retention_ablation` | 7 | 10 | 0.7 | 0 |
| `mtf_full` | 7 | 10 | 0.7 | 0 |

## Paired Versus MTF Full

| baseline | pairs | wins | losses | ties | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 10 | 1 | 2 | 7 | -0.1 |
| `frameskip_proxy_lora` | 10 | 0 | 1 | 9 | -0.1 |
| `uniform_retained_ratio_lora` | 10 | 1 | 2 | 7 | -0.1 |
| `mtf_no_retention_ablation` | 10 | 1 | 1 | 8 | 0.0 |

Next step: Run Stage B on the frozen expansion manifest.
