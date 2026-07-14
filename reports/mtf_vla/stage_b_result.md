# MTF-VLA Stage B Result

Date: `2026-07-14 KST`

Final decision: `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`

- planned episodes: `200`
- completed episodes: `200`
- closed-loop experiment happened: `True`
- confirmatory-test tuning happened: `False`
- elapsed seconds: `3815.594`

## Policy Summary

| policy | successes | total | task-balanced success | exceptions | latency mean s | peak VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 28 | 40 | 0.7 | 0 | 0.007269 | 926.638 |
| `frameskip_proxy_lora` | 27 | 40 | 0.675 | 0 | 0.009853 | 928.365 |
| `uniform_retained_ratio_lora` | 29 | 40 | 0.725 | 0 | 0.010068 | 928.365 |
| `mtf_no_retention_ablation` | 32 | 40 | 0.8 | 0 | 0.009895 | 928.365 |
| `mtf_full` | 26 | 40 | 0.65 | 0 | 0.010173 | 928.365 |

## Paired Versus MTF Full

| baseline | pairs | wins | losses | ties | delta | CI 95% |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `frozen_smolvla` | 40 | 2 | 4 | 34 | -0.05 | [-0.175, 0.075] |
| `frameskip_proxy_lora` | 40 | 1 | 2 | 37 | -0.025 | [-0.1, 0.05] |
| `uniform_retained_ratio_lora` | 40 | 0 | 3 | 37 | -0.075 | [-0.175, 0.0] |
| `mtf_no_retention_ablation` | 40 | 1 | 7 | 32 | -0.15 | [-0.275, -0.025] |

Next step: Archive or pivot under the preregistered governance; do not retune MTF from Stage B outcomes.
