# MARC-VLA Stage A Result

Date: `2026-07-15 KST`

Final decision: `MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE`

- planned episodes: `50`
- completed episodes: `50`
- closed-loop experiment happened: `True`
- confirmatory-test tuning happened: `False`
- elapsed seconds: `1256.189`

## Policy Summary

| policy | successes | total | task-balanced success | exceptions | activation | delta L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 8 | 10 | 0.8 | 0 | 0.0 | 0.0 |
| `openvla_oft_l1_proxy` | 0 | 10 | 0.0 | 0 | 1.0 | 4.187040418 |
| `marc_full` | 0 | 10 | 0.0 | 0 | 1.0 | 0.181213643 |
| `marc_no_disagreement_gate_ablation` | 7 | 10 | 0.7 | 0 | 1.0 | 0.199302367 |
| `static_l1_mixture_baseline` | 7 | 10 | 0.7 | 0 | 1.0 | 0.079836804 |

## Paired Versus MARC Full

| baseline | pairs | wins | losses | ties | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | 10 | 0 | 8 | 2 | -0.8 |
| `openvla_oft_l1_proxy` | 10 | 0 | 0 | 10 | 0.0 |
| `marc_no_disagreement_gate_ablation` | 10 | 0 | 7 | 3 | -0.7 |
| `static_l1_mixture_baseline` | 10 | 0 | 7 | 3 | -0.7 |

Next step: Adjudicate repair or catastrophic kill under the preregistered governance.
