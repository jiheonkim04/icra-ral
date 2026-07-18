# Wrist-Dropout Condition Generality Result

- Decision: `WRIST_DROPOUT_REPEATED_PROBLEM_CONFIRMED`
- Run: `runs/xvla_prior/wrist_dropout_generality_20260718T1510KST`
- Policy: frozen official X-VLA only; no Ours, training, optimizer, or checkpoint.

## Aggregate

- Clean successes: `9/9`
- Dropout successes: `0/9`
- Paired clean-success → dropout-failure flips: `9`
- Tasks with flips: `libero_goal/task0, libero_object/task0, libero_spatial/task5`
- Model forward count: `310`
- CUDA devices: `NVIDIA GeForce RTX 5080`
- Peak CUDA max allocated MiB: `3518.634`

## Paired uncertainty

- Exact McNemar two-sided p-value: `0.00390625`
- Degradation rate among clean-success pairs: `1.0`
- Wilson 95% CI: `{'low': 0.7008549515804559, 'high': 1.0}`

## Task distribution

| task | pairs | clean successes | dropout successes | clean→dropout flips |
|---|---:|---:|---:|---:|
| libero_goal/task0 | 3 | 3 | 0 | 3 |
| libero_object/task0 | 3 | 3 | 0 | 3 |
| libero_spatial/task5 | 3 | 3 | 0 | 3 |

## Pairs

| task | identity | clean success | dropout success | clean steps | dropout steps | clean chunks | dropout chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_goal/task0 | 20260733 | True | False | 112 | 900 | 4 | 30 |
| libero_goal/task0 | 20260734 | True | False | 114 | 900 | 4 | 30 |
| libero_goal/task0 | 20260735 | True | False | 119 | 900 | 4 | 30 |
| libero_object/task0 | 20260733 | True | False | 144 | 900 | 5 | 30 |
| libero_object/task0 | 20260734 | True | False | 144 | 900 | 5 | 30 |
| libero_object/task0 | 20260735 | True | False | 138 | 900 | 5 | 30 |
| libero_spatial/task5 | 20260731 | True | False | 89 | 900 | 3 | 30 |
| libero_spatial/task5 | 20260732 | True | False | 192 | 900 | 7 | 30 |
| libero_spatial/task5 | 20260735 | True | False | 90 | 900 | 3 | 30 |

Next action: Select an actual external prior before any learned Ours method.
