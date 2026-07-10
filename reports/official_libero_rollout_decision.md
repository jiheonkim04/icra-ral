# Official LIBERO Rollout Decision

Date: 2026-07-10 KST

Final decision: `OFFICIAL_ROLLOUT_BASELINE_READY`

## Why

- WSL2 Ubuntu and RTX 5080 CUDA are usable.
- Official LeRobot/LIBERO dependencies import and run.
- Frozen base and all three persisted rank-4 LoRA policies load.
- Model parameters, preprocessed tensors, and action chunks are on `cuda:0`.
- The official smoke completed all `4/4` planned episodes.
- The bounded official pilot completed all `48/48` planned episodes.
- No schema/action mismatch or old custom `LIBERO_7D` route was used.
- Static-mix duplicates were skipped because alpha is exactly `0.0`.

## Pilot Results

| Policy | Offline L2 | Overall success | Avg reward | Peak CUDA alloc |
| --- | ---: | ---: | ---: | ---: |
| `frozen_base` | `0.085579125` | `75.0%` | `0.75` | `926.638 MiB` |
| `rank4_lora_seed_11` | `0.086743582` | `83.3%` | `0.8333` | `928.365 MiB` |
| `rank4_lora_seed_22` | `0.086474081` | `66.7%` | `0.6667` | `928.365 MiB` |
| `rank4_lora_seed_33` | `0.086918872` | `75.0%` | `0.75` | `928.365 MiB` |

Runtime and official eval latency:

| Policy | Env creation | Rollout wall time | Official eval seconds per episode |
| --- | ---: | ---: | ---: |
| `frozen_base` | `38.351s` | `223.785s` | `17.013s` |
| `rank4_lora_seed_11` | `50.296s` | `268.014s` | `20.400s` |
| `rank4_lora_seed_22` | `53.676s` | `249.425s` | `18.816s` |
| `rank4_lora_seed_33` | `53.197s` | `260.552s` | `19.902s` |

Pilot total runtime was `1338.704s`. The official metrics written by `eval_policy_all` exposed success, reward, video paths, total eval seconds, and eval seconds per episode; it did not expose episode length or forward-pass count in the saved result.

## Answers

1. Lower offline L2 did not correspond to higher success in this pilot.
2. Seed 11 improved over frozen base in the bounded pilot; seeds 22 and 33 did not.
3. LoRA effects were not consistent across suites: seed 11 helped `libero_10`, seed 22 helped `libero_spatial` but hurt `libero_object`, seed 33 matched frozen base.
4. Failure categories cannot be assigned at video-level precision because videos were disabled; by task group, failures concentrate most in `libero_10` long-horizon execution and some spatial/object placement cases.
5. Some failures are reset-consistent, especially the recurring `010` pattern in `libero_10`.
6. No method-worthy structured gap is established yet.
7. The pilot is enough to declare official baseline readiness, not enough for method selection.

## Asset Note

Models, checkpoints, and dataset demos were not redownloaded. During an early one-step official environment smoke, `hf-libero` auto-downloaded missing package-local LIBERO assets before the package asset path was symlinked to the copied LIBERO repo assets. This is recorded in `reports/wsl_official_rollout_asset_manifest.json`; the bounded pilot used the repaired package-local asset path.

## Exact Next Step

Run a larger official baseline rollout/failure-mining pass with the same frozen base and all three LoRA seeds, enable official rollout videos for failed episodes, and keep seed/reporting predeclared. Do not select a best LoRA seed or design a new method from this 48-episode pilot alone.
