# Official SmolVLA Duplicate Run Audit

Date: 2026-07-10 KST

Scope: `72ed23e` through `5d48b1e`

Audit-only boundary: no experiments, no training, no GPU, no downloads, no simulator rollout, no OpenVLA-OFT, no LoRA seed rerun.

## Summary

- exact duplicate runs found: `0`
- possible exact duplicates found: `0`
- avoidable regenerations found: `2`
- necessary reproductions found: `3`
- repeated titles without duplicate evidence: treated as not duplicates

An exact duplicate was not inferred from similar report names. The audit required command/config/artifact evidence.

## Duplicate Classification

| item | evidence | classification | result impact |
| --- | --- | --- | --- |
| baseline scaleup vs failure mining | same official checkpoint/dataset and rank-4 LoRA 100 steps, but different sample set/objective and different reports | `NOT_DUPLICATE_DIFFERENT_OBJECTIVE` | no invalidation |
| failure mining vs routing design gate | routing result records `regenerated_lora_for_oracle: true`; same official checkpoint/dataset, 100 steps, 200-sample diagnostic family; different oracle objective | `AVOIDABLE_REGENERATION` | no invalidation, but artifact persistence gap |
| routing design gate vs FCAR tiny gate | FCAR result records `artifact.rank4_lora_regenerated: true`; prediction baseline was regenerated because reusable prediction artifact was missing | `AVOIDABLE_REGENERATION` | no invalidation, but avoidable compute |
| FCAR tiny gate vs robust baseline sweep | robust sweep reads `reports/fcar_prediction_artifact.json`, no model/GPU/training | `NECESSARY_REPRODUCTION` | no invalidation |
| robust sweep vs stable protocol | stable protocol creates a new fixed manifest/metric protocol after instability diagnosis | `NOT_DUPLICATE_DIFFERENT_OBJECTIVE` | supersedes small split evidence |
| stable artifact seed 0 vs LoRA seeds 11/22/33 | same fixed manifest/checkpoint/dataset/steps, different independent seeds and explicit seed robustness objective | `NECESSARY_REPRODUCTION` | strengthens evidence |
| seed 11 vs seed 22 vs seed 33 | same config except seed and output artifact; required independent reproduction | `NECESSARY_REPRODUCTION` | strengthens evidence |

## Checked Comparison Keys

- seed: stable artifact uses seed `0`; seed repro uses `11`, `22`, `33`
- train split: fixed protocol uses `reports/official_smolvla_split_manifest.json`
- val/test split: fixed protocol has `400` val frames and `1200` test frames
- LoRA rank: rank `4` in all final baseline training runs
- step count: `100` steps in scaleup/failure/routing/stable/seed runs
- checkpoint path: `C:\assets\checkpoints\smolvla_libero`
- dataset path: `C:\assets\datasets\lerobot_libero`
- artifact contents: stable and seed artifacts have different SHA256 hashes and seed metadata
- command/config hashes: runner scripts `242` through `249` have distinct SHA256 hashes recorded in `reports/official_smolvla_execution_ledger.json`

## Avoidable Regenerations

1. `4e19ffa` routing design gate regenerated rank-4 LoRA predictions for oracle analysis after failure mining. It was not an exact duplicate because the oracle objective differed, but it would have been avoidable if the previous per-frame prediction artifact had been preserved.
2. `18b3e4b` FCAR tiny gate regenerated the fixed rank-4 LoRA prediction baseline because the earlier diagnostic prediction artifact was not available in reusable form.

## Exact Duplicate Finding

`EXACT_DUPLICATE_CONFIRMED`: none.

`POSSIBLE_EXACT_DUPLICATE`: none with sufficient command/config/artifact evidence.

## Decision Impact

The avoidable regenerations do not invalidate the final stable offline result. They do show a protocol gap: future runs should persist reusable prediction artifacts and, when applicable, adapter checkpoints before follow-on analyses.
