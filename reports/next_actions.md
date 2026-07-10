# Next Actions

Date: 2026-07-10 KST

Current decision: `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`

## Immediate Rule

Do not run official rollout.

The required seed LoRA checkpoint bundles now exist and reload from disk, but regenerated offline metrics did not reproduce the prior seed results within the predeclared tolerance.

## Evidence To Preserve

- `reports/official_smolvla_lora_checkpoint_regen_plan.md`
- `reports/official_smolvla_lora_checkpoint_regen_result.md`
- `reports/official_smolvla_lora_checkpoint_regen_result.json`
- `reports/official_smolvla_lora_checkpoint_manifest.json`
- `reports/official_smolvla_lora_checkpoint_verification.md`
- `reports/official_smolvla_lora_reproduction_comparison.md`
- `reports/official_smolvla_lora_checkpoint_regen_decision.md`
- `reports/official_smolvla_seed_11_prediction_artifact.json`
- `reports/official_smolvla_seed_22_prediction_artifact.json`
- `reports/official_smolvla_seed_33_prediction_artifact.json`
- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11`
- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22`
- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33`

## Required Before Rollout

1. Diagnose the configuration drift between the prior in-memory seed reproduction and the disk-reloaded checkpoint regeneration.
2. Specifically inspect seed `11` and seed `33`, where frozen per-seed tolerance failed.
3. Confirm whether the drift is caused by PEFT wrapper assignment, adapter save/reload semantics, eval-loss-disabled reporting, CUDA nondeterminism, data ordering, or another configuration difference.
4. Do not modify historical result tables; compare old and regenerated results side by side.
5. Do not rerun regeneration unless a new objective explicitly authorizes a drift diagnosis or fixed rerun.

## Still Forbidden

- no closed-loop rollout
- no simulator dependency installation
- no OpenVLA-OFT
- no FCAR revival
- no method design
- no full benchmark
- no asset downloads
- no static-alpha tuning on test
- no favorable-seed-only reporting

## Exact Next Step

No rollout command is safe. The next valid objective is a bounded configuration-drift diagnosis using the saved checkpoint bundles and side-by-side artifacts.
