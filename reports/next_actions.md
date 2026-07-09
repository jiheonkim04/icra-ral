# Next Actions

Date: 2026-07-10 KST

Current decision: `AUDIT_FOUND_PROTOCOL_GAPS_FIX_BEFORE_ROLLOUT`

## Immediate Next Action

Before official rollout, create a no-experiment protocol-fix branch that records Hugging Face model/dataset revision pins, enforces the future baseline naming glossary, and decides whether LoRA adapter checkpoints must be persisted alongside prediction artifacts.

## Do Not Do In The Protocol-Fix Step

- do not run experiments
- do not train
- do not use GPU
- do not download assets unless separately approved
- do not run simulator rollout
- do not run OpenVLA-OFT
- do not revive FCAR
- do not rerun LoRA seeds
- do not modify historical metrics
- do not delete or overwrite existing artifacts

## Required Fixes Before Official Rollout

1. Record source revision pins for the official SmolVLA checkpoint and LeRobot LIBERO dataset, or explicitly document why only local path/file-hash pinning is available.
2. Add the future naming glossary:
   - `task_or_instruction_router_proxy`
   - `validation_selected_action_space_static_mix`
   - `frame_oracle_upper_bound`
   - `task_oracle_upper_bound`
3. Decide whether seed-specific LoRA adapter checkpoints are required for future reproducibility.
4. Keep `reports/official_smolvla_split_manifest.json` and `reports/official_smolvla_metric_protocol.md` fixed unless a future audit explicitly supersedes them.
5. Preserve the current prediction artifacts and their hashes.

## Evidence To Preserve

- `reports/official_smolvla_stable_prediction_artifact.json`
- `reports/official_smolvla_lora_seed_11_prediction_artifact.json`
- `reports/official_smolvla_lora_seed_22_prediction_artifact.json`
- `reports/official_smolvla_lora_seed_33_prediction_artifact.json`
- `reports/official_smolvla_execution_ledger.json`
- `reports/official_smolvla_artifact_integrity_audit.md`

## Current Offline Baseline

Use this name going forward:

`validation_selected_action_space_static_mix`

It remains the strongest realistic offline baseline under the fixed protocol. It is not official rollout evidence.

## Exact Next Step

Create the protocol-fix branch and reports only; do not run model code. After that branch is clean, the next major scientific milestone can be an explicitly approved official LIBERO closed-loop rollout readiness gate.
