# Next Actions

Date: 2026-07-10 KST

Current decision: `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

## Immediate Rule

Do not run official rollout yet.

The protocol is now defined, but execution remains blocked by missing official LoRA adapter checkpoint bundles and missing official LIBERO eval dependencies.

## Required Before Rollout

1. Create a future explicitly approved training/regeneration pass that writes immutable adapter checkpoint bundles for seeds `11`, `22`, and `33`.
2. Validate each adapter bundle contains `adapter_config.json`, `adapter_model.safetensors`, `training_manifest.json`, official pre/postprocessor references, a source lock, and SHA256 manifest.
3. Fix the official LIBERO eval environment by installing/validating `libero` and `robosuite`, preferably under WSL/Linux unless native Windows support is explicitly proven.
4. Re-run only safe readiness checks first: import/help/source checks, not simulator rollout.
5. Execute Stage A bounded readiness pilot only after adapter bundles and official eval dependencies are ready.

## Do Not Do Until Explicitly Approved

- do not train or regenerate LoRA adapters
- do not use GPU
- do not run simulator rollout
- do not download assets
- do not run OpenVLA-OFT
- do not revive FCAR
- do not modify historical metrics
- do not call action-space interpolation adapter soup
- do not call the local task/instruction proxy official MoIRA

## Evidence To Preserve

- `configs/official_smolvla_repro_lock.yaml`
- `reports/official_smolvla_revision_lock.md`
- `reports/official_smolvla_baseline_naming_policy.md`
- `reports/official_smolvla_lora_checkpoint_policy.md`
- `reports/official_smolvla_rollout_action_semantics.md`
- `reports/official_smolvla_rollout_protocol.md`
- `reports/official_smolvla_rollout_readiness.md`
- `reports/official_smolvla_protocol_fix_decision.md`
- `reports/official_smolvla_split_manifest.json`
- `reports/official_smolvla_metric_protocol.md`
- `reports/official_smolvla_stable_prediction_artifact.json`
- `reports/official_smolvla_lora_seed_11_prediction_artifact.json`
- `reports/official_smolvla_lora_seed_22_prediction_artifact.json`
- `reports/official_smolvla_lora_seed_33_prediction_artifact.json`

## Exact Next Command

No next training, regeneration, GPU, download, or rollout command is safe under the current no-experiment boundary.

The future required adapter-regeneration command is documented in `reports/official_smolvla_lora_checkpoint_policy.md`, but it must not be run until a new objective explicitly authorizes training/regeneration.
