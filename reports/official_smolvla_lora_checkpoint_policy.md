# Official SmolVLA LoRA Checkpoint Policy

Date: 2026-07-10 KST

Decision: `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

This audit did not regenerate adapters or run training. It only checked whether immutable adapter checkpoint bundles already exist locally.

## Policy

Every LoRA policy used in an official rollout or final reported result must have an immutable persisted adapter checkpoint bundle. A prediction JSON is not a substitute for an adapter checkpoint because it cannot prove policy identity, training configuration, optimizer-independent adapter weights, or future rollout loadability.

Required bundle contents:

- `adapter_config.json`
- `adapter_model.safetensors`
- `training_manifest.json`
- `eval_preprocessor_postprocessor_refs.json`
- `source_repro_lock.yaml`
- `sha256_manifest.json`

The bundle must identify the base model revision, dataset revision, split manifest hash, metric protocol hash, seed, rank, LoRA target modules, training command, package versions, and adapter file hashes.

## Seed Audit

| Seed | Prediction artifact | Adapter checkpoint status | Identity proven |
| --- | --- | --- | --- |
| `11` | `reports/official_smolvla_lora_seed_11_prediction_artifact.json` | `CHECKPOINT_MISSING` | No |
| `22` | `reports/official_smolvla_lora_seed_22_prediction_artifact.json` | `CHECKPOINT_MISSING` | No |
| `33` | `reports/official_smolvla_lora_seed_33_prediction_artifact.json` | `CHECKPOINT_MISSING` | No |

Old custom-route adapter files under `runs\smolvla_7d_*` do not count for this policy. They are not the official seed-specific SmolVLA-LIBERO LoRA adapters for seeds `11`, `22`, and `33`.

## Consequence

`rank4_lora` and `validation_selected_action_space_static_mix` cannot enter official rollout or final result reporting until the required adapter bundles exist.

The current offline prediction artifacts may remain archived evidence, but they do not satisfy rollout reproducibility.

## Future Regeneration Command

The required future action is adapter checkpoint regeneration. The command below is intentionally not safe to run in this no-experiment protocol-fix pass because it implies training/regeneration and depends on a future reviewed runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\250_official_smolvla_lora_adapter_checkpoint_regen.ps1 -Seeds 11,22,33 -CheckpointPath C:\assets\checkpoints\smolvla_libero -DatasetRoot C:\assets\datasets\lerobot_libero -SplitManifest reports\official_smolvla_split_manifest.json -MetricProtocol reports\official_smolvla_metric_protocol.md -OutputRoot reports\official_smolvla_lora_checkpoints
```

Until that future run is explicitly approved and produces complete immutable bundles, the correct decision remains `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`.

## Post-Regeneration Update

Date: 2026-07-10 KST

The approved checkpoint regeneration pass later produced complete, disk-reload-verified bundles for seeds `11`, `22`, and `33`:

- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11`
- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22`
- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33`

Current checkpoint status is therefore no longer missing. The current blocker is metric reproduction drift, recorded as `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT` in:

- `reports/official_smolvla_lora_checkpoint_regen_result.json`
- `reports/official_smolvla_lora_reproduction_comparison.md`
- `reports/official_smolvla_lora_checkpoint_regen_decision.md`

The policy requirement remains unchanged: prediction JSON cannot replace adapter checkpoints, and official rollout remains blocked until the metric drift is diagnosed.
