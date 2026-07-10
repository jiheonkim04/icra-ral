# Official SmolVLA LoRA Checkpoint Regeneration Plan

Date: 2026-07-10 KST

Status: `PRETRAINING_EXPECTATIONS_FROZEN`

Current main commit before branch: `78bf94a Fix official SmolVLA rollout protocol`

Branch: `codex/regenerate-official-smolvla-lora-checkpoints`

Objective: regenerate, persist, identify, and verify immutable official SmolVLA-LIBERO rank-4 LoRA adapter checkpoint bundles for seeds `11`, `22`, and `33`.

This is necessary checkpoint regeneration and reproducibility verification. It is not a new baseline search, not method development, not FCAR, not OpenVLA-OFT, and not rollout.

## Frozen Source Inputs

- repro lock: `configs/official_smolvla_repro_lock.yaml`
  - SHA256: `C603D36B02E59505C2699F3BEA15C64807468B188E862C0E33F275DE04F351A5`
- model: `lerobot/smolvla_libero`
  - revision: `31d453f7edd78c839a8bbc39744a292686daf0de`
  - local path: `C:\assets\checkpoints\smolvla_libero`
- dataset: `lerobot/libero`
  - revision: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
  - local path: `C:\assets\datasets\lerobot_libero`
- split manifest: `reports/official_smolvla_split_manifest.json`
  - SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- metric protocol: `reports/official_smolvla_metric_protocol.md`
  - SHA256: `64430225940C5168B3734BB40F9F48AD02877E0BA04DC804367AFBB214AE486E`
- prior seed result JSON: `reports/official_smolvla_lora_seed_repro_result.json`
  - SHA256: `BAA9BD61DA4631F8CF7020198147A52F66435DBFCDDF02717BE2188CC8E79505`

These inputs must not be changed during regeneration.

## Frozen Training Configuration

- LoRA rank: `4`
- PEFT method: `LORA`
- target modules:
  - `model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj`
  - `model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)`
- previous trainable parameter count: `185664`
- previous total parameter count: `450231840`
- step count: `100`
- batch size: `1`
- learning rate: `2e-4`
- optimizer: `torch.optim.AdamW`
- scheduler: none
- precision/autocast: no autocast; model parameter dtype observed as `torch.bfloat16`
- gradient settings: `optimizer.zero_grad(set_to_none=True)`, backprop through LoRA trainable parameters only
- train split: fixed manifest `train`, `1200` frames
- validation split: fixed manifest `val`, `400` frames
- test split: fixed manifest `test`, `1200` frames
- frame sampling: seed-specific NumPy permutation over train split, one manifest frame per training step
- chunk size: `50`
- action dim: `7`
- video backend: `pyav`
- CUDA required: yes
- CPU fallback permitted: no

Each seed must set/log Python, NumPy, PyTorch CPU, PyTorch CUDA, and data-order RNG state where available. Determinism settings are logged without changing the historical training behavior merely to force determinism.

## Required Output Roots

- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11`
- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22`
- `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33`

Pretraining inventory: all three target directories were absent before the first regeneration attempt, so no existing complete checkpoint was overwritten.

## Required Bundle Contents

- `adapter_model.safetensors` or official PEFT equivalent
- `adapter_config.json`
- `training_manifest.json`
- `eval_preprocessor_postprocessor_refs.json`
- `source_repro_lock.yaml`
- `sha256_manifest.json`

Optimizer, scheduler, trainer, and RNG states should be saved when supported without changing the established recipe.

## Frozen Prior Per-Seed Expectations

Primary comparison metric: raw 7D action L2 on the fixed protocol.

| seed | frozen_base | rank4_lora | validation_selected_action_space_static_mix | frame_oracle_upper_bound | task_oracle_upper_bound | task_or_instruction_router_proxy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.085558433 | 0.084128699 | 0.077354597 | 0.066234143 | 0.078372683 | 0.085719423 |
| 22 | 0.085558433 | 0.090162398 | 0.080789904 | 0.070815707 | 0.082495298 | 0.089507871 |
| 33 | 0.085558433 | 0.090426934 | 0.083704791 | 0.070301761 | 0.082546947 | 0.089208622 |

Prior aggregate:

- frozen/base action L2: `0.085558433`
- rank-4 LoRA mean/std: `0.088239344` / `0.002908670`
- validation-selected action-space static mix mean/std: `0.080616431` / `0.002595356`
- frame oracle upper-bound mean/std: `0.069117204` / `0.002049401`
- task oracle upper-bound mean/std: `0.081138309` / `0.001955707`
- task/instruction router proxy mean/std: `0.088145305` / `0.001719703`
- static mix realistic seed win count: `3/3`
- realistic task win counts summed across seeds: static mix `93`, frozen/base `20`, rank-4 LoRA `7`

## Frozen Reproduction Tolerance

The tolerance was declared before regenerated results were observed.

- Per-seed rank-4 LoRA raw action L2 absolute difference must be `<= 0.002`.
- Per-seed validation-selected action-space static mix raw action L2 absolute difference must be `<= 0.002`.
- Aggregate rank-4 LoRA mean absolute difference must be `<= 0.002`.
- Aggregate static mix mean absolute difference must be `<= 0.002`.
- The main qualitative conclusion must be preserved: `validation_selected_action_space_static_mix` remains stronger than standalone `rank4_lora` under the fixed protocol.
- Test labels must not be used to choose static alpha.
- Historical result tables must remain archived and must not be silently replaced.

If regenerated metrics fail this frozen tolerance, the run must stop with `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`.

## Required Decisions

The final decision must be exactly one of:

- `LORA_CHECKPOINTS_REGENERATED_AND_VERIFIED`
- `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`
- `CHECKPOINT_BUNDLE_INCOMPLETE`
- `CHECKPOINT_LOAD_FAILED`
- `CHECKPOINT_IDENTITY_UNPROVEN`
- `REVISION_LOCK_MISMATCH`
- `CPU_FALLBACK_BUG`
- `TRAINING_FAILURE`
- `TOO_HEAVY_LOCAL`

All three seeds are required. No best-seed-only acceptance is allowed.
