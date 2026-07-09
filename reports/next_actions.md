# Next Actions

Date: 2026-07-10 KST

Current decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`

## Immediate Next Action

Run independent standard rank-4 LoRA baseline seeds under the fixed stable manifest. Do not design a new method yet.

Required setup:

- fixed manifest: `reports/official_smolvla_split_manifest.json`
- generated artifact: `reports/official_smolvla_stable_prediction_artifact.json`
- stable artifact result: `reports/official_smolvla_stable_artifact_eval_result.json`
- metric protocol: `reports/official_smolvla_metric_protocol.md`
- official checkpoint: `C:\assets\checkpoints\smolvla_libero`
- official dataset: `C:\assets\datasets\lerobot_libero`

## Why This Is Next

- The larger stable artifact was generated successfully with `2800` prediction records.
- The previous `NEEDS_LARGER_PREDICTION_ARTIFACT` blocker is closed.
- Frozen/base test action L2 is `0.085558433`.
- Single-seed rank-4 LoRA test action L2 is worse at `0.091230140`.
- Validation-selected static mix is strongest realistic baseline at `0.081135060`, selected on validation with alpha `0.5`.
- Frame oracle remains better at `0.068470215`, leaving `0.012664845` action-L2 headroom after static mix.
- Task oracle is no longer weak under the larger artifact: `0.079386015`, headroom `0.006172418` over frozen/base.
- Because the rank-4 LoRA artifact is still one regenerated seed, LoRA seed robustness is now the main unresolved blocker.

## Required Boundaries

- Use official assets only.
- Use the fixed manifest without changing train/val/test membership.
- Do not tune static alpha on test.
- Do not run FCAR, FCAR v2, or any routing method.
- Do not use the archived custom `LIBERO_7D` adapter route.
- Do not run OpenVLA-OFT.
- Do not run simulator rollout or full benchmark as a substitute for this offline baseline seed audit.
- Do not download additional assets unless explicitly approved.
- If CUDA is available but model parameters or tensors are on CPU, stop and report `CPU_FALLBACK_BUG`.

## Seed Audit Must Report

- seed list and fixed training budget
- rank-4 LoRA train loss before/after for each seed
- trainable parameter count
- CUDA device, input tensor devices, peak VRAM, autocast status
- frozen/base reused or regenerated status
- per-seed rank-4 LoRA test action L2
- per-seed static mix selected on validation only
- mean/std across seeds for LoRA and static mix
- whether LoRA remains worse than frozen/base
- whether static mix remains strongest realistic baseline
- whether task oracle/frame oracle conclusions remain stable

## Current Evidence To Preserve

- FCAR remains killed and must not be revived or tuned.
- Stable test realistic rank order: static mix, frozen/base, rank-4 LoRA, MoIRA-style task router, mean-action prior.
- Realistic task win counts: static mix `29`, frozen/base `7`, rank-4 LoRA `4`.
- MoIRA-style task/instruction router remains weak at action L2 `0.092209764`.
- Mean-action prior remains much worse at action L2 `1.197255124`.
- Final current decision is `NEEDS_LONGER_LORA_BASELINE_REPRO`.
