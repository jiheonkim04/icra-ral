# Official SmolVLA-LIBERO Baseline Scaleup Plan

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-baseline-scaleup`

## Objective

Run a bounded official SmolVLA-LIBERO rank-4 LoRA baseline on the downloaded official assets and decide whether this official baseline is stable enough for future RA-L method work.

This is baseline reproduction only. It is not a new method, paper claim, full benchmark, simulator rollout, or OpenVLA-OFT run.

## Fixed Inputs

- checkpoint: `C:\assets\checkpoints\smolvla_libero`
- dataset: `C:\assets\datasets\lerobot_libero`
- VLM dependency: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`
- HF cache: `C:\assets\hf_home`
- video backend: `pyav`
- official dataset episode for training: episode `0`
- offline mini-holdout: episode `1` if present, otherwise episode `0`

No additional large assets are allowed.

## Risk Assessment

- task: bounded official SmolVLA-LIBERO LoRA baseline scaleup
- source: already downloaded official Hugging Face assets only
- new download expected bytes: `0`
- target paths: approved local asset roots under `C:\assets`
- expected runtime: preferred under `30` minutes, hard cap `45` minutes
- expected VRAM: below `14 GiB`
- hardware: local RTX 5080 CUDA path
- batch size: `1`
- LoRA rank: `4`
- planned steps: `100`
- simulator rollout/full benchmark/OpenVLA-OFT: not allowed
- decision: proceed with task-local `ALLOW_HEAVY_IMPORT=1` and `ALLOW_GPU_TRAINING=1` gates because the run is inside the repo GPU/training budget.

## STATE 1 Dataset/Sample Audit

The runner must report:

1. available episodes/tasks
2. official split metadata
3. sample counts
4. action dimension and stats range
5. state dimension and stats range
6. image streams
7. processor/preprocessor output shapes and devices
8. labels/action stats loading
9. deterministic repeated sample loading
10. schema mismatch signs

## STATE 2 Baseline Scaleup

Required variants:

1. frozen/base official SmolVLA with no training
2. standard rank-4 LoRA baseline

Optional rank-8 is explicitly skipped in this milestone to avoid adding variants before the rank-4 official baseline is understood.

Training limits:

- rank: `4`
- batch size: `1`
- requested steps: `100`
- max local cap: `200`
- optimizer: `AdamW`
- learning rate: `1e-4`
- no old custom `LIBERO_7D` adapter route
- no full benchmark
- no simulator rollout

Required logs:

- model parameter device
- input tensor devices
- CUDA availability and device name
- autocast/fp16/bf16 status
- loss curve
- nonzero-gradient check and gradient norm
- trainable parameter count
- VRAM peak
- runtime and steps/sec
- action output shape/range/finite checks
- CPU fallback status

## STATE 3 Decision Rule

The final decision must be exactly one of:

- `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA`
- `READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO`
- `NEEDS_OFFICIAL_EVAL_OR_ROLLOUT_SETUP`
- `CPU_FALLBACK_BUG`
- `ACTION_OR_SCHEMA_MISMATCH`
- `TOO_HEAVY_LOCAL`
- `TRAINING_UNSTABLE`

Predeclared conservative interpretation:

- Use `CPU_FALLBACK_BUG` immediately if CUDA is available but model or training inputs are on CPU.
- Use `ACTION_OR_SCHEMA_MISMATCH` if official 8D state / 7D action loading or processor outputs disagree.
- Use `TOO_HEAVY_LOCAL` on OOM, runtime over hard cap, or VRAM over budget.
- Use `TRAINING_UNSTABLE` on NaN/Inf, invalid gradients, or failed training.
- Use `NEEDS_OFFICIAL_EVAL_OR_ROLLOUT_SETUP` if training works but no held-out loss/action metrics are available.
- Use `READY_FOR_LONGER_OFFICIAL_BASELINE_REPRO` if rank-4 LoRA works but loss decrease is small, the run is too small, or mini-holdout action metrics get clearly worse than frozen/base.
- Use `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA` only if rank-4 LoRA trains on CUDA without OOM, loss decreases by at least 10%, label/action metrics are available, VRAM/runtime are acceptable, no CPU fallback occurs, and mini-holdout action L2 is not more than 5% worse than frozen/base.

No method design may start unless the final decision is `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA`.
