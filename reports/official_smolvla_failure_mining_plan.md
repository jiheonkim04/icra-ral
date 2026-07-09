# Official SmolVLA-LIBERO Failure Mining Plan

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-failure-mining`

## Objective

Mine structured failures in the official SmolVLA-LIBERO path before any method implementation.

This is not a method run, not paper novelty, not OpenVLA-OFT, not a simulator rollout, and not a full benchmark.

## Fixed Inputs

- model: `C:\assets\checkpoints\smolvla_libero`
- dataset: `C:\assets\datasets\lerobot_libero`
- VLM dependency: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`
- processor/postprocessor: official checkpoint processor files
- video backend: `pyav`
- training variant: standard rank-4 LoRA only
- training budget: same as previous official baseline scaleup, `100` steps, batch size `1`

No additional large assets may be downloaded.

## Risk Assessment

- task: bounded official failure mining and method-readiness gate
- source: already downloaded official Hugging Face assets only
- new download expected bytes: `0`
- target paths: existing approved local roots under `C:\assets`
- expected runtime: preferred under `1` hour, hard cap `2` hours
- training runtime: no longer than previous official rank-4 LoRA scaleup
- expected VRAM: below repo budget of `14 GiB`
- simulator/full benchmark/OpenVLA-OFT: not allowed
- method implementation: not allowed
- decision: proceed with task-local `ALLOW_HEAVY_IMPORT=1` and `ALLOW_GPU_TRAINING=1` gates because this is one bounded official-data milestone.

## STATE 1 Metric Reconciliation

The report must explain:

- official `forward()` loss is a flow/action-chunk training objective over normalized action chunks;
- postprocessed action L2 is a one-step raw action-space proxy after official unnormalization;
- therefore action L2 can improve while chunk flow eval loss worsens;
- both metrics must be reported until simulator/eval readiness provides task success.

Predeclared recommendation:

- primary offline gate: postprocessed held-out action L2, with translation/rotation/gripper breakdown and trivial-prior comparison;
- secondary stability gate: normalized chunk flow eval loss;
- warning condition: LoRA action L2 improves while eval loss worsens.

## STATE 2 Failure Mining Scope

Evaluate:

1. frozen/base official SmolVLA, no training;
2. standard rank-4 LoRA trained for `100` steps on the official training episode used in the previous scaleup;
3. global mean-action prior from official action stats.

Deterministic held-out selection:

- exclude the LoRA training episode;
- choose multiple task groups from official dataset metadata/data parquet;
- target `200` held-out frames if runtime stays cheap;
- include phase buckets from normalized episode frame position: early, mid, late;
- no simulator or task-success claim.

Metrics:

- action L2;
- normalized chunk eval loss for model variants;
- translation L2;
- rotation L2;
- gripper absolute error and sign accuracy;
- per-task breakdown;
- per-action-dimension absolute error;
- phase/time-index breakdown;
- train/eval gap;
- action range validity;
- LoRA helps/hurts examples.

## STATE 3 Gap Rejection Rules

Reject a gap if it is:

- too small or noisy;
- solved by frozen/base;
- solved by standard LoRA;
- explained by mean-action/trivial prior;
- not measurable with official data;
- not connected to action quality;
- likely already covered by recent VLA work.

## Recent-Paper Checks

Use these only as method-readiness context, not as experimental evidence:

- SmolVLA docs/paper: official lightweight VLA, LeRobot fine-tuning, action chunks.
- Real-Time Chunking: action chunk consistency can be handled at inference time without training-time changes.
- Adaptive Action Chunking: inference-time chunk adaptation addresses fixed chunk-length consistency/reactivity tradeoffs.
- MoIRA: task/instruction routing with low-rank adapters on LIBERO-like benchmarks is already a close routing baseline.

## STATE 5 Decision Set

Final decision must be exactly one of:

- `GO_METHOD_DESIGN_GRIPPER_PHASE`
- `GO_METHOD_DESIGN_CONTROL_STABILITY`
- `GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`
- `NEED_LONGER_OFFICIAL_BASELINE_REPRO`
- `NO_METHOD_WORTHY_GAP`
- `METRIC_CONFLICT_BLOCKS_METHOD`
