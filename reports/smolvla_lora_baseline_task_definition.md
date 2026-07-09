# SmolVLA LoRA Baseline Task Definition

Date: 2026-07-09 KST

## Route Name

SmolVLA LoRA Baseline Reproduction

## Purpose

This route is a baseline foundation, not a paper method and not novelty. Its only purpose is to verify whether standard SmolVLA LoRA can learn from local LIBERO HDF5 action data under the RTX 5080 16GB budget.

## Core Question

Can a standard PEFT LoRA adapter attached to local SmolVLA reduce action error on a deterministic held-out local LIBERO split compared with mean-action and frozen/base SmolVLA baselines?

## Non-Novelty Rule

LoRA is a tool. It is not the contribution.

No new RA-L method should be proposed until standard LoRA behavior is understood on a standard or official split.

## Allowed Evidence

- local SmolVLA load from `C:\assets\checkpoints\smolvla`,
- local LIBERO HDF5 observations and actions,
- deterministic demo-level train/eval split,
- standard PEFT LoRA on action-path modules,
- batch size 1,
- rank 4 first,
- 50 to 200 maximum optimization steps,
- action-space metrics only.

## Forbidden Evidence

- OpenVLA-OFT,
- full VLA fine-tuning,
- full benchmark,
- rollout unless separately approved and bounded,
- large downloads,
- new defense or method variant,
- PatchGuard continuation,
- paper-grade claims.

## STATE 1 Decision Set

The final decision must be exactly one of:

- `READY_FOR_METHOD_ON_TOP_OF_SMOLVLA_LORA`
- `KILL_NO_REAL_LORA_LEARNING`
- `KILL_MEAN_BASELINE_DOMINATED`
- `KILL_FROZEN_BASELINE_DOMINATED`
- `TOO_HEAVY_LOCAL`
- `ENV_BLOCKED`
