# SmolVLA LoRA Baseline Kill Criteria

Date: 2026-07-09 KST

## Hard Stop Labels

| Condition | Exact decision |
| --- | --- |
| PEFT/CUDA/local data/local SmolVLA cannot be used | `ENV_BLOCKED` |
| batch-size-1 rank-4 LoRA is too heavy or OOMs | `TOO_HEAVY_LOCAL` |
| LoRA injects but loss does not decrease meaningfully | `KILL_NO_REAL_LORA_LEARNING` |
| LoRA learns on train but does not beat mean-action on eval action L2 | `KILL_MEAN_BASELINE_DOMINATED` |
| LoRA learns and beats mean-action but does not beat frozen/base SmolVLA | `KILL_FROZEN_BASELINE_DOMINATED` |
| LoRA learns and beats both required baselines under the bounded split | `READY_FOR_METHOD_ON_TOP_OF_SMOLVLA_LORA` |

## Required Baseline Rule

Standard LoRA must beat both:

- mean-action baseline,
- frozen/base SmolVLA.

If either baseline dominates, no new method should be started on top of this local setup.

## Interpretation

A kill decision here does not invalidate SmolVLA or LoRA generally. It means the current local setup cannot yet support a method on top of standard LoRA.

A ready decision does not authorize a method by itself. It only authorizes baseline-preserving planning for a future method that predeclares standard LoRA and simple baselines.

## Triggered Criterion

STATE 1 triggered `KILL_MEAN_BASELINE_DOMINATED`.

The LoRA loss decreased meaningfully and standard LoRA beat frozen/base SmolVLA, but held-out eval action L2 was worse than mean-action:

- mean-action: `0.486561`
- standard LoRA: `0.940196`
- frozen/base SmolVLA: `1.6029`

This blocks method work until the standard LoRA baseline can beat the trivial action prior.
