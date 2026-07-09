# Next Real LoRA Strategy

Date: 2026-07-09 KST

## Decision

Local proxy idea generation should remain stopped.

The real SmolVLA LoRA path is now available, but the next valid research step is not a new method yet.

## Current Boundary

PatchGuard-VLA is archived as `KILL_BASELINE_DOMINATED`. The reusable result is infrastructure:

- PEFT works.
- bitsandbytes works.
- CUDA on RTX 5080 works.
- Local SmolVLA LoRA injection works.
- A tiny batch-size-1 rank-4 training smoke runs within about 2.2 GB peak VRAM.

This does not create a new contribution. It creates the missing baseline capability.

## Next Valid Step

Run a real SmolVLA LoRA baseline reproduction on an official or standard task split.

The first LoRA baseline should answer plain baseline questions before any method design:

- Can standard SmolVLA LoRA preserve clean behavior?
- Can standard SmolVLA LoRA train without collapse on a standard split?
- What is the true memory/runtime envelope beyond one sample?
- What is the baseline patched or perturbed behavior?
- Does generic adversarial augmentation already explain robustness gains?
- Which metrics are stable enough to support future method comparisons?

## Required Baselines Before Any New Method

- frozen/base SmolVLA,
- standard LoRA imitation,
- standard LoRA with generic augmentation if robustness is studied,
- cutout/random-erasing if visual patch robustness is studied,
- no-adaptation or mean-action controls where action-head metrics are used.

## Disallowed Next Work

Do not:

- start another local proxy method,
- rename PatchGuard into a new defense,
- proceed to PatchGuard STATE 2,
- run OpenVLA-OFT,
- run rollout from PatchGuard evidence,
- download large assets,
- make paper claims from the tiny smoke,
- treat LoRA itself as novelty.

## Promotion Rule

No new method should start until standard LoRA baseline behavior is understood on a standard split. A future method must be predeclared against standard LoRA, generic augmentation, and the relevant simple baselines before implementation.
