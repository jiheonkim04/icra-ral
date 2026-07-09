# Next Real LoRA Strategy

Date: 2026-07-09 KST

## Decision

Local proxy idea generation should remain stopped.

The real SmolVLA LoRA path is available, but the first standard LoRA baseline is mean-action dominated. The next valid research step is still not a new method.

## Current Boundary

PatchGuard-VLA is archived as `KILL_BASELINE_DOMINATED`. The reusable result is infrastructure:

- PEFT works.
- bitsandbytes works.
- CUDA on RTX 5080 works.
- Local SmolVLA LoRA injection works.
- A tiny batch-size-1 rank-4 training smoke runs within about 2.2 GB peak VRAM.

This does not create a new contribution. It creates the missing baseline capability.

## Baseline Result

STATE 1 standard LoRA baseline result:

- decision: `KILL_MEAN_BASELINE_DOMINATED`
- loss start/end: `0.06359 / 0.008743`
- LoRA learned loss: yes
- mean-action eval action L2: `0.486561`
- frozen/base eval action L2: `1.6029`
- standard LoRA eval action L2: `0.940196`
- LoRA beat frozen/base: yes
- LoRA beat mean-action: no

## Next Valid Step

Do not start a method. First diagnose why standard LoRA loses to mean-action on the local held-out split.

The next baseline-only questions are:

- Is the action interface or normalization mismatched?
- Is the train/eval split too small or too distribution-shifted?
- Does the official SmolVLA training recipe require different sampling, action normalization, or target construction?
- Can standard LoRA beat mean-action on a credible official or standard split without method additions?
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

## Diagnosis Update

The follow-up diagnosis selected `ACTION_INTERFACE_BUG`.

The next valid step is action-interface repair, not a new method:

- resolve the 6D SmolVLA action head versus 7D LIBERO action label mismatch,
- resolve SO100-style checkpoint normalization versus local LIBERO action scale,
- remove or justify the hard-coded gripper-close adapter,
- rerun one-sample and one-demo overfit before any larger baseline.

No new method should start until standard LoRA baseline behavior beats mean-action after the action-interface blocker is resolved. A future method must be predeclared against standard LoRA, generic augmentation, and the relevant simple baselines before implementation.

## Interface Fix Update

The follow-up interface-fix run selected `READY_FOR_REAL_METHOD_AFTER_INTERFACE_FIX`.

What changed:

- LIBERO labels now remain `7D`.
- The native SmolVLA `6D` SO100 action schema is preserved separately.
- LIBERO labels use train-split-only 7D normalization.
- The gripper dimension is learned rather than hard-coded.
- One-sample and one-demo overfit passed.
- The fixed 7D adapter beat mean-action and frozen/base on action L2.

This still does not authorize a new method claim. The next valid step is standard fixed-interface SmolVLA/LIBERO 7D baseline reproduction on an official or standard split, with mean-action, ridge/MLP, frozen/base, and fixed-interface adapter baselines preserved.
