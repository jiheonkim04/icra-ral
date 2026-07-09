# FCAR First Experiment Plan

Date: 2026-07-09 KST

Branch: `codex/frame-conditional-adapter-retention-plan`

Decision target: `READY_TO_IMPLEMENT_FCAR_TINY_GATE`

## Boundary

This is a plan/specification gate only.

- experiments happened: no
- method training happened: no
- GPU/download/OpenVLA-OFT happened: no / no / no
- simulator rollout/full benchmark happened: no
- old custom `LIBERO_7D` route used: no
- paper contribution claimed: no

## Problem Statement

Working title: Frame-Conditional Adapter Retention (FCAR).

Official SmolVLA-LIBERO low-data adaptation shows adapter interference: a standard rank-4 LoRA adapter helps some frames but hurts others, and is worse than frozen/base on aggregate. Task-level routing has almost no oracle headroom, so the routing decision must be frame/state/action-dependent rather than only task/instruction-dependent.

Current evidence:

- frozen/base action L2: `0.106514960`
- standard rank-4 LoRA action L2: `0.118024259`
- mean-action prior action L2: `1.144859722`
- frame oracle action L2: `0.084582188`
- task oracle action L2: `0.106079976`
- frame oracle gain over frozen/base: `0.021932772`, `20.59%`
- task oracle gain over frozen/base: `0.000434984`, `0.41%`

## Why Each Prior Route Is Insufficient

Frozen/base must remain an expert because it is the strongest realistic aggregate policy in the current official-data evidence.

Standard LoRA is insufficient because it adapts all frames uniformly and is worse than frozen/base on aggregate.

Task routing is insufficient because task oracle improves action L2 by only `0.000434984`, below the fixed `0.005` absolute / `5%` relative headroom gate.

Frame-level routing has headroom because a frame oracle reaches action L2 `0.084582188`, recovering `0.021932772` over frozen/base.

FCAR is not MoIRA because MoIRA-style routing is text/task-level expert assignment. FCAR must use frame/state/action-disagreement signals and must include frozen/base retention.

FCAR is not AAC because AAC adapts action chunking at inference time; FCAR chooses between frozen/base and adapted experts.

FCAR is not adapter soup because adapter soup/static merge has no frame-dependent gate.

## First Experiment

Train a tiny gate only, not the VLA backbone.

Inputs:

- official frozen/base predictions;
- official rank-4 LoRA predictions;
- current 8D state;
- base action;
- LoRA action;
- base-vs-LoRA action disagreement;
- action norms;
- gripper value/disagreement;
- normalized episode phase if available from official frame index only;
- instruction embedding only as secondary context.

Output:

- scalar `alpha in [0, 1]`
- mixed action: `a_mix = alpha * a_lora + (1 - alpha) * a_base`

Training labels:

- primary: direct regression/mixing against official ground-truth 7D action;
- auxiliary: oracle class target `1` if LoRA action L2 is lower than frozen/base action L2, else `0`;
- retention: penalize LoRA selection when base is better or when disagreement is high and no uncertainty evidence supports LoRA.

Scale:

- 200 to 1000 official held-out frames;
- deterministic train/eval split;
- CPU acceptable, GPU optional;
- wall clock under 30 minutes;
- no simulator rollout;
- no VLA retraining.

## Success Gate

FCAR must beat frozen/base by at least one of:

- `5%` relative action L2 improvement;
- `0.005` absolute action L2 improvement.

It must also beat:

- standard rank-4 LoRA;
- MoIRA-style task/instruction router;
- adapter soup/static merge;
- mean-action prior.

Soft target:

- recover at least `30%` of frame-oracle gain: action L2 about `0.0999` or lower;
- `50%` recovery target: action L2 about `0.0955` or lower.

## No-Go Conditions

Stop if:

- saved or regenerated base/LoRA per-frame predictions are missing or inconsistent;
- FCAR cannot beat frozen/base;
- FCAR only matches task/instruction routing;
- adapter soup/static merge matches FCAR;
- the gate depends on ground-truth actions, rewards, future frames, or custom metadata at inference time.
