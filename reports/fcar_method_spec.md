# FCAR Method Spec

Date: 2026-07-09 KST

## Method Name

Frame-Conditional Adapter Retention (FCAR)

## Core Claim To Test

Low-data LoRA adaptation creates frame-level negative transfer. A small gate can preserve frozen/base behavior on LoRA-hurt frames while using LoRA on frames where the adapter helps.

This is only a testable hypothesis, not a paper claim.

## Components

1. Frozen/base SmolVLA expert.
2. Standard rank-4 LoRA SmolVLA expert.
3. Tiny frame-level gate.
4. Retention regularizer.
5. Official SmolVLA pre/postprocessor and official LeRobot LIBERO dataset only.

## Gate Inputs Allowed

- base action;
- LoRA action;
- base-vs-LoRA action disagreement;
- base/LoRA action norms;
- gripper value and gripper disagreement;
- current official 8D state;
- normalized episode phase from official frame index;
- cheap image/state embeddings if extracted from official model outputs;
- instruction embedding as secondary context only.

## Gate Inputs Rejected

- ground-truth action at inference;
- rollout reward;
- success label;
- future frames;
- custom `LIBERO_7D` metadata;
- oracle task category not available in a fair official setting;
- simulator state beyond official observations.

## Gate Output

`alpha in [0, 1]`

Action:

```text
a_mix = alpha * a_lora + (1 - alpha) * a_base
```

For the first experiment, the gate may be:

- supervised from oracle labels;
- trained as direct action regression through the mixture;
- calibrated as a helpful/harmful classifier;
- or a hybrid of classifier plus regression.

The preferred first implementation is hybrid:

```text
loss = L2(a_mix, a_gt)
     + lambda_retention * CE(help_label, alpha)
     + lambda_sparse * min(alpha, 1 - alpha)
```

## Retention Principle

The frozen/base expert is the default. The gate must earn LoRA usage per frame.

This differs from standard LoRA, which applies the adapter uniformly, and from task routing, which cannot switch within a task when LoRA is helpful on some frames and harmful on others.

## Expected Failure Modes

- gate learns to always choose base;
- gate overfits oracle labels and fails held-out frames;
- disagreement features are insufficient;
- adapter soup/static merge matches the gate;
- MoIRA-style instruction router matches task-level behavior;
- offline action L2 does not transfer to rollout success.
