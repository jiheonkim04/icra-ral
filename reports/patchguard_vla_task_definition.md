# PatchGuard-VLA Task Definition

Date: 2026-07-09 KST

Long title: Kinematic-Consistent Defense against Physical Patch Attacks on Vision-Language-Action Robot Policies

## Core Claim Under Test

PatchGuard-VLA is valid only if the contribution is a VLA defense against physical patch-induced visual-proprioceptive hijacking. LoRA, QLoRA, or any adapter is only a low-resource adaptation tool.

The method is not allowed to become "generic patch augmentation plus LoRA." It must use robot kinematics/proprioception inside the action pathway to suppress patch-induced phantom embodiment while preserving clean task behavior.

## Hypothesis

Physical adversarial patches can perturb VLA action prediction by corrupting the model's visual self-localization or embodiment grounding. A lightweight action-path defense should compare visual evidence against non-leaking robot state, such as EEF pose, joint state, gripper state, or an approximate arm mask, and penalize action updates that are inconsistent with the robot's actual kinematic state.

## Method Sketch

Preferred local path:

1. Use local SmolVLA as the first real executable VLA.
2. Use local LIBERO HDF5 observations before any simulator rollout or training.
3. Create clean, random-patch, fixed-visible-patch, cutout, and cheap generic visual augmentation proxy observations.
4. Measure clean-vs-patched action divergence with the same proprioceptive state.
5. Treat PatchGuard as feasible only if a non-leaking visual-proprioceptive signal is available and simple erasing/cutout baselines do not remove the attack.
6. Defer any LoRA/adapter training until STATE 1 proves the attack/signal gate.

## Non-Leakage Rule

Allowed signals:

- observation images,
- robot EEF pose, joint state, gripper state, or state vector exposed at inference,
- local policy action predictions,
- image-space arm masks only if derived from observation/simulator rendering without success labels.

Forbidden signals:

- success/failure labels,
- BDDL target labels as inference hints,
- demonstration filenames as target labels,
- ground-truth object poses used as eval-only supervision,
- simulator privileged state unless separately labeled as an oracle upper bound,
- paper claims from offline proxy evidence.

## Valid Novelty

PatchGuard-VLA remains novel only if it shows all of the following:

- it uses robot kinematics or proprioception,
- it suppresses patch-induced phantom embodiment rather than only training on more image corruptions,
- it preserves clean task success or clean action quality,
- it improves attacked success or attacked action quality more than generic patch augmentation, random erasing, or cutout.

## Current Boundary

This run is STATE 0-1 only. It may run bounded CPU offline SmolVLA inference on local assets. It may not train, run a full benchmark, download large assets, execute OpenVLA-OFT, optimize a full adversarial patch, or make paper claims.

