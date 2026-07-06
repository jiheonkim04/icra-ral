# CSS-Shield Task Definition

Project name: Counterfactual Semantic Safety Shield for VLA Manipulation.

Short name: CSS-Shield.

## Decision Boundary

The previous low-compute Target-Prior TCA-Map RA-L route is killed for RA-L-stable submission. CSS-Shield is a new rollout-first project. It reuses infrastructure and negative evidence, but it does not continue the old route as the main claim.

## Core Claim Target

A lightweight runtime counterfactual semantic safety shield can reduce wrong-target and unsafe VLA actions while preserving useful task behavior.

## Runtime Inputs

- natural-language instruction,
- proposed action from a VLA or diagnostic proposal source,
- object/scene/state proxies when available,
- candidate target and distractor object names when available,
- basic action scale, gripper, translation, and rotation statistics.

## Runtime Outputs

The shield may:

- accept,
- reject,
- damp,
- redirect,
- safe-stop,
- trigger a recovery action.

## Framing

Frame this as runtime semantic/safety intervention for VLA manipulation.

Do not frame it as a new VLA backbone, full VLA retraining, offline-only benchmark paper, or action-head training paper.

