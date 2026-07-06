# ExecSpec-Repair Task Definition

Short name: ExecSpec-Repair.

Long title: Executable Policy Repair for Vision-Language-Action Robots under Action-Space Metadata and Controller Mismatch.

## Decision Boundary

Target-Prior TCA-Map and CSS-Shield are archived negative evidence. This project does not continue either route as the main claim.

## Core Claim Target

VLA checkpoints are executable policies only when paired with action metadata, unnormalizers, gripper conventions, controller interfaces, action dimensions, and robot-specific execution conventions. ExecSpec-Repair tests whether those executable components mismatch and whether a minimal action-space calibration layer can recover action/replay behavior.

## Initial Evidence Target

STATE 1 must produce a concrete mismatch metric from local LIBERO HDF5 expert actions or exact-init replay. Offline supervised calibration is allowed only when labeled as calibration/evaluation, not as rollout policy action generation.
