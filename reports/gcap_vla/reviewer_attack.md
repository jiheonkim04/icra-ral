# GCAP-VLA Reviewer Attack

Date: 2026-07-12 KST

Reviewer B position: approve only with strict kill gates.

## Direct Prior Risk

VLA-IAP already emphasizes interaction-first preservation of geometric continuity and Sobel-style structural anchors for VLA manipulation. AffordVLA and PALM also argue for affordance- or interaction-centric visual representations. GCAP is not a representation-training contribution unless the closed-loop result shows that a cheap inference-time temporal-geometric repair creates a distinct and useful deployment capability.

## Simple Equivalent Methods

The method can be explained away by any of these:

- full-frame hold-last replacement,
- Sobel edge enhancement without temporal patch memory,
- unmodified occluded frozen policy if occlusion does not actually hurt success,
- clean frozen policy if GCAP harms unoccluded execution.

## Required Kill Gates

- If `full_frame_hold_last` matches or beats `gcap_full`, kill as `SIMPLE_TEMPORAL_BASELINE_EXPLAINS_METHOD`.
- If `gcap_no_temporal_ablation` matches or beats `gcap_full`, kill as `TEMPORAL_COMPONENT_NOT_USEFUL`.
- If `gcap_full` does not beat `occluded_frozen_smolvla`, kill as `NO_OCCLUSION_ROBUSTNESS_GAIN`.
- If clean GCAP drops by more than 2 absolute points versus clean frozen, kill as `CLEAN_RETENTION_FAILURE`.
- If there are rollout exceptions, classify as measurement invalid and allow at most one narrow repair.

## Protocol Constraints

Do not tune occlusion boxes after seeing Stage A outcomes. Do not reset the SmolVLA policy every step. Do not use simulator state, object masks, object poses, success, reward, or future observations. Do not call this a paper-ready method unless it survives this prototype and then generalizes to a second backbone and second condition.
