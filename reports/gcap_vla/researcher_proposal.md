# GCAP-VLA Researcher Proposal

Date: 2026-07-12 KST

Method: `GCAP-VLA` - Geometric-Continuity Anchored Perception for VLA observation corruption.

## Claim

Frozen VLA policies can fail under partial camera occlusion because the visual stream loses sparse geometric anchors near the gripper-object interaction region, even when global task semantics and proprioception remain available. GCAP-VLA repairs the policy input at the camera-tensor boundary by replacing only detected occluded patches with the previous repaired view and applying a local Sobel geometric-continuity prior around the missing region.

## Novelty Axis

GCAP-VLA changes the perception pathway, not the action pathway. It is distinct from:

- DICD-VLA: no delay-indexed chunk deployment and no action history adapter.
- FEDO-VLA: no command/realized-action feedback and no residual command compensation.
- ECHO-VLA: no candidate ranking or same-state action selection.
- adaptive chunk or replanning methods: official action queue semantics are preserved.

Closest current papers:

- AffordVLA, arXiv:2605.17517, argues that VLA visual representations over-focus on global appearance and need manipulation-centric affordance alignment.
- VLA-IAP, arXiv:2603.22991, finds that preserving geometric continuity and structural anchors is critical for manipulation under visual token pruning.
- PALM, arXiv:2601.07060, uses structured future affordance prediction for long-horizon manipulation.
- CorridorVLA, arXiv:2604.21241, couples generative action heads with sparse spatial anchors and corridor constraints.
- DreamZero, arXiv:2602.15922, motivates visually grounded dynamics/world-action modeling, but is far outside local compute.

GCAP-VLA differs by being an inference-time, training-free, image-validity and temporal-geometric repair layer for controlled camera occlusion. It does not train VLA representations or require external affordance masks.

## Mechanism

At each policy call:

1. Apply or receive an image-only occlusion mask for one or both camera streams.
2. Replace only masked pixels with the previous repaired camera tensor, preserving current unmasked evidence.
3. Apply a local Sobel edge prior on the dilated mask boundary to preserve interaction-region geometric continuity.
4. Pass the repaired tensors through the unchanged frozen SmolVLA policy.

Inputs allowed at inference:

- current camera tensors,
- previous repaired camera tensors,
- detected occlusion mask or invalid-pixel mask,
- language and proprioception already used by the policy.

Forbidden:

- simulator state,
- object poses,
- reward,
- task success,
- future observations,
- post-hoc candidate labels.

## Why It Might Work

SmolVLA already uses two camera streams. A full-frame hold-last baseline can over-freeze visual evidence and lose current unmasked context. GCAP's patchwise memory keeps current visible context while preserving the missing interaction-region structure. A no-temporal edge-only ablation tests whether the temporal component actually matters.

## Prototype Scope

First backbone: official frozen SmolVLA-LIBERO.

Targeted condition: controlled partial camera occlusion in the preprocessed camera tensors.

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Evaluation identities: `20260713` through `20260717`.

Variants:

1. `occluded_frozen_smolvla`
2. `full_frame_hold_last`
3. `sobel_edge_boost`
4. `gcap_no_temporal_ablation`
5. `gcap_full`
6. `clean_frozen_smolvla`
7. `clean_gcap_full`

GO requires full GCAP to beat the strongest occluded baseline by at least 5 absolute task-balanced success points, beat hold-last and no-temporal ablations, and retain clean performance within 2 points.
