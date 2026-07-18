# RL4IL Action-Oracle Prior Port Preregistration

- Decision: `RL4IL_ACTION_ORACLE_PRIOR_PORT_PREREGISTERED_NOT_LAUNCHED`
- Role: selected external prior comparator, not Ours.
- Reason: RL4IL remains the closest paper-level missing-camera prior, but the official release has constant-label supervision and no checkpoints, so a mechanism-faithful local port is required before any Ours design.

## Frozen panel

Use the same three-task wrist-dropout panel:

- `libero_goal/task0`, identities `20260733, 20260734, 20260735`
- `libero_object/task0`, identities `20260733, 20260734, 20260735`
- `libero_spatial/task5`, identities `20260731, 20260732, 20260735`

Identity mapping is fixed as `initial_state_index = reset_identity - 20260711`. Query observations must be live observations from those official reset states, not same-index HDF5 demos.

## Port mechanism

Preserve the RL4IL prior mechanism: frozen CLIP embeddings, modality-fair normalization, kNN/BFS candidate sets, PPO-guided retrieval, mask_1 in-hand-camera soft imputation, and open-loop replay of the retrieved training demonstration actions.

Replace the official release’s constant scalar labels with a paper-aligned action-sequence oracle: linearly resample each 7D action sequence to 64 steps, compute MSE plus `0.01` normalized length penalty, exclude self-candidates, and use the minimum-distance candidate inside the BFS set as the oracle.

## Bounded Stage A budget

- Conditions: clean and `mask_1` in-hand dropout.
- Episodes: 9 clean + 9 dropout.
- Max steps: 260.
- Settling: 10 no-op steps.
- Training: 1 epoch each for prediction PPO, prediction fusion, mod1 imputation PPO, and mod1 soft imputation.
- No X-VLA training, no Ours method, no SGL/OCR/AWF reruns.

Accept no prior result unless fidelity gates pass: nonconstant action-oracle targets, trainable prior parameters, finite nonzero gradients, optimizer steps, checkpoint write/reload, imputed mask_1 path, live reset-state queries, and full infra reporting.
