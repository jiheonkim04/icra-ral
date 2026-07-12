# SCVC-VLA Preregistration

Date: 2026-07-12 KST

Decision: `IMPLEMENTATION_PREREGISTERED`

## Fixed Sensor Shift

Use a synthetic camera-domain shift on preprocessed SmolVLA image tensors:

- gain: `0.42`
- bias: `0.28`

No Stage A result may change these values.

## Calibration

Calibration identities:

- `20260711..20260715`

Calibration uses clean observations only. It stores per-camera mean and standard deviation.

## Stage A

Held-out identities:

- `20260716..20260720`

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Policies:

1. `clean_frozen_smolvla`
2. `shifted_frozen_smolvla`
3. `known_inverse_affine`
4. `scvc_no_temporal`
5. `scvc_full`

Episode count:

- `50` total episodes.

Primary metric:

- task-balanced closed-loop success.

Mechanism metrics:

- mean image MSE to clean before/after canonicalization;
- mean absolute image delta from shifted input;
- per-camera calibration stats.

## Decision Rules

Permanent Stage A kill follows `reports/current_research_governance.md`.

Additionally, kill this formulation if:

- `known_inverse_affine` matches or beats `scvc_full`;
- `scvc_no_temporal` matches or beats `scvc_full`;
- `shifted_frozen_smolvla` is not degraded relative to clean and no shifted-condition headroom exists.

Otherwise proceed to Stage B.
