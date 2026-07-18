# Wrist-Dropout Generality Study Preregistration

- Decision: `WRIST_DROPOUT_GENERALITY_STUDY_PREREGISTERED`
- Policy: frozen official X-VLA only.
- No Ours method, training, optimizer, checkpoint, or broad natural-reset scan.
- AWF-XVLA is reclassified as `SIMPLE_CAMERA_IMPUTATION_CONTROL`, not a learned missing-modality method.

## Frozen paired panel

Exactly 3 tasks × 3 identities are frozen before execution. Each identity gets a clean episode and a matched wrist-dropout episode.

| suite | task | instruction | identities |
|---|---:|---|---|
| `libero_goal` | 0 | open the middle drawer of the cabinet | `20260733`, `20260734`, `20260735` |
| `libero_object` | 0 | pick up the alphabet soup and place it in the basket | `20260733`, `20260734`, `20260735` |
| `libero_spatial` | 5 | pick up the black bowl on the ramekin and place it on the plate | `20260731`, `20260732`, `20260735` |

Maximum execution: 9 clean episodes and 9 matched `wrist_blackout` episodes.

## Decision rule

Return exactly one:

- `WRIST_DROPOUT_REPEATED_PROBLEM_CONFIRMED`: valid run, clean successes >= 3, clean-success/dropout-failure flips >= 3, and flips across at least 2 tasks or at least 4 independent identities.
- `WRIST_DROPOUT_TASK5_LOCALIZED`: valid run with flips only on original `libero_spatial/task5`.
- `WRIST_DROPOUT_UNDERPOWERED_ONE_FIXED_EXPANSION_ALLOWED`: valid run but neither confirmed nor localized.
- `WRIST_DROPOUT_EVALUATION_INVALID`: missing paired result, infrastructure exception, or wrong episode count.
