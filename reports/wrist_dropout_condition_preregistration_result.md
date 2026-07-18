# Wrist-Camera Dropout Condition Preregistration

- Decision: `CLAIM_SPECIFIC_CONDITION_WRIST_CAMERA_DROPOUT_PREREGISTERED`
- Axis selected: claim-specific controlled condition.
- No Ours method or acronym is selected before condition verification.
- Natural-reset mining remains closed as `NATURAL_RESET_SEARCH_SATURATED`.
- OCR is archived as `OCR_TRIGGER_OBSERVABILITY_FAIL`.

## Condition

`wrist_camera_dropout_partial_observation`: zero the wrist-camera RGB observation before frozen X-VLA action generation while leaving simulator state, rewards, success criteria, proprioception, and agentview RGB unchanged.

This is physically meaningful as wrist-camera occlusion/dropout and can be matched later for Prior and Prior + Ours.

## Verification protocol

- Task: `libero_spatial/task5`.
- Discovery identities: `20260731`, `20260732`.
- Clean baseline: both succeeded during the bounded OCR trace pass.
- Perturbation: `wrist_blackout`.
- Verification passes if the two perturbed episodes complete without infrastructure failure and success drops by at least one episode from the 2/2 clean baseline.
