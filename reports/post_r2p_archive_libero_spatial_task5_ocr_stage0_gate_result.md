# OCR-XVLA Stage 0 Gate

Decision: `OCR_STAGE0_SPEC_FROZEN_NO_TRAINING_NO_OURS`

`OCR-XVLA` is now the active backup candidate after SGL-XVLA was blocked before rollout by simple-control equivalence.

Frozen OCR scope:

- Candidate: Observation-Consistency Retry for X-VLA.
- Mechanism: detect no-progress after the first grasp/lift attempt from allowed RGB/proprio/action-history signals, then allow one bounded re-center/regrasp retry.
- Development residual identities: `20260727`, `20260730`, `20260733`.
- Clean retention: `20260731`, `20260732`.
- Held-out pool: `20260734`–`20260737`.

Still forbidden: simulator state, reward, success flags, reset identity labels, HDF5 identity, training, optimizer steps, checkpoints, control rollout, and Ours rollout. R2P-XVLA and the current frozen SGL executable remain closed.

Validation: `py_compile` passed and focused pytest passed with `53 passed`.

Next: run only a report-only OCR observability/trigger audit from existing artifacts. No simulator episode, training, checkpoint, or Ours.
