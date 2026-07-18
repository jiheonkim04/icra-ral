# SGL-XVLA Stage 0 Identity Manifest Freeze

Decision: `SGL_HELDOUT_IDENTITY_MANIFEST_FROZEN_NO_TRAINING_NO_OURS`

This report-only gate froze identity roles before any SGL-XVLA control rollout, Ours rollout, training, optimizer step, or checkpoint write.

Frozen roles:

- Development residual: `20260727`, `20260730`, `20260733`.
- Clean retention: `20260731`, `20260732`.
- Held-out confirmatory pool: `20260734`, `20260735`, `20260736`, `20260737`.
- Mapping: `initial_state_index = reset_identity - 20260711` for task5.

Anti-cherry-pick rules: no adding, dropping, reordering, or relabeling identities after any SGL/control/Ours result; no held-out inspection before the later rollout protocol is frozen; no sign/threshold/clipping choices from held-out outcomes.

Comparator calibration: clean-retention identities block unacceptable degradation or added clipping, while held-out identities test the residual claim only under a later matched protocol. No universal beat-all rule is applied.

Validation: `py_compile` passed and focused pytest passed with `28 passed`.

Next: write a Stage0 completion/adjudication report, then decide whether a separate no-training rollout protocol may be frozen. Still no training, checkpoints, control rollout, or Ours rollout.
