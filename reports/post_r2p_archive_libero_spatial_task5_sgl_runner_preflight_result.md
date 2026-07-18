# SGL-XVLA Runner Preflight

Decision: `SGL_RUNNER_PREFLIGHT_BLOCKED_SIMPLE_CONTROL_EQUIVALENCE_NO_ROLLOUT`

The runner preflight blocked SGL-XVLA before any simulator episode. Under the frozen Stage0 constraints, the current SGL executable would have the same language-only activation, fixed lift/regrasp schedule, saturation guard, and post-bias clamp as the fixed lift/regrasp simple control.

Scientific interpretation: the simple control explains the current frozen SGL behavior before rollout. This is a bounded kill of the current SGL-XVLA executable, not a reopening of R2P-XVLA or any archived method.

No runtime happened: no model load, no simulator episode, no control rollout, no Ours rollout, no training, no optimizer step, and no checkpoint write.

Comparator-role status:

- `SIMPLE_EXPLANATION_STATUS = SIMPLE_CONTROL_EXPLAINS_GAIN`
- `ABLATION_COMPONENT_STATUS = KEY_COMPONENT_NOT_SUPPORTED`
- `OVERALL_PAPER_CANDIDATE_STATUS = SIMPLE_CONTROL_EXPLAINS_GAIN`

Validation: `py_compile` passed and focused pytest passed with `47 passed`.

Next: do not run SGL-XVLA. Start report-only Stage0 gating for the backup `OCR-XVLA` candidate. Still no training, checkpoints, simulator episode, or Ours rollout.
