# ExecSpec-Repair Risk Register

## Active Risks

- STATE 3 exact-init replay is still diagnostic evidence, not benchmark or deployment evidence.
- Supervised calibration can accidentally look stronger than a deployable runtime repair if it is not labeled clearly.
- Gripper/sign/threshold mismatch may dominate metrics while translation and rotation remain unresolved.
- Existing exact-init replay is local and diagnostic-only, not paper-grade evaluation.
- Future states must not use future expert actions as rollout method actions.
- Simple affine/global baselines can match full ExecSpec-Repair on some mismatch classes.
- A single diagonal affine baseline can match full ExecSpec-Repair on the existing STATE 3 evidence.
- Default reset remains incompatible with the validated exact-init replay claim boundary.

## Current Mitigations

- STATE 3 labels calibration and exact-init replay as bounded diagnostic evidence only.
- Runtime reports record replay/rollout, training, loss, GPU, download, and OpenVLA-OFT status.
- STATE 3 reports calibration/eval split size, action sample counts, and `leakage_detected=false`.
- STATE 3 keeps exact expert, wrong executable spec, identity alias, clipping-only, global, diagonal, gripper-only, and full repair controls.
- The broad route is stopped or reframed because simple baselines matched full repair in `4` degraded replay cases.
- STATE 3.5 kills the broad route because diagonal affine matches full repair on `17 / 19` degraded replay cases and action recovery `1.0`.
- Any future revival must predeclare a harder benchmark where diagonal affine is not an adequate baseline.
