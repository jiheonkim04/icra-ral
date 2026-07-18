# SGL-XVLA No-Training Rollout Protocol Freeze

Decision: `SGL_NO_TRAINING_ROLLOUT_PROTOCOL_FROZEN_NO_EPISODES_RUN`

This report-only artifact froze the first development rollout protocol for SGL-XVLA and the fixed lift/regrasp control. It did not implement a runner, load a model, launch a simulator, train, write checkpoints, run the control, or run Ours.

Frozen development protocol:

- Development residual identities: `20260727`, `20260730`, `20260733`.
- Clean-retention identities: `20260731`, `20260732`.
- Held-out pool: `20260734`–`20260737`, not used in this protocol.
- New episode budget if later authorized: 5 control + 5 SGL = 10 simulator episodes; X-VLA reference reruns = 0.
- Arms: frozen existing X-VLA evidence, `FIXED-LIFT-REGRASP-CONTROL`, and `SGL-XVLA`.

Decision rules are comparator-specific: SGL must pass residual and clean-retention conditions, while the fixed control blocks novelty only if it explains the gain at equal/lower cost. No universal beat-all rule is applied.

Authorization boundary: this artifact authorizes only the next report-only runner preflight. It does not authorize simulator episodes, held-out rollout, training, checkpoint writing, control rollout, or Ours rollout.

Validation: `py_compile` passed and focused pytest passed with `40 passed`.

Next: implement the runner preflight for the frozen protocol. Still no simulator episodes until that preflight passes.
