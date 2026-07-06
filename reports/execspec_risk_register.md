# ExecSpec-Repair Risk Register

## Active Risks

- STATE 2 exact-init repair replay is still a tiny diagnostic, not benchmark evidence.
- Supervised calibration can accidentally look stronger than a deployable runtime repair if it is not labeled clearly.
- Gripper/sign/threshold mismatch may dominate metrics while translation and rotation remain unresolved.
- Existing exact-init replay is local and diagnostic-only, not paper-grade evaluation.
- Future states must not use future expert actions as rollout method actions.
- Calibration may overfit the small local split unless STATE 3 expands held-out tasks and demos carefully.

## Current Mitigations

- STATE 2 labels calibration and exact-init replay as bounded diagnostic evidence only.
- Runtime reports record replay/rollout, training, loss, GPU, download, and OpenVLA-OFT status.
- STATE 2 reports calibration/eval split size, action sample counts, and `leakage_detected=false`.
- STATE 2 keeps exact expert, wrong executable spec, identity, clipping-only, global, diagonal, gripper, split, and full repair replay controls.
- STATE 3 must broaden replay/rollout validation without using held-out eval actions for repair fitting.
