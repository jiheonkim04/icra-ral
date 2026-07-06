# ExecSpec-Repair Risk Register

## Active Risks

- HDF5 action-drift evidence may not predict closed-loop replay degradation.
- Supervised calibration can accidentally look stronger than a deployable runtime repair if it is not labeled clearly.
- Gripper/sign/threshold mismatch may dominate metrics while translation and rotation remain unresolved.
- Existing exact-init replay is local and diagnostic-only, not paper-grade evaluation.
- Future states must not use future expert actions as rollout method actions.
- Exact-init replay now shows mismatch degradation, but calibrated repair replay has not happened yet.

## Current Mitigations

- STATE 1 labels calibration as supervised HDF5 calibration/evaluation.
- Runtime reports record replay/rollout, training, loss, GPU, download, and OpenVLA-OFT status.
- Next state must move the strongest mismatch and repair candidate toward exact-init replay if safe.
- STATE 1 reports separate HDF5 supervised calibration from simulator replay outcomes.
