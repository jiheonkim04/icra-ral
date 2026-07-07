# Phase-Locked Action Chunk Retiming Task Definition

Long title: Event-Locked Retiming of Action Chunks for Robust Robot Policy Execution.

Hypothesis: manipulation replay can fail when action chunks are temporally out of phase, and event-locked retiming can recover by aligning actions to approach, gripper close, object motion, lift, and place/contact events.

Scope:
- one local LIBERO/RoboSuite task first,
- HDF5 expert demonstrations and exact-init replay,
- synthetic temporal perturbations of expert chunks,
- no VLA training, OpenVLA-OFT, downloads, GPU jobs, or paper-grade claims,
- no success-label, BDDL target, task ID, filename, or dataset target leakage for inference-time event selection.

Primary comparison: event-locked retiming must beat raw perturbed replay, fixed time shift, repeat-last/hold, gripper-only correction, global scale, diagonal affine, linear time warp, and nearest-progress demo when feasible.

Evidence level: bounded replay/control diagnostic only.
