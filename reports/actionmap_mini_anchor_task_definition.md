# ActionMap Mini-Anchor Task Definition

Date: 2026-07-08

Purpose: run a bounded local ActionMap-style heatmap/candidate action-head gate before any Target-Grounded ActionMap work.

This is not official ActionMap reproduction, not Target-Grounded ActionMap, not failure mining, and not a paper claim.

## Scope

- Use local LIBERO HDF5 action chunks.
- Use deterministic train/eval split with no eval-label or future-action leakage.
- Run CPU-only tiny NumPy heads.
- Do not load large VLA checkpoints.
- Do not download assets, use GPU, run rollouts, run OpenVLA-OFT, or train a large VLA.

## Required Variants

1. Mean-action baseline.
2. Linear/L1 action head.
3. Cheap simple MLP action head.
4. ActionMap-style translation/rotation/gripper heatmap candidate head.
5. Oracle nearest candidate upper bound, labeled invalid as method evidence.

## Final Decision Labels

The STATE 1 result must choose exactly one:

- `GO_TARGET_GROUNDED_ACTIONMAP_STATE1`
- `KILL_ACTIONMAP_ANCHOR`
- `NEED_OFFICIAL_ACTIONMAP_REPRO`
- `TOO_HEAVY_LOCAL`
- `NO_REAL_METRIC`
