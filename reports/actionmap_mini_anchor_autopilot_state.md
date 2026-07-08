# ActionMap Mini-Anchor Autopilot State

Date: 2026-07-08

## Current State

- Branch: `codex/actionmap-mini-anchor-gate`
- Base commit: `8500db2 Consolidate research reset direction`
- STATE 0 docs: complete.
- STATE 1 diagnostic: complete.
- Final decision: `KILL_ACTIONMAP_ANCHOR`
- Target-Grounded ActionMap implementation: not allowed and not started.
- Full ActionMap reproduction: not run.
- Downloads/GPU/OpenVLA-OFT: no / no / no.
- Large VLA training: no.

## STATE 1 Result

- Dataset: local LIBERO HDF5 under `C:\assets\data\libero`
- Usable demos: `8`
- Split: `deterministic_per_demo_time_holdout`
- Train/eval records: `1008 / 432`
- Mean-action action L2: `0.466767673`
- Linear/L1 action L2: `0.812610317`
- Simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- Oracle nearest-candidate action L2: `0.065653208`
- ActionMap-style beats mean/linear: `false / true`
- Simple MLP matches or beats ActionMap-style: `true`
- Candidate collapse: `true`, unique translation/rotation/gripper bins `5 / 1 / 2`

## Triggered Kill Criteria

- ActionMap-style head collapsed to too few action candidates.
- Mean-action baseline matches or beats the ActionMap-style heatmap head.
- Cheap MLP action head matches or beats the ActionMap-style heatmap head.

## Exact Next Step

Stop. Do not proceed to Target-Grounded ActionMap from this mini-anchor result.
