# ActionMap Anchor Reproduction Plan

## STATE 1 Mini-Anchor Feasibility Diagnostic

Use existing local assets only:

- local LIBERO HDF5 actions,
- HDF5 observation features when present,
- CPU NumPy heads,
- deterministic train/eval split with no eval-action label leakage.

Required variants:

1. `mean_action_baseline`
2. `linear_l1_action_head`
3. `simple_mlp_action_head`
4. `actionmap_heatmap_candidate_head`
5. `oracle_nearest_action_candidate_upper_bound`

Execution order:

1. Produce a LIBERO/HDF5-backed metric.
2. Compute the oracle nearest-candidate upper bound.
3. Stop if the oracle upper bound does not clearly beat mean-action and linear/L1 baselines.
4. Fit the tiny ActionMap-style head only if oracle headroom exists.
5. Compare against mean-action, linear/L1, and cheap MLP.
6. Stop after the STATE 1 decision.

ActionMap-style approximation:

- normalize/clamp 7D LIBERO actions to `[-1, 1]`,
- predict translation voxel heatmap,
- predict rotation voxel heatmap,
- predict binary gripper heatmap,
- decode by candidate argmax,
- report heatmap NLL, top-k candidate accuracy, and candidate collapse.

Metrics:

- 7D action L2,
- translation L2,
- rotation L2,
- gripper error,
- action L1,
- action candidate top-k accuracy,
- heatmap/candidate NLL,
- per-task and per-phase breakdown,
- action diversity/candidate collapse,
- oracle nearest candidate upper bound,
- replay/progress only if a later separate risk assessment is green.

Continue only if ActionMap-style heatmap/candidate head beats mean-action, linear/L1, and cheap MLP baselines on held-out action quality, does not collapse to trivial candidate selection, and has oracle candidate headroom.

Kill if mean-action, linear/L1, or cheap MLP matches or beats the ActionMap-style head, if the oracle candidate upper bound has no headroom, if no real HDF5 metric appears, if the implementation starts becoming official full reproduction, or if reproduction requires full VLA training.
