# ActionMap Mini-Anchor Decision Log

## 2026-07-08: STATE 0 Start

Decision: start the bounded ActionMap Mini-Anchor Gate on branch `codex/actionmap-mini-anchor-gate`.

Reason: the reset decision was `NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`. Before Target-Grounded ActionMap can be considered, a local ActionMap-style heatmap/candidate decoder must beat mean action, linear/L1, and cheap MLP baselines on real LIBERO/HDF5 action metrics without candidate collapse.

Boundary: no Target-Grounded ActionMap implementation, full ActionMap reproduction, benchmark rollout, download, GPU use, OpenVLA-OFT, large VLA training, or new proxy topic.

## 2026-07-08: STATE 1 Result

Final decision: `KILL_ACTIONMAP_ANCHOR`

Reason: the local HDF5-backed diagnostic produced a real metric, but the learned ActionMap-style heatmap head failed the hard gate.

Key metrics:

- dataset/split: `8` local LIBERO HDF5 demos, `deterministic_per_demo_time_holdout`
- train/eval records: `1008 / 432`
- mean-action action L2: `0.466767673`
- linear/L1 action L2: `0.812610317`
- simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- oracle nearest-candidate action L2: `0.065653208`
- candidate top1 / translation top3 / rotation top3 / NLL: `0.018519 / 0.111111 / 0.981481 / 8.41813`
- candidate diversity: unique translation/rotation/gripper bins `5 / 1 / 2`

Triggered kill criteria:

- mean-action baseline matches or beats ActionMap-style head;
- simple MLP matches or beats ActionMap-style head;
- ActionMap-style head collapses to too few candidates.

Consequence: do not proceed to Target-Grounded ActionMap in this run. The oracle row shows discretization headroom, but it is invalid as method evidence because the learned heatmap selector did not exploit it.

Execution boundary:

- bounded mini-anchor diagnostic happened: yes;
- tiny CPU training happened: yes;
- loss computed: yes;
- replay/control happened: no;
- downloads/GPU/OpenVLA-OFT happened: no / no / no;
- full official ActionMap reproduction happened: no;
- Target-Grounded ActionMap implementation happened: no.
