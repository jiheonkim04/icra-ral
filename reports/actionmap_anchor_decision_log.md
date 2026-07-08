# ActionMap Anchor Decision Log

## STATE 0 Start

Decision: start ActionMap Anchor Reproduction and Failure Mining after ContactSet-VLA was archived and pushed.

Reason: recent local routes were killed after method-first ideas lost to simple baselines. This route is reproduction-first: approximate the ActionMap action-heatmap anchor locally, test it against mean, linear/L1, and cheap MLP baselines, and only then decide whether failure mining is justified.

Execution boundary at STATE 0: docs and diagnostic scaffold only. No rollout, GPU job, download, heavy VLA import, OpenVLA-OFT execution, or paper-grade claim.

## STATE 1 Result

Decision: kill or reframe the local ActionMap anchor reproduction before failure mining or any extension.

Reason: the bounded local HDF5 diagnostic produced real action-head metrics over `8` demos and `1008 / 432` deterministic train/eval records, but the ActionMap-style heatmap/candidate head did not beat simple baselines. ActionMap-style action L2 was `0.529931357`, worse than mean-action (`0.466767673`) and matched/beat by cheap MLP (`0.501926707`). The learned heatmap also collapsed to one rotation bin (`5 / 1 / 2` unique translation/rotation/gripper bins). The oracle nearest-candidate upper bound was strong (`0.065653208` action L2), so the failure is in learned candidate selection under this local setup, not only in grid resolution.

Consequence: do not proceed to STATE 2 failure mining and do not invent a new ActionMap extension from this run. A future reframe would need a stronger official-feature reproduction or different bounded data/feature setup that still beats mean, linear/L1, and cheap MLP baselines before any extension work.

Execution boundary: tiny CPU NumPy training happened and loss was computed. No replay/control metric, GPU job, download, heavy VLA import/model load, full VLA fine-tuning, OpenVLA-OFT execution, simulator rollout, token access, or paper-grade claim occurred.
