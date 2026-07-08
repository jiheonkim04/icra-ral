# Project State

Date: 2026-07-08

Branch:

`main`

Current main commit before this archive pass:

`8eabacb Run ActionMap mini-anchor gate`

Current decision:

`KILL_ACTIONMAP_ANCHOR`

## Current Archive Pass Boundary

- Experiments happened: no.
- Training happened: no.
- Loss computation happened: no.
- Rollout/replay happened: no.
- Downloads happened: no.
- GPU use happened: no.
- OpenVLA-OFT happened: no.
- Full official ActionMap reproduction happened: no.
- Target-Grounded ActionMap implementation happened: no.
- New method implementation happened: no.

## Prior Mini-Anchor Evidence

The previous committed gate produced the following bounded local LIBERO/HDF5 metrics:

- usable demos: `8`
- train/eval split: `deterministic_per_demo_time_holdout`
- train/eval records: `1008 / 432`
- mean-action action L2: `0.466767673`
- linear/L1 action L2: `0.812610317`
- simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- oracle nearest-candidate action L2: `0.065653208`
- candidate top1: `0.018518519`
- candidate collapse: yes, unique translation/rotation/gripper bins `5 / 1 / 2`

## Conclusion

The local ActionMap-style heatmap/candidate head failed the hard gate. It did not beat mean action or cheap MLP, and it collapsed candidate diversity. The oracle candidate upper bound remains useful evidence of candidate-space headroom but is invalid as learned method evidence.

## Current Next-Step Boundary

Only three next steps remain valid:

A. Official ActionMap reproduction with official code/assets.

B. Official LIBERO-Safety/SafeManip benchmark reproduction.

C. Stop VLA method search under current constraints.
