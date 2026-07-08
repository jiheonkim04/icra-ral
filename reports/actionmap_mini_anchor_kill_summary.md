# ActionMap Mini-Anchor Kill Summary

Date: 2026-07-08

## Final Decision

`KILL_ACTIONMAP_ANCHOR`

This kills the local minimal ActionMap approximation as a route into Target-Grounded ActionMap. It does not kill the official ActionMap paper.

## Original Anchor-Gate Hypothesis

A bounded ActionMap-style voxel heatmap/candidate decoder over local LIBERO HDF5 action chunks should beat mean-action, linear/L1, and cheap MLP action heads on held-out 7D action metrics before any Target-Grounded ActionMap extension is considered.

## Strongest Positive Evidence

- A real LIBERO/HDF5-backed metric was produced on `8` local demos with `1008 / 432` deterministic train/eval records.
- The oracle nearest-candidate upper bound was strong: action L2 `0.065653208`.
- This means the discretized candidate space had headroom, but only as an oracle upper bound, not as method evidence.

## Decisive Negative Evidence

- Mean-action action L2: `0.466767673`.
- Simple MLP action L2: `0.501926707`.
- ActionMap-style action L2: `0.529931357`.
- Linear/L1 action L2: `0.812610317`.
- Candidate top1: `0.018518519`.
- Candidate collapse: yes, with unique translation/rotation/gripper bins `5 / 1 / 2`.

The learned ActionMap-style head did not exploit the oracle headroom. It lost to mean action and was matched or beaten by the cheap MLP.

## Exact Kill Criteria Triggered

- Mean-action baseline matched or beat the ActionMap-style heatmap head.
- Cheap MLP action head matched or beat the ActionMap-style heatmap head.
- The ActionMap-style candidate selector collapsed to too few bins.

## Why Target-Grounded ActionMap Cannot Proceed

Target-Grounded ActionMap depends on a useful ActionMap-style action heatmap substrate. The local substrate failed before any target grounding was added. Adding target priors now would test a new method on top of a collapsed local decoder, so any gain would be uninterpretable and any failure would not distinguish grounding from decoder weakness.

## Revival Requirement

This family can be revived only by one of:

- official ActionMap reproduction with official code/assets and a credible standard setup; or
- a stronger non-collapsed heatmap implementation that first beats mean-action, linear/L1, and cheap MLP baselines without target grounding.

## No More Local Proxy Approximation

Another local proxy approximation should not be attempted. The project has repeatedly found that local proxy methods can produce plausible headroom or auxiliary metrics while losing to simple baselines. The next ActionMap-family step must be an official anchor reproduction or no ActionMap-family step at all.
