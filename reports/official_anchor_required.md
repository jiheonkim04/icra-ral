# Official Anchor Required

Date: 2026-07-08

## Decision

No new VLA method should be started from local proxy diagnostics under the current constraints.

## Basis

Local proxy and minimal approximations have repeatedly failed once compared with strong simple baselines. The ActionMap mini-anchor is the latest example: a real LIBERO/HDF5 metric and strong oracle candidate upper bound existed, but the learned local heatmap head lost to mean action and cheap MLP and collapsed candidate diversity.

This does not disprove the official ActionMap paper or any official safety benchmark. It does show that local proxy approximations are no longer a valid launch point for new method design in this project.

## Rule

No new method should be started without an official anchor reproduction.

In particular:

- do not implement Target-Grounded ActionMap from the failed local mini-anchor;
- do not tune another local ActionMap proxy until it passes by chance;
- do not create another local proxy topic;
- do not treat oracle, symbolic, auxiliary, or offline-only headroom as method evidence unless the official anchor and simple baselines are already green.

## Only Viable Next Steps

A. Official ActionMap reproduction with official code/assets.

B. Official LIBERO-Safety/SafeManip benchmark reproduction.

C. Stop VLA method search under current constraints.

Any other next step is outside the current evidence boundary.
