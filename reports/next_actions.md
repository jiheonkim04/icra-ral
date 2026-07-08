# Next Actions

Date: 2026-07-08

Current decision:

`KILL_ACTIONMAP_ANCHOR`

## Immediate Next Action

Stop local ActionMap proxy work. Do not proceed to Target-Grounded ActionMap from the mini-anchor result.

## Why

The local ActionMap-style heatmap/candidate substrate failed the hard gate:

- mean-action action L2 `0.466767673` beat ActionMap-style action L2 `0.529931357`;
- cheap MLP action L2 `0.501926707` matched or beat ActionMap-style action L2 `0.529931357`;
- candidate top1 was `0.018518519`;
- candidate diversity collapsed to unique translation/rotation/gripper bins `5 / 1 / 2`.

The oracle nearest-candidate upper bound was strong at action L2 `0.065653208`, but it is invalid as method evidence because the learned selector did not exploit it.

## Only Allowed Next Steps

A. Official ActionMap reproduction with official code/assets.

B. Official LIBERO-Safety/SafeManip benchmark reproduction.

C. Stop VLA method search under current constraints.

## Disallowed Next Work

Do not:

- implement Target-Grounded ActionMap;
- invent a new method;
- tune another local proxy approximation;
- create another local proxy topic;
- treat the oracle candidate upper bound as method evidence;
- run OpenVLA-OFT;
- use GPU;
- download large assets;
- train a large VLA;
- run full benchmark or rollout from this result.
