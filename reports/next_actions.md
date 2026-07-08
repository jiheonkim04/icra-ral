# Next Actions

Date: 2026-07-08

Current decision:

`KILL_ACTIONMAP_ANCHOR`

## Immediate Next Action

Stop. Do not proceed to Target-Grounded ActionMap from this mini-anchor result.

The local ActionMap-style heatmap/candidate substrate failed the hard gate:

- mean action beat ActionMap-style action L2;
- cheap MLP matched or beat ActionMap-style action L2;
- the learned heatmap collapsed to too few candidate bins.

## Allowed Future Work

Only two safe future directions remain:

1. Archive this as a killed local mini-anchor and choose a different official anchor.
2. If ActionMap itself must remain the anchor, plan an official-style ActionMap reproduction/source gate. That is a different task and must still avoid Target-Grounded method implementation until the anchor is green.

## Disallowed Next Work

Do not:

- implement Target-Grounded ActionMap;
- tune the local proxy until it passes by chance;
- treat the oracle candidate upper bound as method evidence;
- run OpenVLA-OFT;
- use GPU;
- download large assets;
- train a large VLA;
- run full benchmark or rollout from this result.

## Exact Next Step

Write a short archive/update pass if needed, then select a different official anchor or explicitly request an official-style ActionMap reproduction plan.
