# Next Actions

Date: 2026-07-09 KST

Current decision:

`READY_FOR_REAL_METHOD_AFTER_INTERFACE_FIX`

## Immediate Next Action

Run a standard fixed-interface SmolVLA/LIBERO 7D baseline reproduction on an official or standard split.

## Why

The local 7D interface blocker is cleared for baseline work:

- LIBERO labels remain `7D`.
- SmolVLA native `6D` SO100 schema is preserved separately instead of being forced onto LIBERO labels.
- LIBERO 7D labels use train-split-only normalization.
- The gripper is learned as output dimension `6`, not hard-coded.
- One-sample overfit passed.
- One-demo overfit passed.
- The fixed 7D adapter beat mean-action and frozen/base on action L2.

## Current Metrics

- one-sample fixed adapter action L2: `0.0`
- one-demo fixed adapter action L2: `0.002593`
- previous split mean-action action L2: `0.486561`
- previous split fixed adapter action L2: `0.353069`
- larger split mean-action action L2: `1.082453`
- larger split fixed adapter action L2: `0.573503`
- larger split best MLP/ridge action L2: `0.518738`
- frozen/base SmolVLA action L2 from previous 6D run: `1.6029`

## Important Caveat

This was a bounded infrastructure repair, not a full SmolVLA LoRA result. The small state/time MLP baseline is still slightly stronger than the fixed 7D adapter on the larger held-out split, so the next step is baseline reproduction and target-module design, not a new method claim.

## Allowed Next Work

- Reproduce a standard fixed-interface SmolVLA/LIBERO 7D baseline on an official or standard split.
- Decide whether the next baseline should use a 7D adapter on frozen SmolVLA features, LoRA plus a 7D adapter, or an action-head replacement with the same train-only normalization.
- Keep mean-action, ridge/MLP, frozen/base, and standard fixed-interface adapter baselines in the table.
- Audit target modules only as baseline engineering.

## Disallowed Next Work

Do not:

- invent a new method,
- continue PatchGuard,
- start Target-Grounded ActionMap, SafeLoRA, PRISM, ActionMap, or another route,
- run OpenVLA-OFT,
- run rollout from this evidence,
- download large assets,
- make paper claims,
- treat the local 7D interface fix as a paper contribution.
