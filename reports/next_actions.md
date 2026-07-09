# Next Actions

Date: 2026-07-09 KST

Current decision:

`ACTION_INTERFACE_BUG`

## Immediate Next Action

Fix or replace the SmolVLA/LIBERO action interface before any method work.

## Why

The diagnosis found enough local data for a larger split, but the action interface is not correct:

- HDF5 actions are `7D`.
- Local SmolVLA action head is `6D`.
- SmolVLA pre/postprocessor action shape is `6D`.
- Checkpoint action normalizer is SO100-style `MEAN_STD`, while local LIBERO actions are small roughly `[-1, 1]` actions.
- The gripper dimension is synthesized by an adapter rather than produced by the model.
- One-sample overfit failed in select-action action L2.
- One-demo overfit failed against the same-demo mean-action baseline.

## What Worked

- Label reconstruction sanity passed.
- Action chunk horizon alignment passed.
- No off-by-one was detected in the chunk builder.
- Bounded LoRA training ran and loss was computed.
- RTX 5080 VRAM stayed low: `1189.167` MB peak.

## Current Metrics

- mean-action action L2: `0.486561`
- frozen/base action L2: `1.6029`
- best LoRA action L2: `0.912258`
- best small MLP/ridge action L2: `0.401848`
- LoRA beats mean-action: no
- LoRA beats small MLP/ridge: no

## Allowed Next Work

- Action dimension adapter audit: decide whether the correct baseline target is 6D first-six action, 7D with learned gripper, or a true SmolVLA-compatible action space.
- Normalization audit: apply or reproduce the correct SmolVLA action normalization/unnormalization path for local LIBERO labels.
- Gripper convention audit: remove hard-coded close if it is invalid for the target baseline.
- Standard LoRA rerun only after the action interface is corrected.

## Disallowed Next Work

Do not:

- invent a new method,
- continue PatchGuard,
- start Target-Grounded ActionMap, SafeLoRA, PRISM, ActionMap, or another route,
- run OpenVLA-OFT,
- run rollout from this evidence,
- download large assets,
- make paper claims,
- treat this local proxy evidence as final paper evidence.
