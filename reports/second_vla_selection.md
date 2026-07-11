# Second VLA Selection

Date: 2026-07-11 KST

Selected second backbone: `OpenVLA-OFT`

Selected checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`

## Why This Backbone

OpenVLA-OFT is selected because it has:

- official open source: `https://github.com/moojink/openvla-oft`
- official LIBERO evaluation path
- released LIBERO checkpoints
- a single combined checkpoint covering `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`
- public non-gated Hugging Face access
- MIT license metadata

## Why Not A Fallback Backbone

The fallback rule only permits A1 or another current open VLA if OpenVLA-OFT is infeasible. OpenVLA-OFT is not infeasible; it is blocked by download approval and hardware validation.

## Selected Checkpoint Metadata

- model id: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- Hugging Face SHA: `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`
- license: `mit`
- gated/private: `false` / `false`
- expected size: `14.845` GiB
- download status: `not_downloaded`

## Current Selection Status

Selection status: `selected_but_not_downloaded`

State 1 decision: `SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED`
