# Project State

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-dataset-alignment`

Current decision: `READY_FOR_OFFICIAL_ASSET_APPROVAL`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, with official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

## Evidence Summary

- Official `lerobot/smolvla_libero` and `lerobot/libero` are public, not gated, Apache-2.0 assets.
- Selected official asset pair size is approximately `2.647 GiB`, exceeding the objective's `2GB` no-approval threshold.
- Official `lerobot/libero` uses 8D state, 7D action, two 256x256 image/video keys, and LeRobot metadata/stats.
- Official `smolvla_libero` has 7D action normalizer stats from LIBERO, but its config has unresolved 6D-state/8D-stats and two-image/three-camera wrinkles that require a tiny shape smoke before training.
- Local HDF5 can likely be converted into a tiny official-compatible LeRobot dataset using 8D state and 7D action without the archived adapter route.
- No training, LoRA, rollout, large download, GPU model execution, or OpenVLA-OFT happened in this alignment pass.

## Conclusion

`READY_FOR_OFFICIAL_ASSET_APPROVAL`

Next valid step: either approve the official asset acquisition command in `reports/official_smolvla_libero_next_decision.md`, or proceed on a new milestone with the no-download tiny local HDF5-to-LeRobot conversion implementation. Do not start method work or LoRA training until an official-compatible sample smoke passes.
