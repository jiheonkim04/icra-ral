# Project State

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-lerobot-baseline`

Current decision: `NEEDS_OFFICIAL_DATASET_CONVERSION`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, with official preprocessing, normalization, action conventions, and evaluation stack.

## Evidence Summary

- The local checkpoint at `C:\assets\checkpoints\smolvla` is a SmolVLA base-style checkpoint.
- Official LeRobot loader and processor factory load the local checkpoint successfully.
- One synthetic CPU-only official-loader forward pass produced a finite `[1, 6]` action.
- CUDA is available on the RTX 5080; bitsandbytes CUDA smoke passed.
- The mini-repro was intentionally CPU-only and did not train.
- The local checkpoint uses SO100 6D action normalizer tensors.
- LeRobot LIBERO expects 8D state and 7D continuous actions.
- Therefore the local checkpoint is not an official LIBERO baseline as-is.

## Conclusion

`NEEDS_OFFICIAL_DATASET_CONVERSION`

Next valid step: align the target LIBERO data/checkpoint path with LeRobot's official LIBERO 8D state / 7D action convention, either by acquiring official small assets or by converting a tiny local subset cleanly. Do not start method work until an official-compatible baseline is reproduced.
