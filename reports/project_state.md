# Project State

Date: 2026-07-09 KST

Branch: `codex/archive-smolvla-custom-adapter-stop-pivot`

Current decision: `OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

## Current Route

The custom SmolVLA 7D adapter route is archived as stopped. The project should not continue custom adapter tuning, range/gripper variants, PatchGuard, TG-7D, SafeLoRA, PRISM, or ActionMap as a main RA-L path.

## Evidence Summary

- PEFT/bitsandbytes/CUDA/RTX 5080 SmolVLA LoRA path works.
- LIBERO 7D interface was fixed.
- Expert replay stable set exists with 6 expert-success eligible cases.
- Live/HDF5 feature schema mismatch was fixed: feature L2 `2.248343 -> 0.033195`.
- Learned adapter still failed replay/progress after feature fix: adapter success `0/6`, progress `-0.041091`.
- Action range fix improved validity but degraded quality/control:
  - clip rate `0.15625 -> 0.0`,
  - controller-valid proxy `0.84375 -> 1.0`,
  - offline action L2 `0.795274 -> 0.976681`,
  - replay progress `-0.041091 -> -0.902509`.
- Clip-only baseline matched or beat the range-fixed adapter: `-0.041091` vs `-0.902509`.

## Conclusion

`OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

Next valid step: reproduce an official SmolVLA/LeRobot/OpenVLA-style baseline recipe with official preprocessing, normalization, action/gripper conventions, and eval/replay stack before any method work.
