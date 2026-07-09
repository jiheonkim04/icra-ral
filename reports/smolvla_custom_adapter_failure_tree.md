# SmolVLA Custom Adapter Failure Tree

Date: 2026-07-09 KST

## Root Outcome

`CLIP_ONLY_BASELINE_DOMINATES` led to strategic stop/pivot:

`OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

## Failure Tree

1. Native SmolVLA-to-LIBERO action interface mismatch
   - Found: native checkpoint path was SO100-style 6D while local LIBERO labels were 7D.
   - Fixed: custom LIBERO_7D adapter path with learned gripper and train-split-only normalization.
   - Residual: offline gains did not establish control reliability.

2. Exact-init replay instability
   - Found: replay judgment was unsafe until expert-success cases were isolated.
   - Fixed: 6-case expert-success eligible set.
   - Residual: learned policies still failed on expert-success cases.

3. Live/HDF5 feature path mismatch
   - Found: offline used HDF5 `ee_states = ee_pos + ee_ori`; live fallback used `robot0_eef_pos + robot0_eef_quat[:3]`.
   - Fixed: canonical XYZW axis-angle feature builder.
   - Evidence: feature L2 `2.248343 -> 0.033195`.
   - Residual: adapter replay still `0/6`.

4. Action range/controller-validity failure
   - Found: gripper dimension dominated clipping.
   - Fixed attempt: bounded output, signed-sigmoid gripper, BCE plus MSE, range penalty.
   - Evidence: clip rate `0.15625 -> 0.0`; controller-valid proxy `0.84375 -> 1.0`.
   - Residual: offline action L2 worsened `0.795274 -> 0.976681`; replay progress worsened `-0.041091 -> -0.902509`.

5. Simple baseline dominance
   - Clip-only replay progress: `-0.041091`.
   - Range-fixed replay progress: `-0.902509`.
   - Mean-action replay progress after feature fix: `0.038336`.
   - Conclusion: custom adapter route is not a reliable base for method claims.

## Stop Node

The route stops at `CLIP_ONLY_BASELINE_DOMINATES`, not because LoRA cannot run, but because local custom adapter repairs did not produce a control-reliable baseline.
