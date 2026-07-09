# SmolVLA Custom Adapter Reusable Artifacts

Date: 2026-07-09 KST

## Reuse Boundary

These artifacts are reusable infrastructure and diagnostics. They are not method success evidence and should not be used to justify another custom adapter iteration without an official recipe baseline.

## Reusable Environment

- PEFT `0.19.1` import and LoRA smoke.
- bitsandbytes `0.49.2` import, 4-bit CUDA smoke, and 8-bit CUDA smoke.
- PyTorch `2.10.0+cu128`, CUDA runtime `12.8`.
- NVIDIA GeForce RTX 5080 local smoke with VRAM peak `2224.845` MB for tiny LoRA.
- Local SmolVLA checkpoint path: `C:\assets\checkpoints\smolvla`.

## Reusable Code Paths

- LIBERO 7D interface diagnostics and adapter code.
- Canonical LIBERO EEF feature builder:
  - HDF5 `obs/ee_states` path,
  - live `robot0_eef_pos + robot0_eef_quat` conversion,
  - selected orientation convention `xyzw_quaternion_axis_angle_0_to_2pi`.
- Exact-init expert replay stabilization and eligible-set construction.
- Replay bridge for expert, mean, ridge, MLP, previous adapter, clip-only, and range-fixed variants.
- Action range and gripper validity audit.

## Reusable Reports

- `reports/lora_environment_status.md`
- `reports/smolvla_libero_7d_interface_fix.md`
- `reports/exact_init_expert_replay_stabilization.json`
- `reports/smolvla_7d_live_feature_schema_fix.md`
- `reports/smolvla_7d_action_range_fix.md`
- `reports/smolvla_7d_action_validity_audit.md`
- `reports/smolvla_7d_gripper_range_audit.md`
- `reports/smolvla_7d_normalization_range_audit.md`

## Reuse Guidance

Reuse these artifacts to verify official baseline reproduction, inspect action/feature conventions, and prevent regressions. Do not reuse them as proof that the custom adapter route is RA-L-ready.

## Not Reusable As Claims

- Offline action L2 improvement alone.
- Custom gripper/range fixes.
- Clip-only or train-split affine calibration as method success.
- Exact-init replay failure analysis as a publishable method result.
