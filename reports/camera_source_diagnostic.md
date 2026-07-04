# Camera-Source Diagnostic

This report documents the bounded camera-source diagnostic after adapter-strategy, action-scale, and prompt-format diagnostics.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\88_plan_camera_source_diagnostic.ps1
```

Bounded runner command:

```powershell
$env:ALLOW_CAMERA_SOURCE_DIAGNOSTIC="1"; powershell -ExecutionPolicy Bypass -File scripts\89_bounded_camera_source_diagnostic.ps1; Remove-Item Env:\ALLOW_CAMERA_SOURCE_DIAGNOSTIC -ErrorAction SilentlyContinue
```

The planner is read-only. The runner is bounded to one task, at most 10 steps per camera alias variant, CPU execution, no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no benchmark claim, and no paper-grade claim.

## Current Local Result

Latest bounded runner result: `passed` as diagnostic execution only.

The runner executed one `libero_10` task for up to 10 steps under:

- prompt strategy: `bddl_language`,
- action adapter strategy: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- action scale: `1.0`,
- camera alias strategies:
  - `current_aliases`,
  - `camera3_eye_in_hand`,
  - `all_agentview`.

Observed image sources:

- `current_aliases`: camera1 `agentview_image`, camera2 `robot0_eye_in_hand_image`, camera3 `agentview_image`,
- `camera3_eye_in_hand`: camera1 `agentview_image`, camera2 `robot0_eye_in_hand_image`, camera3 `robot0_eye_in_hand_image`,
- `all_agentview`: camera1 `agentview_image`, camera2 `agentview_image`, camera3 `agentview_image`.

Observed result:

- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- camera source changes produced different continuous action previews,
- rollout scaling ready: false,
- benchmark claim: false,
- paper-grade claim: false.

Interpretation: camera-source wiring is working and source choices are recorded in metadata, but changing only camera alias selection did not produce reward or task success on the selected diagnostic task. The next safe rung is a bounded state-sufficiency diagnostic. This remains diagnostic/local-pilot evidence only.
