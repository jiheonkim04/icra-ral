# State-Sufficiency Diagnostic

This report documents the bounded state-sufficiency diagnostic after adapter-strategy, action-scale, prompt-format, and camera-source diagnostics.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\90_plan_state_sufficiency_diagnostic.ps1
```

Bounded runner command:

```powershell
$env:ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC="1"; powershell -ExecutionPolicy Bypass -File scripts\91_bounded_state_sufficiency_diagnostic.ps1; Remove-Item Env:\ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC -ErrorAction SilentlyContinue
```

The planner is read-only. The runner is bounded to one task, at most 10 steps per state variant, CPU execution, no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no benchmark claim, and no paper-grade claim.

## Current Local Result

Latest bounded runner result: `passed` as diagnostic execution only.

The runner executed one `libero_10` task for up to 10 steps under:

- prompt strategy: `bddl_language`,
- camera alias strategy: `current_aliases`,
- action adapter strategy: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- action scale: `1.0`,
- state adapter strategies:
  - `eef_pos_quat_first3`,
  - `eef_pos_quat_last3`,
  - `eef_pos_zero_rot`.

Observed state mappings:

- `eef_pos_quat_first3`: `robot0_eef_pos` plus quaternion entries `[0:3]`,
- `eef_pos_quat_last3`: `robot0_eef_pos` plus quaternion entries `[1:4]`,
- `eef_pos_zero_rot`: `robot0_eef_pos` plus zero rotation features.

Observed result:

- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- state strategy changes produced different continuous action previews,
- rollout scaling ready: false,
- benchmark claim: false,
- paper-grade claim: false.

Interpretation: state-vector strategy is execution-tested and recorded in metadata, but changing only the state mapping did not produce reward or task success on the selected diagnostic task. The learned-policy diagnostic stack has now tested gripper strategy, action scale, prompt format, camera source, and state sufficiency without finding a positive signal. The next safe rung is a diagnostic synthesis/no-go report or a more basic environment-policy compatibility check, not rollout scaling.
