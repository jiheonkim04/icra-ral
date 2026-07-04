# Action-Scale Diagnostic

This report documents the bounded action-scale diagnostic after adapter-strategy diagnostics.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\84_plan_action_scale_diagnostic.ps1
```

Bounded runner command:

```powershell
$env:ALLOW_ACTION_SCALE_DIAGNOSTIC="1"; powershell -ExecutionPolicy Bypass -File scripts\85_bounded_action_scale_diagnostic.ps1; Remove-Item Env:\ALLOW_ACTION_SCALE_DIAGNOSTIC -ErrorAction SilentlyContinue
```

The planner is read-only. The runner is bounded to one task, at most 10 steps per scale variant, CPU execution, no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no benchmark claim, and no paper-grade claim.

## Current Local Result

Latest bounded runner result: `passed` as diagnostic execution only.

The runner executed one `libero_10` task for up to 10 steps under:

- action adapter strategy: `policy_6d_delta_pose_plus_gripper_zero_hold`,
- action scales: `0.25`, `0.5`, and `1.0`.

Observed result:

- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- last action max absolute value by scale:
  - `0.25`: about `0.198273`,
  - `0.5`: about `0.396546`,
  - `1.0`: about `0.793093`,
- rollout scaling ready: false,
- benchmark claim: false,
- paper-grade claim: false.

Interpretation: explicit action scaling is wired and observable, and action magnitude changes as expected. However, changing only the action scale did not produce reward or task success on the selected diagnostic task. The next safe rung is a bounded prompt-format, camera-source, or state-sufficiency diagnostic. This remains diagnostic/local-pilot evidence only.
