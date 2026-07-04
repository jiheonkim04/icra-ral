# Adapter-Strategy/Action-Scale Diagnostics Plan

This report defines the planning gate after explicit rollout bridge adapter wiring.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\82_plan_adapter_strategy_action_scale_diagnostics.ps1
```

The planner reads existing reports and source files only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

The planner is green only if:

- the action-interface audit is ready for adapter-strategy diagnosis,
- the zero-action comparison is ready for adapter-strategy diagnosis,
- reduced-scope rollout metrics include explicit adapter metadata,
- the rollout bridge source uses the explicit action adapter hook.

## Planned Diagnostic Envelope

The first runner should be separately gated and bounded:

- task count: 1,
- max steps per variant: 10,
- max first-run variants: 3,
- expected runtime: under 15 minutes,
- expected VRAM: 0 GB,
- no downloads,
- no training,
- no GPU job,
- no OpenVLA-OFT,
- no benchmark, SOTA, or paper-grade claim.

First diagnostic variants:

- `policy_6d_delta_pose_plus_gripper_zero_hold`,
- `policy_6d_delta_pose_plus_gripper_open`,
- `policy_6d_delta_pose_plus_gripper_close`.

Later diagnostics may add action-scale variants, prompt-format checks, and camera-alias checks after the gripper-strategy runner is implemented and validated.

## Runner

Bounded runner command:

```powershell
$env:ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC="1"; powershell -ExecutionPolicy Bypass -File scripts\83_bounded_adapter_strategy_diagnostic.ps1; Remove-Item Env:\ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC -ErrorAction SilentlyContinue
```

The runner must refuse broad rollout, benchmark, OpenVLA-OFT, GPU, matrix, tiny-rollout, and training gates. It writes ignored runtime outputs:

- `reports\adapter_strategy_action_scale_diagnostic_report.json`,
- `reports\adapter_strategy_action_scale_diagnostic_report.md`.

## Current Local Result

Latest local planner result: `proceed`.

The planner found:

- adapter metadata is present,
- adapter wiring is clean,
- current action adapter strategy is `policy_6d_delta_pose_plus_gripper_zero_hold`,
- diagnostic success rate remains 0.0,
- reward sum remains 0.0,
- rollout bridge has the explicit strategy hook,
- ready for a separately gated adapter-strategy diagnostic runner,
- not ready for rollout scaling.

Latest bounded runner result: `passed` as diagnostic execution only.

The runner executed one `libero_10` task for up to 10 steps under each of the first three gripper strategies:

- `policy_6d_delta_pose_plus_gripper_zero_hold`,
- `policy_6d_delta_pose_plus_gripper_open`,
- `policy_6d_delta_pose_plus_gripper_close`.

Observed result:

- variants completed: 3,
- wrapper/execution passed for all variants,
- diagnostic success rate: 0.0 for all variants,
- reward sum: 0.0 for all variants,
- observed gripper components: `0.0`, `1.0`, and `-1.0`,
- rollout scaling ready: false,
- benchmark claim: false,
- paper-grade claim: false.

Interpretation: explicit gripper-strategy selection is wired and observable, but changing only the gripper strategy did not produce reward or task success on the selected diagnostic task. The next safe rung is a bounded action-scale, prompt-format, camera-source, or state-sufficiency diagnostic. This remains diagnostic/local-pilot evidence only.
