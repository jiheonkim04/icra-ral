# Action-Interface Metadata Audit

This report defines the metadata/report-only audit after the action-interface diagnostic planner.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\78_audit_action_interface_metadata.ps1
```

The audit reads:

- `reports\action_interface_diagnostic_plan_report.json`,
- `reports\reduced_scope_rollout_metric_summary_report.json`,
- local SmolVLA `config.json`,
- local SmolVLA `policy_preprocessor.json`,
- local SmolVLA `policy_postprocessor.json`,
- the local rollout bridge source file.

It writes ignored runtime outputs:

- `reports\action_interface_metadata_audit_report.json`,
- `reports\action_interface_metadata_audit_report.md`.

It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

The audit is intended to decide whether the next safe step should be a zero-action versus SmolVLA-action diagnostic comparison, an action adapter patch plan, or more metadata inspection.

## Current Local Result

Latest local audit result: `proceed`.

High-priority findings:

- `action_dim_mismatch`: SmolVLA action dimension is 6 while the LIBERO environment action dimension is 7.
- `gripper_constant_zero`: the gripper component is currently padded as `0.0`.
- `state_truncation_risk`: the state builder collects end-effector pose, gripper qpos, joint pos, and joint vel, then truncates to policy state dimension 6.
- `nontrivial_actions_zero_reward`: actions have nontrivial magnitude but diagnostic success rate and reward remain 0.0.

Medium-priority finding:

- `camera_feature_name_mismatch`: config image feature names and preprocessor image feature names use different naming conventions.

Recommended next step: create a bounded zero-action versus SmolVLA-action diagnostic and an explicit action/state adapter patch plan before rollout scaling.
