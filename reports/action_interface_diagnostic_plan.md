# Action-Interface Diagnostic Plan

This report defines the planning-only gate after the reduced-scope learned-policy rollout still produced diagnostic success rate `0.0` and reward sum `0.0`.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\77_plan_action_interface_diagnostics.ps1
```

The planner reads `reports\reduced_scope_rollout_metric_summary_report.json` and optionally inspects local SmolVLA checkpoint metadata such as `config.json`. It writes ignored runtime outputs:

- `reports\action_interface_diagnostic_plan_report.json`,
- `reports\action_interface_diagnostic_plan_report.md`.

It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Primary current hypotheses:

- policy action shape is `[1, 6]` while LIBERO environment action dimension is 7,
- the gripper component is currently padded as `0.0`,
- action scale is nontrivial, with final action max absolute value about `0.793`,
- the selected task remains unsolved, so rollout scaling is premature.

Recommended next diagnostics:

- action normalization and unnormalizer metadata audit,
- gripper mapping and padding audit,
- observation state vector mapping audit,
- camera feature mapping audit,
- language prompt audit,
- zero-action versus SmolVLA-action diagnostic comparison.

All outputs remain diagnostic/local-pilot evidence only.

## Current Local Result

Latest local planner result: `proceed`.

Observed signals:

- diagnostic success rate: 0.0,
- reward sum: 0.0,
- policy action dimension: 6,
- environment action dimension: 7,
- action dimension mismatch: true,
- gripper component: 0.0,
- gripper padded zero: true,
- action max absolute value: about 0.793,
- action L2 norm: about 1.222,
- nontrivial action magnitude: true.

High-priority diagnostics:

- action dimension and gripper mapping,
- action normalization and scale,
- observation state mapping.

The planner also marks camera mapping, language prompt mapping, and zero-action versus SmolVLA-action comparison as useful next diagnostics.
