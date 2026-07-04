# Reduced-Scope Rollout Metric Summary

This report defines the summary-only metric layer for the one-task, 10-step reduced-scope learned-policy LIBERO diagnostic.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\76_generate_reduced_scope_rollout_metric_summary.ps1
```

The script reads `reports\bounded_reduced_scope_learned_policy_rollout_report.json` and writes ignored runtime outputs:

- `reports\reduced_scope_rollout_metric_summary_report.json`,
- `reports\reduced_scope_rollout_metric_summary_report.md`.

It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Tracked metrics:

- source runner pass/fail,
- task count, step count, and policy call count,
- diagnostic success count and diagnostic success rate,
- reward sum,
- policy latency summary,
- action shape and environment action dimensions,
- final action preview, action max absolute value, action L2 norm, and gripper component,
- observed failure modes.

The output evidence label remains `reduced_scope_learned_policy_diagnostic`. It is not standard LIBERO success, not benchmark success, not counterfactual robustness evidence, not SOTA evidence, and not paper-grade evidence.

## Current Local Result

Latest local summary passed as report-only diagnostic evidence:

- source runner passed: true,
- tasks completed: 1,
- total steps: 10,
- policy calls: 10,
- diagnostic success count: 0,
- diagnostic success rate: 0.0,
- reward sum: 0.0,
- mean policy latency from the recorded final step: about 0.147 seconds,
- policy action shape: `[1, 6]`,
- environment action dimension: 7,
- final action preview: `[-0.353193, 0.021613, 0.742327, 0.793093, 0.392145, -0.18439, 0.0]`,
- final action max absolute value: about 0.793,
- final action L2 norm: about 1.222,
- gripper component: 0.0,
- failure mode: `diagnostic_success_check_false`.

This confirms that the longer diagnostic execution path is stable and nontrivial actions are being sent, but the selected task is still not solved. The next safe step is an action-interface diagnostic plan before scaling rollout count.
