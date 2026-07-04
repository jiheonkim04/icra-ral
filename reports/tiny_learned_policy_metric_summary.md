# Tiny Learned-Policy Metric Summary

This report defines the summary-only diagnostic layer for the first learned-policy LIBERO rollout.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\73_generate_tiny_learned_policy_metric_summary.ps1
```

The script reads `reports\tiny_learned_policy_rollout_report.json` and writes ignored runtime outputs:

- `reports\tiny_learned_policy_metric_summary_report.json`,
- `reports\tiny_learned_policy_metric_summary_report.md`.

It does not download assets, install packages, load models, run inference, run simulators, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Tracked metrics:

- source rollout pass/fail,
- task count and step count,
- policy call count,
- diagnostic success count and diagnostic success rate,
- reward sum,
- policy latency summary when available,
- policy action shape,
- environment action dimensions,
- observed failure modes.

The output evidence label remains `tiny_learned_policy_diagnostic`. It is not standard LIBERO success, not benchmark success, not counterfactual robustness evidence, not SOTA evidence, and not paper-grade evidence.

Next safe rung after this summary is a bounded small learned-policy rollout matrix planner with explicit task/step/runtime limits and no multi-seed or paper-grade claim.

## Current Local Result

Latest local summary passed as report-only diagnostic evidence:

- source rollout report exists: true,
- source rollout passed: true,
- tasks completed: 1,
- total steps: 3,
- policy calls: 3,
- diagnostic success count: 0,
- diagnostic success rate: 0.0,
- reward sum: 0.0,
- mean policy latency: about 0.157 seconds for the last recorded inference,
- policy action shape: `[1, 6]`,
- environment action dimension: 7,
- failure mode: `diagnostic_success_check_false`.

This is a clean integration metric summary, not a task-success result. The next step should plan a bounded small learned-policy rollout matrix while keeping the evidence label diagnostic/local pilot only.
