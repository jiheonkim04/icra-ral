# Zero-Action Versus SmolVLA-Action Diagnostic Comparison

This report defines a summary-only comparison between:

- the bounded zero-action LIBERO/RoboSuite diagnostic rollout, and
- the bounded reduced-scope SmolVLA learned-policy diagnostic rollout.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\79_compare_zero_action_policy_diagnostic.ps1
```

The comparison reads existing reports only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

It writes ignored runtime outputs:

- `reports\zero_action_policy_diagnostic_comparison_report.json`,
- `reports\zero_action_policy_diagnostic_comparison_report.md`.

## Current Local Result

Latest local comparison result: `proceed`.

Observed diagnostic comparison:

- zero-action diagnostic: 1 `libero_10` task, 3 zero-action steps, success `false`, reward sum `0.0`,
- SmolVLA-action diagnostic: 1 matching `libero_10` task, 10 policy steps, success `false`, reward sum `0.0`,
- SmolVLA action was nontrivial, with max absolute action about `0.793093`,
- learned-policy action did not outperform zero-action on this diagnostic comparison,
- simulator reset/step/render plumbing is less likely to be the primary blocker than the action/state interface.

Interpretation:

- if both zero-action and learned-policy diagnostics have success `false` and reward `0.0`, but learned-policy actions are nontrivial, the next step is not rollout scaling,
- if zero-action simulator plumbing passes, the zero reward is less likely to be caused by a basic reset/step/render failure,
- if learned-policy actions remain 6D while the environment accepts 7D, with a constant gripper component, the next work should be an explicit action/state adapter patch plan.

This is diagnostic/local-pilot evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.
