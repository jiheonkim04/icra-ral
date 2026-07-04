# Bounded Learned-Policy Rollout Matrix Plan

This report defines the planning-only gate after the first tiny learned-policy LIBERO diagnostic rollout.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1
```

The planner reads `reports\tiny_learned_policy_metric_summary_report.json` and writes ignored runtime outputs:

- `reports\bounded_learned_policy_rollout_matrix_plan_report.json`,
- `reports\bounded_learned_policy_rollout_matrix_plan_report.md`.

It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Decision logic:

- If the metric summary is missing or failed, stop.
- If the learned-policy topology passed but diagnostic success rate is `0.0`, reduce scope to a one-task longer diagnostic before any multi-task rollout matrix.
- If the diagnostic success rate is positive, allow a bounded small matrix plan capped by the local risk budget.

Default bounds:

- multi-task cap: at most 3 tasks for the first matrix,
- reduced-scope cap: 1 task,
- max steps per task: 10,
- expected runtime: no more than 30 minutes,
- device: CPU first,
- evidence label: diagnostic/local pilot only.

The planner is deliberately conservative because the current one-task, three-step learned-policy diagnostic had diagnostic success rate `0.0` and reward sum `0.0`. A clean integration path is not yet manipulation success.

## Current Local Result

Latest local planner result: `reduce_scope`.

Reason: the topology passed, but the metric summary recorded diagnostic success rate `0.0` and reward sum `0.0`.

Recommended next rung:

- create a separately gated one-task longer diagnostic runner,
- use task-local `ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1` only for that runner,
- keep task count at 1,
- increase max steps per task to 10,
- keep expected runtime within 30 minutes,
- keep evidence label diagnostic/local pilot only.

The planner explicitly keeps the bounded small multi-task matrix blocked until a positive diagnostic success signal appears or a separate risk/research decision justifies another reduced-scope probe.
