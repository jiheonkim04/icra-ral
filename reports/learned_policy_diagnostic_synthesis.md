# Learned-Policy Diagnostic Synthesis

This report records the report-only synthesis after the bounded learned-policy diagnostic ladder:

- zero-action comparison,
- adapter strategy,
- action scale,
- prompt format,
- camera source,
- state sufficiency.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\92_generate_learned_policy_diagnostic_synthesis.ps1
```

The synthesis reads existing diagnostic reports only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Runtime outputs are ignored:

- `reports\learned_policy_diagnostic_synthesis_report.json`,
- `reports\learned_policy_diagnostic_synthesis_report.md`.

## Current Local Result

Latest synthesis result: `no_go_rollout_scaling`.

The diagnostic ladder is complete:

- zero-action comparison: passed, learned-policy reward `0.0`, diagnostic success `0.0`,
- adapter strategy: passed, best variant `policy_6d_delta_pose_plus_gripper_close`, reward `0.0`, diagnostic success `0.0`,
- action scale: passed, best variant `1.0`, reward `0.0`, diagnostic success `0.0`,
- prompt format: passed, best variant `bddl_language_period`, reward `0.0`, diagnostic success `0.0`,
- camera source: passed, best variant `all_agentview`, reward `0.0`, diagnostic success `0.0`,
- state sufficiency: passed, best variant `eef_pos_zero_rot`, reward `0.0`, diagnostic success `0.0`.

Synthesis decision:

- positive diagnostic signal found: false,
- rollout scaling ready: false,
- paper-grade claim ready: false,
- benchmark claim: false,
- SOTA claim: false.

No-go reason:

```text
All bounded learned-policy diagnostic axes completed, but none produced nonzero reward or diagnostic success. Every diagnostic report keeps ready_for_rollout_scaling=false.
```

Recommended next step:

Create a bounded environment-policy compatibility audit focused on task/checkpoint alignment, action convention, and observation convention before another one-task diagnostic. Do not scale learned-policy rollouts from the current evidence.
