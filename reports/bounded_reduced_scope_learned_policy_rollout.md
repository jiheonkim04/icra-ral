# Bounded Reduced-Scope Learned-Policy Rollout

This rung runs the conservative follow-up selected by `scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1`.

Command:

```powershell
$env:ALLOW_BOUNDED_LEARNED_POLICY_MATRIX="1"
powershell -ExecutionPolicy Bypass -File scripts\75_bounded_reduced_scope_learned_policy_rollout.ps1
Remove-Item Env:\ALLOW_BOUNDED_LEARNED_POLICY_MATRIX -ErrorAction SilentlyContinue
```

Default bounds:

- WSL venv: `/home/jiheon/.venvs/tca_map_sim`,
- task suite: `libero_10`,
- task count: 1,
- max steps per task: 10,
- camera size: 64,
- device: CPU,
- timeout: 30 minutes.

The runner does not train, use GPU jobs, download assets, execute OpenVLA-OFT, run multi-seed evaluation, or make benchmark/SOTA/paper-grade claims.

This is a reduced-scope diagnostic/local-pilot rung. It exists because the first learned-policy diagnostic passed the integration path but had diagnostic success rate `0.0` and reward sum `0.0`. Passing this runner means only that the longer single-task diagnostic executed within bounds. It does not establish standard LIBERO success or paper-grade evidence.

## Current Local Result

Latest local execution passed as a bounded reduced-scope diagnostic:

- planner decision: `reduce_scope`,
- task suite: `libero_10`,
- task count: 1,
- max steps per task: 10,
- completed steps: 10,
- policy calls: 10,
- policy action shape: `[1, 6]`,
- LIBERO environment action dimension: 7,
- diagnostic success check: `false`,
- reward sum: `0.0`,
- inner runtime: about 35.8 seconds,
- no downloads, no installs, no training, no GPU job, no OpenVLA-OFT, no multi-seed evaluation, no token access, and no paper claim.

Interpretation: the longer one-task policy-control loop is stable, but it still does not solve the selected LIBERO task. This is evidence for execution readiness and a weak policy-performance diagnostic, not benchmark success.
