# Tiny Learned-Policy LIBERO Rollout

This rung is the first bounded diagnostic that combines local SmolVLA policy inference with a real LIBERO/RoboSuite environment.

Planning-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\71_plan_tiny_learned_policy_rollout.ps1
```

Bounded execution, only after the planner says `proceed`:

```powershell
$env:ALLOW_TINY_LEARNED_POLICY_ROLLOUT="1"
powershell -ExecutionPolicy Bypass -File scripts\72_bounded_tiny_learned_policy_rollout.ps1
Remove-Item Env:\ALLOW_TINY_LEARNED_POLICY_ROLLOUT -ErrorAction SilentlyContinue
```

Default bounds:

- WSL venv: `/home/jiheon/.venvs/tca_map_sim`,
- task suite: `libero_10`,
- task count: 1,
- max steps per task: 3,
- camera size: 64,
- device: CPU,
- timeout: 30 minutes.

It must not train, use GPU jobs, download assets, run OpenVLA-OFT, run multi-seed evaluation, access tokens, or make benchmark/SOTA/paper-grade claims.

Passing this rung is tiny learned-policy diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.

## Current Local Result

Latest local execution passed as a bounded diagnostic rung:

- planner decision: `proceed`,
- wrapper result: return code 0, no timeout,
- LIBERO suite: `libero_10`,
- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`,
- tasks completed: 1,
- environment steps: 3,
- SmolVLA policy calls: 3,
- policy action shape: `[1, 6]`,
- LIBERO environment action dimension: 7,
- action conversion: pad/truncate and clip policy action to the environment action space,
- diagnostic success check: `false`,
- reward sum: `0.0`,
- total inner runtime: about 30.6 seconds.

This result proves only that the local WSL simulator, local CPU SmolVLA policy load, real LIBERO observations, policy action generation, and short environment stepping can run together in one bounded topology. It does not show task success, standard LIBERO success, counterfactual robustness, benchmark performance, or paper-grade evidence.

Next safe rung: add a tiny benchmark-metric diagnostic report that records success, reward, policy latency, action dimensions, and failure modes without changing the evidence label or claiming standard performance.
