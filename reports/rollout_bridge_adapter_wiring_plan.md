# Rollout Bridge Adapter Wiring Plan

This report defines the planning gate before pure interface adapters are wired into the learned-policy LIBERO rollout bridge.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\81_plan_rollout_bridge_adapter_wiring.ps1
```

The planner reads existing reports and source files only. It does not download assets, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

The planner is green only if:

- the action/state adapter patch plan passed,
- the synthetic single-sample smoke recorded adapter metadata,
- pure adapter helpers exist,
- the rollout bridge still needs explicit wiring,
- no execution gates are set.

If green, the next implementation may wire pure adapters into the rollout bridge and update reports/tests. It must still avoid running rollouts until a separate bounded diagnostic rollout gate is green.

## Current Local Result

Latest local planner result: `proceed`.

The planner found:

- synthetic single-sample adapter metadata is recorded,
- pure action/state/image adapter helpers exist,
- rollout bridge still has implicit action padding,
- rollout bridge still has state truncation,
- rollout bridge still has local fallback image alias logic,
- rollout bridge adapter wiring is ready for implementation,
- rollout execution is not ready and remains blocked.
