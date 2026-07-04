# Simulator Render/Reset-Step Plan

This plan records the next simulator-readiness boundary after bounded import-only smoke.

Run the planning-only check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\58_plan_simulator_render_reset.ps1
```

The planner requires a passed bounded simulator import-only report before it can recommend a later render-smoke branch:

```text
reports\bounded_simulator_import_smoke_report.json
```

It does not render, reset or step simulator environments, rollout, train, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper-grade claims.

Readiness states:

- `ready_for_bounded_render_smoke_plan=true` means import-only readiness exists and a separate render-smoke risk gate may be created.
- `ready_for_bounded_reset_step_smoke_plan=true` requires a future bounded render-smoke report to pass first.
- `ready_for_rollout=false` remains false here. Rollout requires separate render, reset/step, and tiny diagnostic rollout gates.

Execution gates such as `ALLOW_SIMULATOR_RENDER_SMOKE=1`, `ALLOW_SIMULATOR_RESET_STEP=1`, `ALLOW_TINY_ROLLOUT=1`, `ALLOW_HEAVY_IMPORT=1`, or `ALLOW_TINY_TRAINING=1` must not be set while running this planner.
