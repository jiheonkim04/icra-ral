# Bounded Simulator Render Smoke

This report documents the tiny simulator render-smoke boundary after bounded import-only readiness.

Run the dry planner first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\58_plan_simulator_render_reset.ps1
```

Run the bounded render smoke only when that planner reports:

```text
ready_for_bounded_render_smoke_plan=true
```

Task-local command:

```powershell
$env:ALLOW_SIMULATOR_RENDER_SMOKE="1"
powershell -ExecutionPolicy Bypass -File scripts\59_bounded_simulator_render_smoke.ps1
Remove-Item Env:\ALLOW_SIMULATOR_RENDER_SMOKE -ErrorAction SilentlyContinue
```

Scope:

- uses the selected WSL Python venv,
- imports `mujoco`,
- attempts one tiny 64x64 offscreen render with `MUJOCO_GL=osmesa`,
- does not create, reset, or step LIBERO/RoboSuite environments,
- does not rollout,
- does not train,
- does not run GPU jobs,
- does not download or install packages,
- does not import heavy VLA models,
- does not execute OpenVLA-OFT,
- does not access tokens,
- does not make paper-grade claims.

If the render fails because OSMesa or another graphics dependency is missing, stop before system graphics changes. Reset/step smoke and rollout remain separate later gates.

Current local result:

```text
bounded_simulator_render_smoke_passed=false
reason=bounded MuJoCo offscreen render probe failed; likely OSMesa/offscreen GL is unavailable or misconfigured
error=AttributeError: 'NoneType' object has no attribute 'glGetError'
```

No packages were installed, no system graphics changes were made, no reset/step was run, and no rollout was run.
