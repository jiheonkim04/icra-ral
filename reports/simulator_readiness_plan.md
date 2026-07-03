# Simulator Readiness Plan

This plan adds a check-only gate before any LIBERO, RoboSuite, MuJoCo, or simulator import/render/rollout work.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\43_plan_simulator_readiness.ps1
```

The planner writes ignored runtime reports:

```text
reports\simulator_readiness_plan_report.json
reports\simulator_readiness_plan_report.md
```

It does not install packages, download assets, import simulators, run render smoke, run rollouts, use GPU, train, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper-grade claims.

Proceed to a separate bounded simulator import-smoke task only if:

- `LIBERO_ROOT` exists,
- `ROBOSUITE_ROOT` exists,
- the runtime target is WSL2/Linux,
- the task is import-only and remains under the simulator readiness budget,
- no rollout, render loop, policy execution, dataset evaluation, OpenVLA-OFT execution, or paper claim is included.

Native Windows remains a planning/readiness path. Real simulator work should use WSL2/Linux unless a later risk assessment proves a native path is safe.

The source-resolution/setup path for code checkouts is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\46_prepare_libero_robosuite_sources.ps1
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

This may make `LIBERO_ROOT` and `ROBOSUITE_ROOT` path-ready, but it still does not install simulator dependencies, import simulator modules, render, rollout, train, or make paper-grade claims.
