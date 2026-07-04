# Bounded Simulator Import Smoke

This gate is the first simulator-adjacent step after the LIBERO offline bounded pilot report.

Run the planning gate first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\43_plan_simulator_readiness.ps1
```

Run the bounded import-only smoke only after a green risk assessment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\41_risk_assess_task.ps1 -Task "Bounded simulator import smoke" -Category simulator -Source "official local LIBERO and RoboSuite source checkouts" -TargetPath "C:\assets\repos" -ExpectedSizeGb 0 -ExpectedRuntimeMinutes 2 -ExpectedRamGb 2 -ExpectedVramGb 0 -SimulatorInstalled -OfficialSource
$env:ALLOW_SIMULATOR_IMPORT_SMOKE="1"
powershell -ExecutionPolicy Bypass -File scripts\55_bounded_simulator_import_smoke.ps1
Remove-Item Env:\ALLOW_SIMULATOR_IMPORT_SMOKE -ErrorAction SilentlyContinue
```

The script may import only the local WSL-visible `robosuite` and `libero` Python packages. It must not install packages, download assets, render, create environments, step simulators, rollout policies, train, run GPU jobs, import heavy VLA models, execute OpenVLA-OFT, access tokens, upload externally, or make paper-grade claims.

Outputs are ignored runtime reports:

```text
reports\bounded_simulator_import_smoke_report.json
reports\bounded_simulator_import_smoke_report.md
```

Passing this gate means package import plumbing is visible in WSL/Linux. It is not rollout success, not standard success, and not paper-grade evidence. Render smoke and rollouts require separate risk assessments and separate bounded scripts.
