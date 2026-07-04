# Bounded Simulator Reset/Step Smoke

This document describes the bounded reset/step smoke rung after WSL import-only and tiny MuJoCo render smoke pass.

Run the script only after `scripts\58_plan_simulator_render_reset.ps1` reports `ready_for_bounded_reset_step_smoke_plan=true`:

```powershell
$env:ALLOW_SIMULATOR_RESET_STEP="1"
powershell -ExecutionPolicy Bypass -File scripts\61_bounded_simulator_reset_step_smoke.ps1
Remove-Item Env:\ALLOW_SIMULATOR_RESET_STEP -ErrorAction SilentlyContinue
```

Scope:

- uses the existing WSL venv at `/home/jiheon/.venvs/tca_map_sim`,
- uses a tiny in-memory MuJoCo XML model,
- performs bounded `mj_resetData`, `mj_forward`, and a small number of `mj_step` calls,
- does not create LIBERO or RoboSuite environments,
- does not run policy inference, rollout, training, GPU jobs, downloads, heavy VLA imports, OpenVLA-OFT, token access, or paper claims.

This is reset/step plumbing evidence only. It is not LIBERO rollout success, benchmark success, standard success, or paper-grade evidence.
