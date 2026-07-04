# WSL Simulator Dependency Setup

This report documents the bounded WSL Python packaging setup path for simulator import readiness.

Run the dry-run risk check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\57_setup_wsl_simulator_deps.ps1
```

If the report says `decision=proceed`, Codex may execute the bounded setup with a task-local gate:

```powershell
$env:ALLOW_WSL_SIM_DEPS="1"
powershell -ExecutionPolicy Bypass -File scripts\57_setup_wsl_simulator_deps.ps1 -Execute -Packages numpy,termcolor,Pillow,mujoco,opencv-python,numba,scipy
Remove-Item Env:\ALLOW_WSL_SIM_DEPS -ErrorAction SilentlyContinue
```

The default target is:

```text
~/.venvs/tca_map_sim
```

The default package set is intentionally minimal for a first dry-run:

```text
numpy
```

The local import-only pass used the venv and installed only bounded RoboSuite import-readiness packages:

```text
numpy, termcolor, Pillow, mujoco, opencv-python, numba, scipy
```

The setup script does not use sudo or apt. It reuses the existing venv by default and bootstraps pip only if missing. Use `-ClearVenv` only when an intentional clean venv rebuild is needed. If `python3 -m venv` is unavailable, stop before package management unless a separate WSL apt risk assessment is green and no sudo password is required.

After setup, rerun:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\56_check_wsl_simulator_deps.ps1
$env:ALLOW_SIMULATOR_IMPORT_SMOKE="1"
powershell -ExecutionPolicy Bypass -File scripts\55_bounded_simulator_import_smoke.ps1
Remove-Item Env:\ALLOW_SIMULATOR_IMPORT_SMOKE -ErrorAction SilentlyContinue
```

Current local result: bounded simulator import-only smoke passed using the venv at `~/.venvs/tca_map_sim`; both `robosuite` and `libero` imported. This setup is not render evidence, not rollout evidence, and not paper-grade evidence. Render smoke, reset/step smoke, and tiny rollout diagnostics remain separate risk gates.
