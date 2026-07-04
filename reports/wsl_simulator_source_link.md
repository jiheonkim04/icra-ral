# WSL Simulator Source Link

This report documents the bounded local source-link step for WSL simulator import readiness.

Run the dry-run check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\60_link_wsl_simulator_sources.ps1
```

If the report says `decision=proceed`, Codex may execute the bounded source link with a task-local gate:

```powershell
$env:ALLOW_WSL_SIM_SOURCE_LINK="1"
powershell -ExecutionPolicy Bypass -File scripts\60_link_wsl_simulator_sources.ps1 -Execute
Remove-Item Env:\ALLOW_WSL_SIM_SOURCE_LINK -ErrorAction SilentlyContinue
```

Scope:

- uses the existing WSL venv at `/home/jiheon/.venvs/tca_map_sim`,
- does not create a repo-local `.venv`,
- links existing local source checkouts from `C:\assets\repos\robosuite` and `C:\assets\repos\LIBERO`,
- uses editable `pip install --no-index --no-deps --no-build-isolation -e ...`,
- writes a venv-local `.pth` entry for LIBERO's nested source layout,
- writes a noninteractive WSL `~/.libero/config.yaml` that points at the local LIBERO source and `LIBERO_DATA_ROOT`,
- performs no package downloads,
- performs no render, reset/step, rollout, training, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, token access, or paper claims.

This step is useful when `robosuite` imports only with an explicit source path but is missing from the active WSL venv, or when `libero` is path-visible but first import prompts for dataset configuration.
