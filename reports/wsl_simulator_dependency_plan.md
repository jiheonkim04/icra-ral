# WSL Simulator Dependency Plan

This check records why the bounded simulator import-only smoke can or cannot run in WSL.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\56_check_wsl_simulator_deps.ps1
```

The checker probes:

- `wsl` availability,
- WSL `python3`,
- WSL `python3 -m pip`,
- WSL `python3 -m ensurepip`,
- WSL `numpy` import,
- whether the previous bounded import smoke reported missing modules.

It performs no installs, downloads, render smoke, rollouts, simulator environment steps, GPU jobs, training, heavy VLA imports, OpenVLA-OFT execution, token access, or paper-grade claims.

Current local result:

```text
global WSL python3 still lacks pip, ensurepip, and numpy, but the selected venv at ~/.venvs/tca_map_sim has pip and numpy.
```

Because global `pip` and `ensurepip` can be missing, dependency installation should stay inside the selected venv unless a separate apt/system risk assessment is green. The bounded venv setup path is documented in:

```text
reports\wsl_simulator_dependency_setup.md
```

The preferred setup uses `python3 -m venv ~/.venvs/tca_map_sim` and installs only minimal import-readiness Python dependencies there. It does not use sudo or apt. Apt-based setup remains separate and must stop if sudo password, token/license/payment, CUDA/driver, graphics-stack, OpenVLA-OFT, render, rollout, or paper-claim gates appear.

The latest bounded import-only smoke selected the venv Python and imported both `robosuite` and `libero`. This clears import-only readiness only. Render smoke, reset/step smoke, rollout, and benchmark claims remain separate risk gates.
