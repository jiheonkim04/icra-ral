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

Current local blocker:

```text
robosuite import in WSL requires numpy, but WSL python3 does not have numpy, pip, or ensurepip.
```

Because `pip` and `ensurepip` are missing, dependency installation is not automatic. Any apt-based or system-level WSL setup must be a separate risk-assessed task and should not run as part of this checker.
