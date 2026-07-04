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

## WSL Simulator Dependency Ladder Standing Approval

If WSL path and `python3` probes pass but Python packaging or import dependencies are missing, Codex may continue autonomously through a bounded WSL dependency ladder after a green risk assessment. This does not authorize simulator rollout, benchmark evaluation, OpenVLA-OFT, GPU training, or paper claims.

Allowed autonomous WSL dependency steps:

- inspect WSL distro status and Python packaging status,
- install minimal WSL Python packaging tools only if safe,
- create or reuse `~/.venvs/tca_map_sim`,
- install minimal import-readiness dependencies such as `numpy`, `scipy`, `h5py`, `pyyaml`, `tqdm`, `gymnasium` or `gym` if required, `mujoco` if required for import/render checks, and `robosuite`/LIBERO dependencies only when official docs and budget are green,
- rerun bounded simulator import smoke.

Allowed apt packages are limited to `python3-pip`, `python3-venv`, `python3-dev` if needed, `build-essential` only if required for Python package builds, `git` if missing, and `curl` or `wget` only for official setup checks.

Stop before sudo password input, token/secret/login, paid service or license click-through, CUDA driver/toolkit install, major graphics-stack changes, Windows driver changes, OpenVLA-OFT download/import/load/execution, full fine-tuning, training over 30 minutes, VRAM over 14GB, downloads beyond approved budget, rollout beyond tiny diagnostic limits, benchmark/paper-grade claims, multi-seed experiments, external upload/submission/publishing, or deletion outside approved repo/cache cleanup.

Readiness progression:

1. Import readiness: import `numpy`, `libero`, `robosuite`, and `mujoco` if installed/needed. No rendering, rollout, policy evaluation, training, or paper claim.
2. Bounded render smoke: only after import readiness passes and a render risk assessment is green; runtime <=10 minutes, headless/offscreen preferred.
3. Bounded reset/step smoke: only after import/render readiness passes; at most one environment and at most 5 reset/step attempts, runtime <=10 minutes.
4. Bounded tiny rollout diagnostic: only after earlier stages pass; task count <=5, runtime <=30 minutes, no OpenVLA-OFT, no training, no multi-seed, no paper claim.

The source-resolution/setup path for code checkouts is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\46_prepare_libero_robosuite_sources.ps1
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

This may make `LIBERO_ROOT` and `ROBOSUITE_ROOT` path-ready, but it still does not install simulator dependencies, import simulator modules, render, rollout, train, or make paper-grade claims.
