# SmolVLA Runtime Dependency Plan

## Purpose

This plan covers the runtime packages needed before any actual SmolVLA load-only model smoke. It is an environment planning document only. It does not authorize package installation, CUDA/PyTorch changes, downloads, model loading, inference, training, rollouts, simulator execution, or OpenVLA-OFT.

## Current Local Status

The current `tca_map` Python environment is missing the packages needed for a real SmolVLA load-only smoke:

```text
torch=false
transformers=false
lerobot=false
safetensors=false
```

This means the local files are ready, but the runtime is not ready for actual model loading.

## Check Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
```

It writes an ignored runtime report:

```text
reports\smolvla_runtime_deps_report.json
```

## Hard-Stop Boundary

Installing or changing these packages is a hard-stop gate:

- PyTorch,
- CUDA-enabled PyTorch wheels,
- LeRobot,
- Transformers,
- Safetensors,
- Accelerate,
- any CUDA toolkit or driver-level dependency.

Do not install or upgrade them automatically. A later explicit environment task must define exact versions, expected disk usage, CUDA compatibility, rollback plan, and validation commands.

The install approval boundary is tracked in:

```text
reports\smolvla_runtime_install_request.md
```

The check-only planner is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
```

## Recommended Install Strategy Later

Prefer a separate environment task that:

- pins package versions,
- avoids changing the existing CUDA driver,
- records `pip freeze` or `conda env export` before and after,
- validates `torch.cuda.is_available()` only after install approval,
- reruns `scripts\17_check_smolvla_runtime_deps.ps1`,
- reruns `scripts\16_smolvla_load_only_smoke.ps1` without inference or training.

Native Windows may work for dependency checking, but WSL2/Linux remains safer for later simulator, rollout, or larger VLA workflows.
