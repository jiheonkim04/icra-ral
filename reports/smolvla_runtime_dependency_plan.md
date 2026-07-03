# SmolVLA Runtime Dependency Plan

## Purpose

This plan covers the runtime packages needed before any actual SmolVLA load-only model smoke. It records the current local runtime state and does not authorize package upgrades, CUDA/PyTorch changes, downloads, model loading, inference, training, rollouts, simulator execution, or OpenVLA-OFT.

## Current Local Status

The current `tca_map` Python environment has the packages needed for the standing-approved bounded SmolVLA load-only smoke:

```text
torch=2.10.0+cu128
torchvision=0.25.0+cu128
transformers=4.57.6
lerobot=0.4.4
safetensors=0.8.0
accelerate=1.14.0
huggingface-hub=0.35.3
num2words=0.5.14
```

This means the local files and Python runtime dependencies are ready for the standing-approved bounded SmolVLA load-only task. `ALLOW_HEAVY_IMPORT=1` may be set only inside that task.

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

Installing, upgrading, or changing these packages is a hard-stop gate:

- PyTorch,
- CUDA-enabled PyTorch wheels,
- LeRobot,
- Transformers,
- Safetensors,
- Num2words,
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

## Completed Install Record

The explicitly approved runtime install used:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0+cu128 lerobot==0.4.4 transformers==4.57.6 safetensors==0.8.0 accelerate==1.14.0
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pip install num2words==0.5.14
```

No model loading, inference, training, rollouts, simulator execution, OpenVLA-OFT execution, token access, or dataset/checkpoint acquisition was authorized by this install.

## Recommended Upgrade Strategy Later

Prefer a separate environment task for any future upgrade that:

- pins package versions,
- avoids changing the existing CUDA driver,
- records `pip freeze` or `conda env export` before and after,
- validates `torch.cuda.is_available()` only after install approval,
- reruns `scripts\17_check_smolvla_runtime_deps.ps1`,
- reruns `scripts\16_smolvla_load_only_smoke.ps1` without inference or training.

Native Windows may work for dependency checking, but WSL2/Linux remains safer for later simulator, rollout, or larger VLA workflows.
