# SmolVLA Runtime Install Approval Request

## Purpose

This document records the approval boundary that was used for installing the runtime packages needed for a future SmolVLA load-only smoke. It is not approval for any further package changes.

The local SmolVLA checkpoint and tokenizer/processor files are present, and the Python environment now has the core runtime packages:

```text
torch==2.10.0+cu128
torchvision==0.25.0+cu128
transformers==4.57.6
lerobot==0.4.4
safetensors==0.8.0
accelerate==1.14.0
huggingface-hub==0.35.3
num2words==0.5.14
```

Installing, upgrading, or changing these packages remains a hard-stop gate because it can alter CUDA/PyTorch behavior on the RTX 5080 Windows environment.

## Planning Command

Run this check-only planner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
```

It writes the ignored runtime report:

```text
reports\smolvla_runtime_install_plan_report.json
```

The planner does not install packages, download assets, import heavy VLA models, load models, run inference, train, rollout, access tokens, or execute OpenVLA-OFT. It refuses to run if dangerous gates such as `ALLOW_DOWNLOADS=1` or `ALLOW_HEAVY_IMPORT=1` are already set.

## Completed Approved Install

The approved install used:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0+cu128 lerobot==0.4.4 transformers==4.57.6 safetensors==0.8.0 accelerate==1.14.0
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pip install num2words==0.5.14
```

No model checkpoints, datasets, simulator assets, OpenVLA-OFT assets, tokens, or secrets were part of this package install approval.

## Approval Boundary

Any later environment task must explicitly approve:

- whether the existing `tca_map` conda environment may be modified,
- exact package versions and install source,
- whether CUDA-enabled PyTorch is allowed,
- expected disk use,
- rollback plan,
- validation commands after install.

Until then, Codex may only run checkers and update planning documents.

## Safe Validation After Any Approved Install

After a separately approved install task, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
powershell -ExecutionPolicy Bypass -File scripts\16_smolvla_load_only_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest -q
```

`scripts\16_smolvla_load_only_smoke.ps1` still requires `ALLOW_HEAVY_IMPORT=1`, which may be set only inside the standing-approved bounded SmolVLA load-only task. Runtime installation alone does not authorize inference, training beyond the tiny-smoke budget, rollouts, simulator execution, OpenVLA-OFT, or paper-level claims.
