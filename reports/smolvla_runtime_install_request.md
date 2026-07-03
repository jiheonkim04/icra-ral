# SmolVLA Runtime Install Approval Request

## Purpose

This document prepares the approval boundary for installing the runtime packages needed for a future SmolVLA load-only smoke. It is not approval to install anything.

The current local SmolVLA checkpoint and tokenizer/processor files are present, but the Python environment is missing core runtime packages:

```text
torch
transformers
lerobot
safetensors
```

Installing or changing these packages is a hard-stop gate because it can alter CUDA/PyTorch behavior on the RTX 5080 Windows environment.

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

## Packages Requiring Explicit Approval

Required before a real SmolVLA load-only smoke:

- `torch`,
- `transformers`,
- `lerobot`,
- `safetensors`.

Likely optional but useful:

- `accelerate`,
- `huggingface_hub`.

No package should be installed or upgraded automatically. A later approved install task should pin versions, preserve an environment snapshot, avoid token access, avoid dataset/checkpoint downloads, and validate with repository checkers.

## Approval Boundary

The next environment task must explicitly approve:

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

`scripts\16_smolvla_load_only_smoke.ps1` still requires a separate `ALLOW_HEAVY_IMPORT=1` gate before any actual model load path can proceed. Runtime installation alone does not authorize model loading, inference, training, rollouts, simulator execution, OpenVLA-OFT, or paper-level claims.
