# Local Windows Run Instructions

These commands run safe scaffold, asset-readiness, and planning checks. They do not run GPU training, downloads, real adapters, or rollouts.

## Command Prompt

```bat
cd C:\Users\jiheo\tca_map
git fetch origin
git switch codex/local-papergrade-runner
conda activate tca_map
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\20_system_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\21_make_asset_dirs.ps1
powershell -ExecutionPolicy Bypass -File scripts\22_plan_local_experiment_matrix.ps1
powershell -ExecutionPolicy Bypass -File scripts\23_cloud_handoff_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\24_wsl2_setup_check.ps1
```

## PowerShell

```powershell
cd C:\Users\jiheo\tca_map
git fetch origin
git switch codex/local-papergrade-runner
conda activate tca_map
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\20_system_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\21_make_asset_dirs.ps1
powershell -ExecutionPolicy Bypass -File scripts\22_plan_local_experiment_matrix.ps1
powershell -ExecutionPolicy Bypass -File scripts\23_cloud_handoff_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\24_wsl2_setup_check.ps1
```

If `python` resolves to the Microsoft Store alias, pass the interpreter explicitly for Python-backed scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\20_system_readiness.ps1 -Python "C:\Path\To\python.exe"
```

## Path to Paper-Grade Experiments Without Leaving Home

1. Local Windows scaffold validation: run tree check, preflight, dummy smoke, pytest, real asset readiness, and system readiness.
2. WSL2/Linux rollout setup: run `scripts\24_wsl2_setup_check.ps1`, install Ubuntu manually if needed, then check `wsl nvidia-smi`.
3. SmolVLA-first local smoke: configure `SMOLVLA_CKPT`, `HF_HOME`, and `CHECKPOINT_ROOT`; model execution remains a later approved task.
4. Small local rollout: configure `LIBERO_ROOT`, `LIBERO_DATA_ROOT`, and `ROBOSUITE_ROOT`; run rollouts only after a separate explicit approval.
5. OpenVLA-OFT frozen smoke: configure `OPENVLA_OFT_CKPT`, `HF_HOME`, and `CHECKPOINT_ROOT`; avoid full local fine-tuning.
6. Cloud/remote GPU handoff for large baseline: regenerate `reports\cloud_handoff_manifest.md` and `.json`, then use remote Linux commands from the manifest.

## Configure local assets later

Dry-run asset directory plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\21_make_asset_dirs.ps1
```

Actually create the recommended directories:

```powershell
$env:ALLOW_CREATE_DIRS="1"
powershell -ExecutionPolicy Bypass -File scripts\21_make_asset_dirs.ps1
```

Configure paths:

```powershell
Copy-Item configs\paths.local.yaml.example configs\paths.local.yaml
notepad configs\paths.local.yaml
```

Or set environment variables:

```powershell
$env:OPENVLA_OFT_CKPT="C:\assets\checkpoints\openvla-oft"
$env:SMOLVLA_CKPT="C:\assets\checkpoints\smolvla"
$env:LIBERO_ROOT="C:\assets\repos\LIBERO"
$env:LIBERO_DATA_ROOT="C:\assets\data\libero"
$env:ROBOSUITE_ROOT="C:\assets\repos\robosuite"
$env:DATA_ROOT="C:\assets\data"
$env:CHECKPOINT_ROOT="C:\assets\checkpoints"
$env:HF_HOME="C:\assets\hf_home"
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
```

## Heavy Action Gates

These gates must remain unset for planning-only tasks:

```powershell
$env:ALLOW_DOWNLOADS="1"
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_GPU_TRAINING="1"
$env:ALLOW_ROLLOUTS="1"
$env:ALLOW_CLOUD_HANDOFF="1"
```

Do not set them unless a later task explicitly calls for that specific heavy action.
