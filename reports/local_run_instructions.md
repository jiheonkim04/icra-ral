# Local Windows Run Instructions

These commands run safe scaffold and asset-readiness checks. They do not run GPU training, downloads, real adapters, or rollouts.

## Command Prompt

```bat
cd C:\Users\jiheo\tca_map
git fetch origin
git switch codex/real-asset-setup-plan
conda activate tca_map
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
```

## PowerShell

```powershell
cd C:\Users\jiheo\tca_map
git fetch origin
git switch codex/real-asset-setup-plan
conda activate tca_map
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
```

If `python` resolves to the Microsoft Store alias, pass the interpreter explicitly for Python-backed scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1 -Python "C:\Path\To\python.exe"
```

## Configure local assets later

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

## Recommended next step

Use SmolVLA first for the eventual real-adapter smoke on the local RTX 5080 16GB system. Keep OpenVLA-OFT as the primary paper-grade baseline target, but defer full OpenVLA-OFT work until a separate approved task and preferably WSL2/Linux or larger GPU capacity.

Do not run GPU training, downloads, real OpenVLA-OFT/SmolVLA execution, or real rollouts during asset-readiness validation.
