# Local Windows Run Instructions

These commands run only safe scaffold checks: tree check, preflight, dummy train smoke, and dummy eval smoke. They do not run GPU training, downloads, real adapters, or rollouts.

## Command Prompt

```bat
cd C:\Users\jiheo\tca_map
git fetch origin
git switch codex/scaffold-smoke
conda activate tca_map
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
```

## PowerShell

```powershell
cd C:\Users\jiheo\tca_map
git fetch origin
git switch codex/scaffold-smoke
conda activate tca_map
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
```

If `python` resolves to the Microsoft Store alias, pass the interpreter explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1 -Python "C:\Path\To\python.exe"
```

Optional asset setup for later real-adapter checks:

```powershell
Copy-Item configs\paths.local.yaml.example configs\paths.local.yaml
notepad configs\paths.local.yaml
$env:OPENVLA_OFT_CKPT="C:\path\to\openvla-oft"
$env:SMOLVLA_CKPT="C:\path\to\smolvla"
$env:LIBERO_ROOT="C:\path\to\LIBERO"
$env:LIBERO_DATA_ROOT="C:\path\to\libero\data"
$env:ROBOSUITE_ROOT="C:\path\to\robosuite"
$env:DATA_ROOT="C:\path\to\data"
$env:CHECKPOINT_ROOT="C:\path\to\checkpoints"
$env:HF_HOME="C:\path\to\hf_home"
```

Do not run GPU training, downloads, real OpenVLA-OFT/SmolVLA work, or real rollouts during scaffold validation.
