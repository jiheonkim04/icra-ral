# Local Windows Run Instructions

These commands run safe scaffold, asset-readiness, compute-budget, and planning checks. They do not run GPU training, downloads, real adapters, heavy VLA imports, OpenVLA-OFT execution, or rollouts.

## Command Prompt

```bat
cd C:\Users\jiheo\tca_map
git fetch origin
git switch main
git pull origin main
conda activate tca_map
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\30_enforce_compute_budget.ps1
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_distributional_tca_select.py tests\test_lora_config_guards.py tests\test_tca_select.py
```

## PowerShell

```powershell
cd C:\Users\jiheo\tca_map
git fetch origin
git switch main
git pull origin main
conda activate tca_map
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\30_enforce_compute_budget.ps1
& "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe" -m pytest tests\test_distributional_tca_select.py tests\test_lora_config_guards.py tests\test_tca_select.py
```

## Cursor one-command safe check

Use this command when Cursor Agent needs to run the routine safe local validation stack without pasting each command manually:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\40_cursor_safe_local_check.ps1
```

This wrapper uses `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe` directly, writes ignored runtime reports under `reports\`, and does not run GPU jobs, downloads, rollouts, real training, heavy VLA imports, or OpenVLA-OFT.

If `python` resolves to the Microsoft Store alias, pass the interpreter explicitly for Python-backed scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\00_preflight.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\04_train_smoke.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\05_eval_smoke.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts\20_system_readiness.ps1 -Python "C:\Path\To\python.exe"
```

## Optional Planning Checks

These commands are planning/readiness-only unless a later task explicitly enables a heavy action gate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\20_system_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\21_make_asset_dirs.ps1
powershell -ExecutionPolicy Bypass -File scripts\22_plan_local_experiment_matrix.ps1
powershell -ExecutionPolicy Bypass -File scripts\23_cloud_handoff_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\24_wsl2_setup_check.ps1
```

## Configure Local Assets Later

Dry-run asset directory plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\21_make_asset_dirs.ps1
```

## SmolVLA asset prep and adapter smoke readiness

These commands are safe planning/readiness checks. They do not download assets, train, run rollouts, require LIBERO, import heavy VLA models, or execute OpenVLA-OFT.

Dry-run SmolVLA checkpoint acquisition plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
```

Dry-run SmolVLA asset setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\12_prepare_smolvla_assets.ps1
```

Create only the local directory skeleton when explicitly desired:

```powershell
$env:ALLOW_CREATE_DIRS="1"
powershell -ExecutionPolicy Bypass -File scripts\12_prepare_smolvla_assets.ps1
```

Check whether a local SmolVLA checkpoint is ready for a later separately approved load-only adapter smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

Path readiness and adapter smoke readiness are different. An empty `SMOLVLA_CKPT` directory can pass a path check, but it is not adapter-smoke-ready until config, tokenizer, and weights files are present and `HF_HOME` or `CHECKPOINT_ROOT` exists.

Plan the later load-only adapter smoke without importing SmolVLA or loading a model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

This planner writes `reports\smolvla_load_only_smoke_plan_report.json`, which is ignored by git. It refuses to run if `ALLOW_HEAVY_IMPORT=1` is already set.

The bounded execution scaffold is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\16_smolvla_load_only_smoke.ps1
```

Without `ALLOW_HEAVY_IMPORT=1`, it exits before any heavy import or model load. With the gate set in a separately approved task, it still checks runtime dependencies, local files, memory policy, and forbidden gates before any future loader path can proceed.

Check the runtime package prerequisites without installing anything:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
```

Linux/WSL equivalents:

```bash
bash scripts/14_plan_smolvla_download.sh
bash scripts/12_prepare_smolvla_assets.sh
bash scripts/13_check_smolvla_adapter_smoke.sh
```

On Windows, `C:\Users\jiheo\AppData\Local\Microsoft\WindowsApps\bash.exe` is a launcher shim, not a usable GNU Bash for script validation. Use the PowerShell scripts as the supported Windows path, or install/use Git Bash or WSL when Bash validation is needed.

Actually create the recommended directories only when explicitly approved:

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

Do not set them unless a later task explicitly calls for that specific heavy action. Keep OpenVLA-OFT large local experiments disabled.
