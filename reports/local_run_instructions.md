# Local Windows Run Instructions

From PowerShell in your local clone of `jiheonkim04/icra-ral`:

```powershell
cd C:\Users\jiheo\tca_map

git fetch origin
git switch codex/scaffold-smoke

python --version
powershell -ExecutionPolicy Bypass -File scripts/99_tree_check.ps1
powershell -ExecutionPolicy Bypass -File scripts/00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts/04_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/05_eval_smoke.ps1
```

If `python` resolves to the Microsoft Store alias, use your real Python executable explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/00_preflight.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts/04_train_smoke.ps1 -Python "C:\Path\To\python.exe"
powershell -ExecutionPolicy Bypass -File scripts/05_eval_smoke.ps1 -Python "C:\Path\To\python.exe"
```

Optional asset setup for later real-adapter checks:

```powershell
Copy-Item configs/paths.local.yaml.example configs/paths.local.yaml
notepad configs/paths.local.yaml
$env:OPENVLA_OFT_CKPT="C:\path\to\openvla-oft"
$env:SMOLVLA_CKPT="C:\path\to\smolvla"
$env:LIBERO_ROOT="C:\path\to\LIBERO"
$env:LIBERO_DATA_ROOT="C:\path\to\libero\data"
```

Do not run GPU training, downloads, or real rollouts during the scaffold validation step.
