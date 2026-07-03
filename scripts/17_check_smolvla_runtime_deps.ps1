param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\smolvla_runtime_deps_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$script = @'
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

required = ["torch", "transformers", "lerobot", "safetensors"]
optional = ["accelerate", "huggingface_hub", "numpy", "yaml"]

def has_module(name):
    return importlib.util.find_spec(name) is not None

def nvidia():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return {"available": False}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False}
    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    return {
        "available": True,
        "gpu_name": parts[0] if len(parts) > 0 else None,
        "memory_total_mb": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
        "driver_version": parts[2] if len(parts) > 2 else None,
    }

required_status = {name: has_module(name) for name in required}
optional_status = {name: has_module(name) for name in optional}
missing_required = [name for name, present in required_status.items() if not present]

report = {
    "policy": {
        "check_only": True,
        "installs_performed": False,
        "downloads_performed": False,
        "heavy_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
    },
    "python": {
        "executable": sys.executable,
        "version": platform.python_version(),
        "platform": platform.platform(),
    },
    "runtime_dependencies": {
        "required": required_status,
        "optional": optional_status,
        "missing_required": missing_required,
        "ready_for_load_only_runtime": not missing_required,
    },
    "gpu": nvidia(),
    "hard_stop": {
        "installing_large_packages_requires_user_approval": bool(missing_required),
        "reason": "Missing runtime packages require explicit install/CUDA/PyTorch approval." if missing_required else None,
    },
    "recommended_next_step": (
        "Stop before installing PyTorch/Transformers/LeRobot. Prepare an environment install plan for explicit approval."
        if missing_required
        else "Runtime dependencies appear present. A bounded load-only smoke may be considered under ALLOW_HEAVY_IMPORT=1."
    ),
}

Path(os.environ["TCA_MAP_RUNTIME_DEPS_REPORT"]).parent.mkdir(parents=True, exist_ok=True)
Path(os.environ["TCA_MAP_RUNTIME_DEPS_REPORT"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
'@

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}
$env:TCA_MAP_RUNTIME_DEPS_REPORT = $reportFullPath

Write-Host "SmolVLA runtime dependency check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script checks module availability only. It does not install packages, download assets, import heavy VLA models, load models, train, rollout, or execute OpenVLA-OFT."

$script | & $Python -
exit $LASTEXITCODE
