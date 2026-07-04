param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\smolvla_runtime_install_plan_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "SmolVLA runtime install planning check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script plans a runtime install risk assessment only. It does not install packages, download assets, import heavy VLA models, load models, train, rollout, or execute OpenVLA-OFT."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$dangerousGates = @(
    "ALLOW_INSTALLS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to run planning check while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}
$env:TCA_MAP_RUNTIME_INSTALL_PLAN_REPORT = $reportFullPath

$script = @'
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

package_plan = [
    {
        "distribution": "torch",
        "module": "torch",
        "required_for": "SmolVLA load-only model construction and CUDA/CPU tensor runtime",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "transformers",
        "module": "transformers",
        "required_for": "SmolVLA/SmolVLM tokenizer, processor, and model config classes",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "lerobot",
        "module": "lerobot",
        "required_for": "SmolVLA policy implementation",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "safetensors",
        "module": "safetensors",
        "required_for": "local safetensors checkpoint reads",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "num2words",
        "module": "num2words",
        "required_for": "SmolVLM processor text utility used during local SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "draccus",
        "module": "draccus",
        "required_for": "LeRobot configuration parsing used by SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "datasets",
        "module": "datasets",
        "required_for": "LeRobot data/config utilities imported during SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "imageio",
        "module": "imageio",
        "required_for": "LeRobot video/image utilities imported during SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "diffusers",
        "module": "diffusers",
        "required_for": "LeRobot diffusion policy components imported during SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "pyserial",
        "module": "serial",
        "required_for": "LeRobot device utilities imported during SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "deepdiff",
        "module": "deepdiff",
        "required_for": "LeRobot config comparison utilities imported during SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "av",
        "module": "av",
        "required_for": "LeRobot video utilities imported during SmolVLA policy construction",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "einops",
        "module": "einops",
        "required_for": "SmolVLA tensor rearrangement layers",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "accelerate",
        "module": "accelerate",
        "required_for": "optional memory/device placement support",
        "install_requires_risk_assessment": True,
    },
    {
        "distribution": "huggingface_hub",
        "module": "huggingface_hub",
        "required_for": "optional local cache utilities; no network use without a green download risk assessment",
        "install_requires_risk_assessment": True,
    },
]

def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

def run_small_command(args):
    try:
        result = subprocess.run(
            args,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

packages = []
for item in package_plan:
    version = installed_version(item["distribution"])
    packages.append(
        {
            **item,
            "installed": version is not None,
            "installed_version": version,
        }
    )

missing_required = [
    item["distribution"]
    for item in packages
    if item["distribution"] in {
        "torch",
        "transformers",
        "lerobot",
        "safetensors",
        "num2words",
        "draccus",
        "datasets",
        "imageio",
        "diffusers",
        "pyserial",
        "deepdiff",
        "av",
        "einops",
    }
    and not item["installed"]
]

report = {
    "policy": {
        "planning_only": True,
        "risk_assessment_required_before_install": True,
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
    "pip": run_small_command([sys.executable, "-m", "pip", "--version"]),
    "nvidia_smi": run_small_command(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    ),
    "packages": packages,
    "missing_required": missing_required,
    "ready_for_install_risk_assessment": bool(missing_required),
    "risk_gate": {
        "install_requires_risk_assessment": True,
        "reason": "Installing or changing PyTorch/CUDA/LeRobot/Transformers/Safetensors/Draccus or LeRobot import dependencies requires a green package/runtime risk assessment.",
    },
    "recommended_next_step": (
        "Run a pinned SmolVLA runtime install risk assessment before installing packages."
        if missing_required
        else "Runtime packages appear installed; rerun scripts/17_check_smolvla_runtime_deps.ps1 and then continue to the risk-assessed bounded load-only smoke."
    ),
}

Path(os.environ["TCA_MAP_RUNTIME_INSTALL_PLAN_REPORT"]).parent.mkdir(parents=True, exist_ok=True)
Path(os.environ["TCA_MAP_RUNTIME_INSTALL_PLAN_REPORT"]).write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)
print(json.dumps(report, indent=2))
'@

$script | & $Python -
exit $LASTEXITCODE
