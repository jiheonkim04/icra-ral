param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\qlora_feasibility_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "QLoRA feasibility check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is check-only. It does not install packages, download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$executionGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF"
)

$setExecutionGates = @()
foreach ($gate in $executionGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setExecutionGates += $gate
    }
}

if ($setExecutionGates.Count -gt 0) {
    Write-Host ("Refusing QLoRA feasibility check while execution gates are set: " + ($setExecutionGates -join ", "))
    exit 20
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}

$env:TCA_MAP_QLORA_FEASIBILITY_REPORT = $reportFullPath

$script = @'
import importlib.util
import json
import os
import platform
from pathlib import Path

import yaml

from tca_map.adapters import validate_lora_policy_config

REPO_ROOT = Path.cwd()
REPORT_PATH = Path(os.environ["TCA_MAP_QLORA_FEASIBILITY_REPORT"])
QLORA_CONFIG = REPO_ROOT / "configs" / "qlora_adapter_lowcompute.yaml"


def has_module(name):
    return importlib.util.find_spec(name) is not None


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


config = load_yaml(QLORA_CONFIG)
validation = validate_lora_policy_config(config)
training = config.get("training", {})
qlora = config.get("qlora", {})
system = platform.system()
module_availability = {
    "peft": has_module("peft"),
    "bitsandbytes": has_module("bitsandbytes"),
    "transformers": has_module("transformers"),
    "accelerate": has_module("accelerate"),
    "torch": has_module("torch"),
}

blockers = []
warnings = []
if not validation["passed"]:
    blockers.extend(validation["errors"])
if not module_availability["peft"]:
    blockers.append("PEFT is not available without installing packages.")
if not module_availability["bitsandbytes"]:
    blockers.append("bitsandbytes is not available without installing packages.")
if not module_availability["transformers"]:
    blockers.append("transformers is not available.")
if not module_availability["accelerate"]:
    warnings.append("accelerate is not available; QLoRA orchestration may be limited.")
if system == "Windows":
    warnings.append("Windows-native QLoRA/bitsandbytes support may be limited; prefer WSL2/Linux if QLoRA is needed.")
if training.get("max_steps", 0) > 100:
    blockers.append("QLoRA tiny smoke must keep max_steps <= 100.")
if training.get("train_backbone") is not False:
    blockers.append("QLoRA must keep train_backbone=false.")
if qlora.get("quantization_bits") != 4:
    blockers.append("QLoRA feasibility track expects 4-bit quantization.")

tooling_present = module_availability["peft"] and module_availability["bitsandbytes"] and module_availability["transformers"]
locally_feasible_now = validation["passed"] and tooling_present and system != "Windows"

report = {
    "policy": {
        "check_only": True,
        "required_qlora_feasibility_track": True,
        "installs_performed": False,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "adapter_construction_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
        "cuda_or_pytorch_changed": False,
    },
    "environment": {
        "platform": system,
        "python": platform.python_version(),
        "windows_native": system == "Windows",
    },
    "module_availability_checked_without_importing_heavy_models": module_availability,
    "config": {
        "path": str(QLORA_CONFIG),
        "validation": validation,
        "quantization_bits": qlora.get("quantization_bits"),
        "rank": qlora.get("rank"),
        "trainable_modules": qlora.get("trainable_modules"),
        "max_steps": training.get("max_steps"),
        "train_backbone": training.get("train_backbone"),
        "full_finetune": training.get("full_finetune"),
    },
    "feasibility": {
        "tooling_present": tooling_present,
        "memory_target_gb": 14,
        "requires_cuda_or_pytorch_change": False,
        "locally_feasible_now": locally_feasible_now,
        "safe_to_run_qlora_now": False,
        "defer_to_linux_or_cloud_if_needed": bool(blockers or system == "Windows"),
    },
    "blockers": blockers,
    "warnings": warnings,
    "recommended_next_step": (
        "Update the go/no-go report with QLoRA feasibility status. Do not run QLoRA training unless tooling, memory, and bounded-runner gates are explicitly satisfied."
        if not blockers
        else "Keep QLoRA as a required feasibility track, but defer execution until blockers are resolved without unapproved installs or CUDA/PyTorch changes."
    ),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
