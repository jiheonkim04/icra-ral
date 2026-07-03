param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\lora_tiny_smoke_scaffold_report.json",
    [int]$MaxSteps = 16,
    [int]$MaxRuntimeSeconds = 900
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LoRA tiny smoke scaffold planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not construct adapters, train, download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 100) {
    Write-Host "Refusing: MaxSteps must be between 1 and 100 for LoRA tiny-smoke planning."
    exit 11
}

if ($MaxRuntimeSeconds -lt 1 -or $MaxRuntimeSeconds -gt 900) {
    Write-Host "Refusing: MaxRuntimeSeconds must be between 1 and 900."
    exit 12
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
    Write-Host ("Refusing to plan LoRA tiny smoke while execution gates are set: " + ($setExecutionGates -join ", "))
    exit 20
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}

$env:TCA_MAP_LORA_TINY_SMOKE_SCAFFOLD_REPORT = $reportFullPath
$env:TCA_MAP_LORA_TINY_SMOKE_MAX_STEPS = [string]$MaxSteps
$env:TCA_MAP_LORA_TINY_SMOKE_MAX_RUNTIME_SECONDS = [string]$MaxRuntimeSeconds

$script = @'
import json
import os
from pathlib import Path

import yaml

from tca_map.adapters import validate_lora_policy_config

REPO_ROOT = Path.cwd()
REPORT_PATH = Path(os.environ["TCA_MAP_LORA_TINY_SMOKE_SCAFFOLD_REPORT"])
MAX_STEPS = int(os.environ["TCA_MAP_LORA_TINY_SMOKE_MAX_STEPS"])
MAX_RUNTIME_SECONDS = int(os.environ["TCA_MAP_LORA_TINY_SMOKE_MAX_RUNTIME_SECONDS"])
MAX_STEPS_CAP = 100
MAX_RUNTIME_CAP_SECONDS = 900

CONFIGS = {
    "lora": REPO_ROOT / "configs" / "lora_adapter_lowcompute.yaml",
    "qlora": REPO_ROOT / "configs" / "qlora_adapter_lowcompute.yaml",
}


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


configs = {name: load_yaml(path) for name, path in CONFIGS.items()}
validations = {name: validate_lora_policy_config(config) for name, config in configs.items()}

hard_stop_reasons = []
for name, validation in validations.items():
    if not validation["passed"]:
        hard_stop_reasons.extend(f"{name} config: {error}" for error in validation["errors"])

if MAX_STEPS > MAX_STEPS_CAP:
    hard_stop_reasons.append(f"requested max steps exceeds {MAX_STEPS_CAP}")
if MAX_RUNTIME_SECONDS > MAX_RUNTIME_CAP_SECONDS:
    hard_stop_reasons.append(f"requested runtime exceeds {MAX_RUNTIME_CAP_SECONDS} seconds")

report = {
    "policy": {
        "planning_only": True,
        "scaffold_only": True,
        "required_lora_track": True,
        "required_qlora_feasibility_track": True,
        "future_execution_requires_allow_tiny_training": True,
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
    },
    "bounds": {
        "requested_max_steps": MAX_STEPS,
        "max_steps_cap": MAX_STEPS_CAP,
        "requested_max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "max_runtime_cap_seconds": MAX_RUNTIME_CAP_SECONDS,
        "max_vram_target_gb": 14,
        "batch_size": 1,
        "freeze_backbone": True,
        "full_finetune": False,
        "rollouts_allowed": False,
        "simulator_allowed": False,
        "openvla_oft_allowed": False,
    },
    "configs": {
        name: {
            "path": str(CONFIGS[name]),
            "run_mode": configs[name].get("run", {}).get("mode"),
            "config_max_steps": configs[name].get("training", {}).get("max_steps"),
            "validation": validations[name],
        }
        for name in CONFIGS
    },
    "future_runner_requirements": [
        "Require ALLOW_TINY_TRAINING=1 only inside the bounded LoRA tiny-smoke execution task.",
        "Refuse ALLOW_DOWNLOADS, ALLOW_HEAVY_IMPORT, ALLOW_GPU_TRAINING, ALLOW_ROLLOUTS, ALLOW_RUNTIME_INSTALL, ALLOW_SINGLE_SAMPLE_INFERENCE, and ALLOW_CLOUD_HANDOFF.",
        "Use cached/dummy features or synthetic interface data only.",
        "Train LoRA adapter weights only and keep the backbone frozen.",
        "Report ActionMap+LoRA and TCA-Map+LoRA separately from Distributional TCA-Select gains.",
    ],
    "required_experiment_tracks": [
        "ActionMap + LoRA",
        "TCA-Map + LoRA",
        "TCA-Map + LoRA + Distributional TCA-Select",
        "TCA-Map + QLoRA + Distributional TCA-Select if memory/tooling allows",
    ],
    "hard_stop_reasons": hard_stop_reasons,
    "lora_tiny_smoke_scaffold_ready": not hard_stop_reasons,
    "safe_to_execute_lora_tiny_smoke_now": False,
    "recommended_next_step": (
        "Create the required TCA-Map + LoRA comparison plan next. Actual LoRA tiny-smoke execution remains gated by a separate bounded runner."
        if not hard_stop_reasons
        else "Fix LoRA tiny-smoke scaffold hard-stop reasons before creating an execution runner."
    ),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
