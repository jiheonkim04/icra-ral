param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\lora_comparison_plan_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LoRA comparison planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not construct adapters, train, download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

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
    Write-Host ("Refusing to plan LoRA comparison while execution gates are set: " + ($setExecutionGates -join ", "))
    exit 20
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}

$env:TCA_MAP_LORA_COMPARISON_REPORT = $reportFullPath

$script = @'
import json
import os
from pathlib import Path

import yaml

from tca_map.adapters import validate_lora_policy_config

REPO_ROOT = Path.cwd()
REPORT_PATH = Path(os.environ["TCA_MAP_LORA_COMPARISON_REPORT"])

CONFIG_PATHS = {
    "actionmap_head_only": REPO_ROOT / "configs" / "actionmap_head_only_lowcompute.yaml",
    "tca_map_head_only": REPO_ROOT / "configs" / "tca_map_head_only_lowcompute.yaml",
    "lora": REPO_ROOT / "configs" / "lora_adapter_lowcompute.yaml",
    "qlora": REPO_ROOT / "configs" / "qlora_adapter_lowcompute.yaml",
    "distributional_tca_select": REPO_ROOT / "configs" / "distributional_tca_select.yaml",
}


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


configs = {name: load_yaml(path) for name, path in CONFIG_PATHS.items()}
lora_validation = validate_lora_policy_config(configs["lora"])
qlora_validation = validate_lora_policy_config(configs["qlora"])

hard_stop_reasons = []
if not lora_validation["passed"]:
    hard_stop_reasons.extend(f"lora config: {error}" for error in lora_validation["errors"])
if not qlora_validation["passed"]:
    hard_stop_reasons.extend(f"qlora config: {error}" for error in qlora_validation["errors"])

comparison_arms = [
    {
        "stage": 0,
        "name": "Native SmolVLA / frozen baseline",
        "adapter": "none",
        "head": "native_or_proxy",
        "distributional_tca_select": False,
        "required": True,
    },
    {
        "stage": 1,
        "name": "ActionMap head-only",
        "adapter": "none",
        "head": "actionmap",
        "distributional_tca_select": False,
        "required": True,
    },
    {
        "stage": 2,
        "name": "TCA-Map head-only",
        "adapter": "none",
        "head": "tca_map",
        "distributional_tca_select": False,
        "required": True,
    },
    {
        "stage": 3,
        "name": "TCA-Map head-only + Distributional TCA-Select",
        "adapter": "none",
        "head": "tca_map",
        "distributional_tca_select": True,
        "required": True,
    },
    {
        "stage": 4,
        "name": "ActionMap + LoRA",
        "adapter": "lora",
        "head": "actionmap",
        "distributional_tca_select": False,
        "required": True,
    },
    {
        "stage": 5,
        "name": "TCA-Map + LoRA",
        "adapter": "lora",
        "head": "tca_map",
        "distributional_tca_select": False,
        "required": True,
    },
    {
        "stage": 6,
        "name": "TCA-Map + LoRA + Distributional TCA-Select",
        "adapter": "lora",
        "head": "tca_map",
        "distributional_tca_select": True,
        "required": True,
    },
    {
        "stage": 7,
        "name": "TCA-Map + QLoRA + Distributional TCA-Select",
        "adapter": "qlora",
        "head": "tca_map",
        "distributional_tca_select": True,
        "required": "if memory/tooling allows",
    },
]

required_comparisons = [
    {
        "name": "TCA-Map head-only vs ActionMap head-only",
        "isolates": "target-conditioned head gain",
        "left": "TCA-Map head-only",
        "right": "ActionMap head-only",
    },
    {
        "name": "TCA-Map + TCA-Select vs TCA-Map without TCA-Select",
        "isolates": "inference-time selection gain",
        "left": "TCA-Map head-only + Distributional TCA-Select",
        "right": "TCA-Map head-only",
    },
    {
        "name": "TCA-Map + LoRA vs ActionMap + LoRA",
        "isolates": "target-conditioned head gain under LoRA",
        "left": "TCA-Map + LoRA",
        "right": "ActionMap + LoRA",
    },
    {
        "name": "TCA-Map + LoRA + TCA-Select vs TCA-Map + LoRA only",
        "isolates": "inference-time selection gain under LoRA",
        "left": "TCA-Map + LoRA + Distributional TCA-Select",
        "right": "TCA-Map + LoRA",
    },
    {
        "name": "QLoRA variant if feasible under local compute budget",
        "isolates": "memory-saving adaptation feasibility",
        "left": "TCA-Map + QLoRA + Distributional TCA-Select",
        "right": "TCA-Map + LoRA + Distributional TCA-Select",
    },
]

report = {
    "policy": {
        "planning_only": True,
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
        "privileged_inference_allowed": False,
    },
    "configs": {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "run_mode": configs[name].get("run", {}).get("mode"),
            "head": configs[name].get("head", {}).get("name"),
            "training_max_steps": configs[name].get("training", {}).get("max_steps"),
        }
        for name, path in CONFIG_PATHS.items()
    },
    "lora_validation": lora_validation,
    "qlora_validation": qlora_validation,
    "comparison_arms": comparison_arms,
    "required_comparisons": required_comparisons,
    "fairness_controls": {
        "same_checkpoint_family": True,
        "same_cached_feature_contract": True,
        "same_offline_proxy_schema": True,
        "same_tiny_smoke_step_cap": 100,
        "same_runtime_cap_seconds": 900,
        "same_vram_target_gb": 14,
        "single_seed_only_until_later_risk_assessment": True,
        "no_privileged_inference": True,
    },
    "minimum_metrics": [
        "action_l1",
        "action_mse",
        "action_voxel_hit_rate_or_distance_to_expert_voxel",
        "target_heatmap_top1_topk_accuracy",
        "wrong_target_proxy_rate",
        "counterfactual_target_action_separation_margin",
        "nuisance_stability_score",
        "latency",
        "max_gpu_memory",
        "trainable_parameter_estimate",
    ],
    "hard_stop_reasons": hard_stop_reasons,
    "lora_comparison_plan_ready": not hard_stop_reasons,
    "safe_to_run_lora_comparison_now": False,
    "recommended_next_step": (
        "Create the QLoRA feasibility check next. Do not run LoRA comparison training until a bounded runner is explicitly present and the hard-stop gates remain clear."
        if not hard_stop_reasons
        else "Fix LoRA comparison plan hard-stop reasons before moving to QLoRA feasibility."
    ),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
