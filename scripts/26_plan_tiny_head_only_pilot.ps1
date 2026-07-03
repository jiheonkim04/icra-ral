param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\tiny_head_only_pilot_plan_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Tiny head-only pilot planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script plans a future tiny head-only pilot only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, or execute OpenVLA-OFT."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to run tiny pilot planner while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}
$env:TCA_MAP_TINY_PILOT_PLAN_REPORT = $reportFullPath

$script = @'
import json
import os
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required for this planner: {exc}")

CONFIGS = [
    Path("configs/actionmap_head_only_lowcompute.yaml"),
    Path("configs/tca_map_head_only_lowcompute.yaml"),
]
BUDGET = Path("configs/compute_budget.yaml")

def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

budget = load_yaml(BUDGET)
limits = budget["limits"]
plans = []
errors = []
for path in CONFIGS:
    cfg = load_yaml(path)
    head = cfg.get("head", {})
    training = cfg.get("training", {})
    heatmap = cfg.get("heatmap", {})
    openvla = cfg.get("openvla_oft", {})
    run = cfg.get("run", {})
    checks = {
        "downloads_disabled": run.get("downloads_allowed") is False,
        "gpu_training_disabled_in_plan": run.get("gpu_training_allowed") is False,
        "rollouts_disabled": run.get("rollouts_allowed") is False,
        "backbone_frozen": cfg.get("backbone", {}).get("freeze") is True,
        "uses_cached_features": cfg.get("backbone", {}).get("use_cached_features") is True,
        "train_backbone_false": training.get("train_backbone") is False,
        "head_only": training.get("train_heads") is True and training.get("train_backbone") is False,
        "max_steps_within_budget": int(training.get("max_steps", 0)) <= int(limits["max_local_pilot_steps_initial"]),
        "trainable_params_within_budget": float(head.get("trainable_params_millions_estimate", 0)) <= float(limits["max_trainable_params_millions_initial"]),
        "grid_within_budget": int(heatmap.get("grid_size", 0)) <= int(limits["max_heatmap_grid_initial"]),
        "low_resolution": heatmap.get("low_resolution") is True,
        "no_high_resolution_voxel_heatmap": heatmap.get("high_resolution_voxel_heatmap") is False,
        "openvla_oft_disabled": openvla.get("enabled") is False,
        "no_full_finetune": openvla.get("full_finetune") is False,
        "no_rollout": openvla.get("full_rollout") is False,
        "no_multiseed": openvla.get("multiseed_sweep") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        errors.append({"config": str(path), "failed_checks": failed})
    plans.append(
        {
            "config": str(path),
            "run_name": run.get("name"),
            "head": head.get("name"),
            "max_steps": training.get("max_steps"),
            "batch_size": training.get("batch_size"),
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
            "trainable_params_millions_estimate": head.get("trainable_params_millions_estimate"),
            "heatmap_grid_size": heatmap.get("grid_size"),
            "checks": checks,
        }
    )

report = {
    "policy": {
        "planning_only": True,
        "risk_assessed_autonomy_for_tiny_training_smoke": True,
        "risk_assessment_required_before_training": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "openvla_oft_executed": False,
        "paper_grade_claims_made": False,
    },
    "budget_limits": limits,
    "plans": plans,
    "errors": errors,
    "configs_pass_policy": not errors,
    "safe_to_run_training_now": False,
    "ready_for_autonomous_tiny_training_smoke": not errors,
    "ready_for_tiny_training_risk_assessment": not errors,
    "offline_proxy_name_policy": "Use offline_standard_proxy or standard_proxy_score; do not call it standard success.",
    "hard_stop": {
        "tiny_training_requires_risk_assessment": True,
        "runtime_install_requires_risk_assessment": False,
        "real_smolvla_feature_extraction_requires_heavy_import_risk_assessment": False,
        "risk_assessed_autonomy": "SmolVLA autonomous pilot risk envelope",
        "remaining_true_hard_stops": [
            "training longer than 30 minutes or more than 300 steps after stable smaller smoke",
            "using more than 14GB VRAM",
            "rollout or simulator execution",
            "OpenVLA-OFT execution",
            "paper-level empirical claims"
        ],
    },
    "recommended_next_step": (
        "Continue autonomously to a tiny head-only training smoke only after a green risk assessment and only if the implementation enforces max_steps<=300 after stable smaller smoke, frozen backbone, no rollout, no OpenVLA-OFT, no paper claim, runtime<=30 minutes, and VRAM<=14GB."
        if not errors
        else "Fix config policy errors before any tiny head-only training smoke."
    ),
}

report_path = Path(os.environ["TCA_MAP_TINY_PILOT_PLAN_REPORT"])
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
