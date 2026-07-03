param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$InputReportPath = "reports\tiny_lora_smoke_report.json",
    [string]$JsonReportPath = "reports\tiny_lora_comparison_report.json",
    [string]$MarkdownReportPath = "reports\tiny_lora_comparison_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Tiny LoRA comparison"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads existing offline proxy LoRA smoke reports only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

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
    Write-Host ("Refusing tiny LoRA comparison while execution gates are set: " + ($setExecutionGates -join ", "))
    exit 20
}

$inputFullPath = if ([System.IO.Path]::IsPathRooted($InputReportPath)) {
    $InputReportPath
} else {
    Join-Path $RepoRoot $InputReportPath
}
$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) {
    $JsonReportPath
} else {
    Join-Path $RepoRoot $JsonReportPath
}
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) {
    $MarkdownReportPath
} else {
    Join-Path $RepoRoot $MarkdownReportPath
}

$env:TCA_MAP_LORA_COMPARISON_INPUT = $inputFullPath
$env:TCA_MAP_LORA_COMPARISON_JSON = $jsonFullPath
$env:TCA_MAP_LORA_COMPARISON_MARKDOWN = $markdownFullPath

$script = @'
import json
import os
from pathlib import Path

INPUT = Path(os.environ["TCA_MAP_LORA_COMPARISON_INPUT"])
JSON_OUT = Path(os.environ["TCA_MAP_LORA_COMPARISON_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_LORA_COMPARISON_MARKDOWN"])

if not INPUT.exists():
    raise SystemExit(f"missing tiny LoRA smoke report: {INPUT}")

source = json.loads(INPUT.read_text(encoding="utf-8"))
if not source.get("tiny_lora_smoke_passed"):
    raise SystemExit("tiny LoRA smoke has not passed; refusing comparison")

arms = {item.get("arm"): item for item in source.get("arms", [])}
required = ("actionmap_lora", "tca_map_lora", "tca_map_lora_distributional_select")
missing = [name for name in required if name not in arms]
if missing:
    raise SystemExit("missing LoRA arms: " + ", ".join(missing))

def metric(arm, name, default=None):
    return arms[arm].get("metrics", {}).get(name, default)

def delta(left_arm, right_arm, name):
    left = metric(left_arm, name)
    right = metric(right_arm, name)
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)

def param_delta(left_arm, right_arm):
    return int(arms[left_arm].get("trainable_lora_parameter_count", 0)) - int(
        arms[right_arm].get("trainable_lora_parameter_count", 0)
    )

comparison = {
    "offline_standard_proxy_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "offline_standard_proxy"),
    "standard_proxy_score_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "standard_proxy_score"),
    "action_l1_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "action_l1"),
    "action_mse_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "action_mse"),
    "target_top1_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "target_top1_accuracy"),
    "target_topk_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "target_topk_accuracy"),
    "wrong_target_proxy_rate_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "wrong_target_proxy_rate"),
    "counterfactual_margin_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "counterfactual_separation_margin"),
    "latency_ms_delta_tca_lora_minus_actionmap_lora": delta("tca_map_lora", "actionmap_lora", "latency_ms"),
    "trainable_lora_params_delta_tca_lora_minus_actionmap_lora": param_delta("tca_map_lora", "actionmap_lora"),
    "offline_standard_proxy_delta_select_minus_tca_lora": delta("tca_map_lora_distributional_select", "tca_map_lora", "offline_standard_proxy"),
    "standard_proxy_score_delta_select_minus_tca_lora": delta("tca_map_lora_distributional_select", "tca_map_lora", "standard_proxy_score"),
    "action_l1_delta_select_minus_tca_lora": delta("tca_map_lora_distributional_select", "tca_map_lora", "action_l1"),
    "wrong_target_proxy_rate_delta_select_minus_tca_lora": delta("tca_map_lora_distributional_select", "tca_map_lora", "wrong_target_proxy_rate"),
    "counterfactual_margin_delta_select_minus_tca_lora": delta("tca_map_lora_distributional_select", "tca_map_lora", "counterfactual_separation_margin"),
    "latency_ms_delta_select_minus_tca_lora": delta("tca_map_lora_distributional_select", "tca_map_lora", "latency_ms"),
}

report = {
    "policy": {
        "bounded_local_pilot": True,
        "offline_proxy_only": True,
        "not_standard_success": True,
        "not_paper_grade": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
    },
    "source_report": str(INPUT),
    "cache_record_count": source.get("cache_record_count"),
    "max_steps": source.get("max_steps"),
    "max_steps_cap": source.get("max_steps_cap"),
    "max_samples": source.get("max_samples"),
    "elapsed_seconds": source.get("elapsed_seconds"),
    "actionmap_lora": {
        "trainable_lora_parameter_count": arms["actionmap_lora"].get("trainable_lora_parameter_count"),
        "metrics": arms["actionmap_lora"].get("metrics", {}),
    },
    "tca_map_lora": {
        "trainable_lora_parameter_count": arms["tca_map_lora"].get("trainable_lora_parameter_count"),
        "metrics": arms["tca_map_lora"].get("metrics", {}),
    },
    "tca_map_lora_distributional_select": {
        "trainable_lora_parameter_count": arms["tca_map_lora_distributional_select"].get("trainable_lora_parameter_count"),
        "metrics": arms["tca_map_lora_distributional_select"].get("metrics", {}),
    },
    "comparison": comparison,
    "tiny_lora_comparison_passed": True,
    "interpretation": (
        "Offline proxy diagnostic only: ActionMap+LoRA, TCA-Map+LoRA, and "
        "TCA-Map+LoRA+Distributional TCA-Select are compared on cached/dummy tiny data. "
        "This is not standard success, not rollout success, and not paper-grade evidence."
    ),
    "recommended_next_step": "Update bounded local pilot status or create a consolidated local pilot report. Stop before real benchmark data, rollouts, simulator execution, OpenVLA-OFT, package/CUDA/PyTorch changes, or paper claims.",
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Tiny LoRA Comparison Report",
    "",
    "This is an offline proxy diagnostic only. It is not standard success, not rollout success, and not paper-grade evidence.",
    "",
    f"- source report: `{INPUT}`",
    f"- cache records: `{report['cache_record_count']}`",
    f"- max steps: `{report['max_steps']}`",
    "",
    "## Deltas",
]
for key, value in comparison.items():
    lines.append(f"- `{key}`: `{value}`")
lines.extend(["", "## Next Step", report["recommended_next_step"], ""])
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
