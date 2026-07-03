param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$InputReportPath = "reports\tiny_head_only_smoke_report.json",
    [string]$JsonReportPath = "reports\head_only_tiny_comparison_report.json",
    [string]$MarkdownReportPath = "reports\head_only_tiny_comparison_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Head-only tiny comparison"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads existing offline proxy smoke reports only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

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
    Write-Host ("Refusing head-only comparison while execution gates are set: " + ($setExecutionGates -join ", "))
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

$env:TCA_MAP_HEAD_ONLY_COMPARISON_INPUT = $inputFullPath
$env:TCA_MAP_HEAD_ONLY_COMPARISON_JSON = $jsonFullPath
$env:TCA_MAP_HEAD_ONLY_COMPARISON_MARKDOWN = $markdownFullPath

$script = @'
import json
import os
from pathlib import Path

INPUT = Path(os.environ["TCA_MAP_HEAD_ONLY_COMPARISON_INPUT"])
JSON_OUT = Path(os.environ["TCA_MAP_HEAD_ONLY_COMPARISON_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_HEAD_ONLY_COMPARISON_MARKDOWN"])

if not INPUT.exists():
    raise SystemExit(f"missing tiny head-only smoke report: {INPUT}")

source = json.loads(INPUT.read_text(encoding="utf-8"))
if not source.get("tiny_head_only_smoke_passed"):
    raise SystemExit("tiny head-only smoke has not passed; refusing comparison")

heads = {item.get("head"): item for item in source.get("heads", [])}
missing = [name for name in ("actionmap", "tca_map") if name not in heads]
if missing:
    raise SystemExit("missing head outputs: " + ", ".join(missing))

def metric(head, name, default=None):
    return heads[head].get("metrics", {}).get(name, default)

def delta(name):
    left = metric("tca_map", name)
    right = metric("actionmap", name)
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)

comparison = {
    "offline_standard_proxy_delta_tca_minus_actionmap": delta("offline_standard_proxy"),
    "standard_proxy_score_delta_tca_minus_actionmap": delta("standard_proxy_score"),
    "action_l1_delta_tca_minus_actionmap": delta("action_l1"),
    "action_mse_delta_tca_minus_actionmap": delta("action_mse"),
    "target_top1_delta_tca_minus_actionmap": delta("target_top1_accuracy"),
    "target_topk_delta_tca_minus_actionmap": delta("target_topk_accuracy"),
    "wrong_target_proxy_rate_delta_tca_minus_actionmap": delta("wrong_target_proxy_rate"),
    "counterfactual_separation_margin_delta_tca_minus_actionmap": delta("counterfactual_separation_margin"),
    "nuisance_stability_delta_tca_minus_actionmap": delta("nuisance_stability_score"),
    "latency_ms_delta_tca_minus_actionmap": delta("latency_ms"),
    "trainable_parameter_count_delta_tca_minus_actionmap": int(heads["tca_map"].get("trainable_parameter_count", 0))
    - int(heads["actionmap"].get("trainable_parameter_count", 0)),
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
    "elapsed_seconds": source.get("elapsed_seconds"),
    "actionmap": {
        "trainable_parameter_count": heads["actionmap"].get("trainable_parameter_count"),
        "metrics": heads["actionmap"].get("metrics", {}),
    },
    "tca_map": {
        "trainable_parameter_count": heads["tca_map"].get("trainable_parameter_count"),
        "metrics": heads["tca_map"].get("metrics", {}),
    },
    "comparison": comparison,
    "head_only_tiny_comparison_passed": True,
    "interpretation": (
        "Offline proxy diagnostic only: TCA-Map and ActionMap are compared on cached/dummy tiny data. "
        "This is not standard success, not rollout success, and not paper-grade evidence."
    ),
    "recommended_next_step": "Proceed to bounded tiny LoRA smoke scaffolding/execution inside standing approval; stop before real benchmarks, rollouts, simulator execution, OpenVLA-OFT, package/CUDA/PyTorch changes, or paper claims.",
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Head-Only Tiny Comparison Report",
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
