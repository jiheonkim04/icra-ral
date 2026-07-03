param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CacheDir = "runs\feature_cache\dummy_contract",
    [string]$HeadOnlyReportPath = "reports\bounded_head_only_extension_report.json",
    [string]$JsonReportPath = "reports\bounded_local_pilot_extension_report.json",
    [string]$MarkdownReportPath = "reports\bounded_local_pilot_extension_report.md",
    [int]$MaxSteps = 64,
    [int]$MaxRuntimeSeconds = 900,
    [switch]$PrepareDummyCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded local pilot extension"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs a bounded cached-feature head-only smoke extension. It does not download assets, use GPU, import heavy VLA models, load models, run model inference, rollout, execute simulators, execute OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$localPolicyMaxSteps = 300
$runnerMaxStepsCap = 100
if ($MaxSteps -lt 1 -or $MaxSteps -gt $runnerMaxStepsCap) {
    Write-Host "Refusing: MaxSteps must be between 1 and $runnerMaxStepsCap for this extension runner. The broader local policy cap remains $localPolicyMaxSteps."
    exit 11
}

if ($MaxRuntimeSeconds -lt 1 -or $MaxRuntimeSeconds -gt 900) {
    Write-Host "Refusing: MaxRuntimeSeconds must be between 1 and 900."
    exit 12
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing bounded local pilot extension while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

if ($env:ALLOW_TINY_TRAINING -ne "1") {
    Write-Host "Refusing: ALLOW_TINY_TRAINING=1 is required for bounded local pilot extension."
    exit 21
}

$headReportFullPath = if ([System.IO.Path]::IsPathRooted($HeadOnlyReportPath)) { $HeadOnlyReportPath } else { Join-Path $RepoRoot $HeadOnlyReportPath }
$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }

$argsList = @(
    "-m",
    "tca_map.features.tiny_head_only_smoke",
    "--cache-dir",
    $CacheDir,
    "--report-path",
    $headReportFullPath,
    "--max-steps",
    [string]$MaxSteps,
    "--max-runtime-seconds",
    [string]$MaxRuntimeSeconds
)
if ($PrepareDummyCache) {
    $argsList += "--prepare-dummy-cache"
}

$headOutput = & $Python @argsList 2>&1
if ($LASTEXITCODE -ne 0) {
    $headOutput | ForEach-Object { Write-Host $_ }
    exit $LASTEXITCODE
}

$env:TCA_MAP_EXTENSION_HEAD_REPORT = $headReportFullPath
$env:TCA_MAP_EXTENSION_JSON = $jsonFullPath
$env:TCA_MAP_EXTENSION_MARKDOWN = $markdownFullPath
$env:TCA_MAP_EXTENSION_MAX_STEPS = [string]$MaxSteps
$env:TCA_MAP_EXTENSION_MAX_RUNTIME_SECONDS = [string]$MaxRuntimeSeconds

$summaryScript = @'
import json
import os
from pathlib import Path

head_report_path = Path(os.environ["TCA_MAP_EXTENSION_HEAD_REPORT"])
json_path = Path(os.environ["TCA_MAP_EXTENSION_JSON"])
markdown_path = Path(os.environ["TCA_MAP_EXTENSION_MARKDOWN"])
max_steps = int(os.environ["TCA_MAP_EXTENSION_MAX_STEPS"])
max_runtime_seconds = int(os.environ["TCA_MAP_EXTENSION_MAX_RUNTIME_SECONDS"])

head_report = json.loads(head_report_path.read_text(encoding="utf-8"))

def summarize_head(item):
    metrics = item.get("metrics", {})
    return {
        "head": item.get("head"),
        "max_steps": item.get("max_steps"),
        "trainable_parameter_count": item.get("trainable_parameter_count"),
        "finite_losses": item.get("finite_losses"),
        "action_l1": metrics.get("action_l1"),
        "action_mse": metrics.get("action_mse"),
        "offline_standard_proxy": metrics.get("offline_standard_proxy"),
        "target_top1_accuracy": metrics.get("target_top1_accuracy"),
        "wrong_target_proxy_rate": metrics.get("wrong_target_proxy_rate"),
        "counterfactual_separation_margin": metrics.get("counterfactual_separation_margin"),
    }

heads = [summarize_head(item) for item in head_report.get("heads", [])]
passed = bool(
    head_report.get("tiny_head_only_smoke_passed")
    and max_steps <= 100
    and head_report.get("policy", {}).get("training_performed") is True
    and all(item.get("finite_losses") for item in head_report.get("heads", []))
)

report = {
    "policy": {
        "bounded_local_pilot_extension": True,
        "risk_assessed_autonomy_for_tiny_training_smoke": True,
        "cached_features_used": True,
        "backbone_frozen": True,
        "head_only": True,
        "offline_proxy_only": True,
        "not_standard_success": True,
        "not_paper_grade": True,
        "real_dataset_used": False,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": True,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
    },
    "bounds": {
        "local_policy_max_steps": 300,
        "runner_max_steps_cap": 100,
        "requested_max_steps": max_steps,
        "max_runtime_seconds": max_runtime_seconds,
        "batch_size": 1,
        "expected_vram_gb": 0,
    },
    "head_only_report_path": str(head_report_path),
    "head_only_report_passed": bool(head_report.get("tiny_head_only_smoke_passed")),
    "cache_dir": head_report.get("cache_dir"),
    "cache_record_count": head_report.get("cache_record_count"),
    "elapsed_seconds": head_report.get("elapsed_seconds"),
    "heads": heads,
    "bounded_local_pilot_extension_passed": passed,
    "safe_to_run_real_dataset_pilot": False,
    "safe_to_run_rollouts": False,
    "recommended_next_step": (
        "Use this only as bounded offline proxy/interface evidence. Stop before real dataset training, simulator execution, rollout, OpenVLA-OFT, or paper claims."
        if passed
        else "Fix the bounded local pilot extension before any larger local pilot work."
    ),
}

json_path.parent.mkdir(parents=True, exist_ok=True)
markdown_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Bounded Local Pilot Extension Report",
    "",
    "This is a bounded cached-feature offline proxy smoke. It is not standard success, rollout success, or paper-grade evidence.",
    "",
    f"- passed: `{passed}`",
    f"- requested max steps: `{max_steps}`",
    f"- runner max steps cap: `100`",
    f"- local policy max steps: `300`",
    f"- head-only report: `{head_report_path}`",
    "",
    "## Heads",
]
for item in heads:
    lines.append(
        f"- `{item['head']}`: action_l1=`{item['action_l1']}`, action_mse=`{item['action_mse']}`, offline_standard_proxy=`{item['offline_standard_proxy']}`"
    )
lines.extend(["", "## Next Step", report["recommended_next_step"], ""])
markdown_path.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$summaryScript | & $Python -
exit $LASTEXITCODE
