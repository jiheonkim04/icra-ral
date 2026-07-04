param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$JsonReportPath = "reports\local_pilot_status_report.json",
    [string]$MarkdownReportPath = "reports\local_pilot_status_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Local pilot status report"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script summarizes existing reports only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

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
    Write-Host ("Refusing local pilot status generation while execution gates are set: " + ($setExecutionGates -join ", "))
    exit 20
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

$env:TCA_MAP_LOCAL_PILOT_STATUS_JSON = $jsonFullPath
$env:TCA_MAP_LOCAL_PILOT_STATUS_MARKDOWN = $markdownFullPath

$script = @'
import json
import os
from pathlib import Path

JSON_OUT = Path(os.environ["TCA_MAP_LOCAL_PILOT_STATUS_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_LOCAL_PILOT_STATUS_MARKDOWN"])
REPO = Path.cwd()

REPORTS = {
    "preflight": REPO / "reports" / "preflight_report.json",
    "smoke": REPO / "reports" / "smoke_report.json",
    "runtime_deps": REPO / "reports" / "smolvla_runtime_deps_report.json",
    "load_only": REPO / "reports" / "smolvla_load_only_smoke_report.json",
    "single_sample_interface": REPO / "reports" / "smolvla_single_sample_interface_report.json",
    "feature_cache_eval": REPO / "reports" / "feature_cache_eval_report.json",
    "tiny_head_only": REPO / "reports" / "tiny_head_only_smoke_report.json",
    "head_only_comparison": REPO / "reports" / "head_only_tiny_comparison_report.json",
    "tiny_lora": REPO / "reports" / "tiny_lora_smoke_report.json",
    "tiny_lora_comparison": REPO / "reports" / "tiny_lora_comparison_report.json",
    "bounded_local_pilot_extension": REPO / "reports" / "bounded_local_pilot_extension_report.json",
    "libero_metadata_subset": REPO / "reports" / "libero_metadata_subset_report.json",
    "libero_offline_interface": REPO / "reports" / "libero_offline_interface_smoke_report.json",
    "libero_offline_counterfactual_split": REPO / "reports" / "libero_offline_counterfactual_split_report.json",
    "libero_offline_head_comparison": REPO / "reports" / "libero_offline_actionmap_tca_comparison_report.json",
    "libero_offline_lora_comparison": REPO / "reports" / "libero_offline_lora_comparison_report.json",
    "libero_offline_bounded_pilot": REPO / "reports" / "libero_offline_bounded_pilot_report.json",
    "simulator_readiness": REPO / "reports" / "simulator_readiness_plan_report.json",
    "bounded_simulator_import_smoke": REPO / "reports" / "bounded_simulator_import_smoke_report.json",
    "wsl_simulator_dependency": REPO / "reports" / "wsl_simulator_dependency_report.json",
    "go_no_go": REPO / "reports" / "go_no_go_status_report.json",
}

def read_json(path):
    if not path.exists():
        return {"exists": False, "data": None}
    try:
        return {"exists": True, "data": json.loads(path.read_text(encoding="utf-8-sig"))}
    except json.JSONDecodeError as exc:
        return {"exists": True, "data": None, "error": str(exc)}

loaded = {name: read_json(path) for name, path in REPORTS.items()}

def data(name):
    return loaded[name].get("data") or {}

def result_passed(name):
    return bool(data(name).get("result", {}).get("passed") or data(name).get(f"{name}_passed"))

def source_summary(name):
    payload = data(name)
    return {
        "path": str(REPORTS[name]),
        "exists": bool(loaded[name].get("exists")),
        "error": loaded[name].get("error"),
        "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
    }

status = {
    "preflight_passed": bool(data("preflight").get("safe_to_run_dummy_smoke")),
    "dummy_smoke_passed": bool(data("smoke").get("train_smoke_passed") and data("smoke").get("eval_smoke_passed")),
    "runtime_deps_ready": bool(data("runtime_deps").get("runtime_dependencies", {}).get("ready_for_load_only_runtime")),
    "smolvla_load_only_smoke_passed": result_passed("load_only"),
    "single_sample_interface_passed": result_passed("single_sample_interface"),
    "feature_cache_eval_passed": bool(data("feature_cache_eval").get("cache_valid") and not data("feature_cache_eval").get("validation_errors")),
    "tiny_head_only_smoke_passed": bool(data("tiny_head_only").get("tiny_head_only_smoke_passed")),
    "head_only_comparison_passed": bool(data("head_only_comparison").get("head_only_tiny_comparison_passed")),
    "tiny_lora_smoke_passed": bool(data("tiny_lora").get("tiny_lora_smoke_passed")),
    "tiny_lora_comparison_passed": bool(data("tiny_lora_comparison").get("tiny_lora_comparison_passed")),
    "bounded_local_pilot_extension_passed": bool(data("bounded_local_pilot_extension").get("bounded_local_pilot_extension_passed")),
    "libero_metadata_subset_ready": bool(data("libero_metadata_subset").get("ready_for_metadata_only_subset")),
    "libero_offline_interface_ready": bool(data("libero_offline_interface").get("ready_for_offline_interface_smoke")),
    "libero_offline_interface_decision": data("libero_offline_interface").get("decision"),
    "libero_offline_counterfactual_split_ready": bool(data("libero_offline_counterfactual_split").get("ready_for_tiny_offline_counterfactual_split")),
    "libero_offline_actionmap_tca_ready": bool(data("libero_offline_counterfactual_split").get("ready_for_tiny_offline_actionmap_tca_comparison")),
    "libero_offline_counterfactual_pair_count": data("libero_offline_counterfactual_split").get("counterfactual_pair_count"),
    "libero_offline_head_comparison_passed": bool(data("libero_offline_head_comparison").get("libero_offline_head_comparison_passed")),
    "libero_ready_for_required_tiny_lora_comparison": bool(data("libero_offline_head_comparison").get("ready_for_required_tiny_lora_comparison")),
    "libero_offline_lora_comparison_passed": bool(data("libero_offline_lora_comparison").get("libero_offline_lora_comparison_passed")),
    "libero_ready_for_bounded_local_pilot_report": bool(data("libero_offline_lora_comparison").get("ready_for_bounded_local_pilot_report")),
    "libero_offline_bounded_pilot_report_passed": bool(data("libero_offline_bounded_pilot").get("libero_offline_bounded_pilot_report_passed")),
    "libero_ready_for_simulator_readiness_risk_assessment": bool(data("libero_offline_bounded_pilot").get("ready_for_simulator_readiness_risk_assessment")),
    "simulator_readiness_report_present": bool(loaded["simulator_readiness"].get("exists")),
    "simulator_readiness_decision": data("simulator_readiness").get("decision"),
    "simulator_effective_runtime_platform": (data("simulator_readiness").get("host") or {}).get("effective_runtime_platform"),
    "simulator_path_ready": bool(data("simulator_readiness").get("ready_for_simulator_path_check")),
    "simulator_dataset_path_ready": bool(data("simulator_readiness").get("ready_for_dataset_path_check")),
    "simulator_import_smoke_ready": bool(data("simulator_readiness").get("ready_for_simulator_import_smoke")),
    "simulator_render_smoke_ready": bool(data("simulator_readiness").get("ready_for_simulator_render_smoke")),
    "simulator_rollout_ready": bool(data("simulator_readiness").get("ready_for_libero_rollout")),
    "simulator_stop_reasons": data("simulator_readiness").get("stop_reasons", []),
    "bounded_simulator_import_smoke_report_present": bool(loaded["bounded_simulator_import_smoke"].get("exists")),
    "bounded_simulator_import_smoke_passed": bool(data("bounded_simulator_import_smoke").get("bounded_simulator_import_smoke_passed")),
    "bounded_simulator_import_smoke_decision": data("bounded_simulator_import_smoke").get("decision"),
    "bounded_simulator_import_smoke_imports_attempted": bool((data("bounded_simulator_import_smoke").get("policy") or {}).get("simulator_imports_attempted")),
    "bounded_simulator_import_smoke_rollouts_performed": bool((data("bounded_simulator_import_smoke").get("policy") or {}).get("rollouts_performed")),
    "wsl_simulator_dependency_report_present": bool(loaded["wsl_simulator_dependency"].get("exists")),
    "wsl_simulator_dependency_decision": data("wsl_simulator_dependency").get("decision"),
    "wsl_ready_for_user_level_pip_install": bool(data("wsl_simulator_dependency").get("ready_for_user_level_pip_install")),
    "wsl_ready_for_simulator_import_retry": bool(data("wsl_simulator_dependency").get("ready_for_simulator_import_retry")),
    "wsl_simulator_dependency_stop_reasons": data("wsl_simulator_dependency").get("stop_reasons", []),
    "libero_rollout_ready": bool(data("libero_offline_interface").get("ready_for_rollout")),
    "ready_for_bounded_local_pilot": bool(data("go_no_go").get("ready_for_bounded_local_pilot")),
    "blocked_for_larger_paper_grade_stage": bool(data("go_no_go").get("blocked_for_larger_paper_grade_stage", True)),
}

all_bounded_smokes_passed = all(
    status[key]
    for key in [
        "preflight_passed",
        "dummy_smoke_passed",
        "runtime_deps_ready",
        "smolvla_load_only_smoke_passed",
        "single_sample_interface_passed",
        "feature_cache_eval_passed",
        "tiny_head_only_smoke_passed",
        "head_only_comparison_passed",
        "tiny_lora_smoke_passed",
        "tiny_lora_comparison_passed",
        "bounded_local_pilot_extension_passed",
    ]
)

missing_reports = [name for name, item in loaded.items() if not item.get("exists")]
parse_errors = {name: item.get("error") for name, item in loaded.items() if item.get("error")}
recommended_next_step = (
    "Bounded simulator import smoke passed. Create a separate bounded render-smoke risk gate if needed; keep rollout blocked."
    if status["bounded_simulator_import_smoke_passed"]
    else "WSL simulator dependency check is complete and blocks import retry. Run the standing-approved WSL simulator dependency ladder risk assessment; if green, set up minimal WSL Python packaging/dependencies, then rerun import smoke. Keep render/rollout blocked."
    if status["wsl_simulator_dependency_report_present"] and not status["wsl_ready_for_simulator_import_retry"]
    else "Bounded simulator import smoke ran but did not pass. Resolve WSL/Linux dependency or import errors before render smoke or rollout."
    if status["bounded_simulator_import_smoke_report_present"]
    else "Simulator readiness planner is green for import-smoke planning only. Create a separate bounded import-smoke branch; keep render/rollout blocked."
    if status["simulator_import_smoke_ready"]
    else "Simulator readiness planner ran and keeps import/render/rollout blocked. Resolve the listed stop reasons before any simulator import smoke."
    if status["simulator_readiness_report_present"]
    else "Run a simulator readiness/import-render risk assessment if installed locally; stop before rollout unless the assessment is green and inside budget."
    if status["libero_ready_for_simulator_readiness_risk_assessment"]
    else (
        "Choose the next concrete stage and run a risk assessment. Proceed automatically if the assessment is inside budget; stop only if risk is ambiguous, outside budget, external/irreversible, OpenVLA-OFT-related, token/license/payment-related, credentialed/system-driver/license-gated, or paper-claim-related."
        if all_bounded_smokes_passed
        else "Regenerate the missing or failed bounded local pilot reports before any larger step."
    )
)

report = {
    "policy": {
        "summary_only": True,
        "bounded_local_pilot": True,
        "risk_assessed_autonomy_policy": True,
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
    "source_reports": {name: source_summary(name) for name in REPORTS},
    "status": status,
    "all_bounded_smoke_reports_present_and_passed": all_bounded_smokes_passed,
    "missing_reports": missing_reports,
    "parse_errors": parse_errors,
    "risk_assessed_next_gates": [
        "simulator readiness/import-render smoke after WSL dependency/import readiness is green",
        "bounded rollout only after simulator smoke, task_count<=5, runtime<=30 minutes, no paper claim",
        "bounded local training extension beyond the current cached-feature smoke only after a fresh green risk assessment",
        "QLoRA feasibility or tooling only if package/CUDA/PyTorch risk is inside budget",
    ],
    "external_irreversible_stop_gates": [
        "OpenVLA-OFT download/import/load/execution until separate risk budget exists",
        "token or secret access",
        "paid service",
        "license click-through",
        "external upload/submission/publishing",
        "system-wide CUDA/PyTorch/driver changes",
        "credentialed/system-driver/license-gated system setup",
        "paper-grade empirical claims",
    ],
    "hard_stop_boundaries": [
        "OpenVLA-OFT execution",
        "token/secret/payment/license gate",
        "external upload/submission/publishing",
        "system-level CUDA/PyTorch/driver change",
        "paper-grade empirical claims",
    ],
    "local_pilot_status_passed": not parse_errors and all_bounded_smokes_passed,
    "recommended_next_step": recommended_next_step,
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Local Pilot Status Report",
    "",
    "This is a summary-only report over existing bounded local smoke outputs. It is not standard success, not rollout success, and not paper-grade evidence.",
    "",
    "## Status",
]
for key, value in status.items():
    lines.append(f"- `{key}`: `{value}`")
lines.extend(["", "## Missing Reports"])
if missing_reports:
    for name in missing_reports:
        lines.append(f"- `{name}`")
else:
    lines.append("- none")
lines.extend(["", "## Risk-Assessed Next Gates"])
for item in report["risk_assessed_next_gates"]:
    lines.append(f"- {item}")
lines.extend(["", "## External Stop Gates"])
for item in report["external_irreversible_stop_gates"]:
    lines.append(f"- {item}")
lines.extend(["", "## Next Step", report["recommended_next_step"], ""])
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
