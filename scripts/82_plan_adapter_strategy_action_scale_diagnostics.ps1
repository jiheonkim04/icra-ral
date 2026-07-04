param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ActionAuditReportPath = "reports\action_interface_metadata_audit_report.json",
    [string]$MetricSummaryReportPath = "reports\reduced_scope_rollout_metric_summary_report.json",
    [string]$ZeroComparisonReportPath = "reports\zero_action_policy_diagnostic_comparison_report.json",
    [string]$RolloutBridgeSourcePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$JsonReportPath = "reports\adapter_strategy_action_scale_diagnostics_plan_report.json",
    [string]$MarkdownReportPath = "reports\adapter_strategy_action_scale_diagnostics_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Adapter-strategy/action-scale diagnostics planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This planner reads existing reports and source files only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    "ALLOW_WSL_SMOLVLA_SINGLE_ACTION"
)

$setExecutionGates = @()
foreach ($gate in $executionGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setExecutionGates += $gate
    }
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

$env:TCA_MAP_ADAPTER_STRATEGY_AUDIT = Resolve-RepoPath -Path $ActionAuditReportPath
$env:TCA_MAP_ADAPTER_STRATEGY_METRIC = Resolve-RepoPath -Path $MetricSummaryReportPath
$env:TCA_MAP_ADAPTER_STRATEGY_COMPARISON = Resolve-RepoPath -Path $ZeroComparisonReportPath
$env:TCA_MAP_ADAPTER_STRATEGY_SOURCE = Resolve-RepoPath -Path $RolloutBridgeSourcePath
$env:TCA_MAP_ADAPTER_STRATEGY_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_ADAPTER_STRATEGY_MARKDOWN = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_ADAPTER_STRATEGY_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

AUDIT = Path(os.environ["TCA_MAP_ADAPTER_STRATEGY_AUDIT"])
METRIC = Path(os.environ["TCA_MAP_ADAPTER_STRATEGY_METRIC"])
COMPARISON = Path(os.environ["TCA_MAP_ADAPTER_STRATEGY_COMPARISON"])
SOURCE = Path(os.environ["TCA_MAP_ADAPTER_STRATEGY_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_ADAPTER_STRATEGY_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_ADAPTER_STRATEGY_MARKDOWN"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_ADAPTER_STRATEGY_GATES", "").split(";") if item]

def load_json(path):
    if not path.exists():
        return None, f"Missing input report: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:
        return None, f"Could not read {path}: {exc}"

def write_outputs(report):
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Adapter-Strategy/Action-Scale Diagnostics Plan Report",
        "",
        f"- decision: {report['decision']}",
        f"- planner passed: {report['adapter_strategy_action_scale_diagnostics_plan_passed']}",
        f"- ready for diagnostic runner implementation: {report['ready_for_adapter_strategy_diagnostic_runner_implementation']}",
        f"- ready for rollout scaling: {report['ready_for_rollout_scaling']}",
        f"- adapter metadata present: {report['inputs']['adapter_metadata_present']}",
        f"- current action adapter strategies: {report['inputs']['action_adapter_strategies']}",
        f"- diagnostic success rate: {report['inputs']['diagnostic_success_rate']}",
        f"- reward sum: {report['inputs']['reward_sum_total']}",
        f"- standard success claimed: {report['claims']['standard_success_claimed']}",
        f"- paper-grade claim made: {report['claims']['paper_grade_claim_made']}",
        "",
        report["recommended_next_step"],
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

audit, audit_error = load_json(AUDIT)
metric_report, metric_error = load_json(METRIC)
comparison, comparison_error = load_json(COMPARISON)
source_text = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""

stop_reasons = []
warnings = []
if SET_GATES:
    stop_reasons.append("Refusing adapter-strategy planning while execution gates are set: " + ", ".join(SET_GATES))
for error in [audit_error, metric_error, comparison_error]:
    if error:
        stop_reasons.append(error)
if not SOURCE.exists():
    stop_reasons.append(f"Missing rollout bridge source: {SOURCE}")

metric = (metric_report or {}).get("metric_summary") or {}
comparison_data = (comparison or {}).get("comparison") or {}
audit_ready = bool((audit or {}).get("ready_for_adapter_strategy_diagnosis"))
comparison_ready = bool((comparison or {}).get("ready_for_adapter_strategy_diagnosis"))
adapter_metadata_present = bool(metric.get("adapter_metadata_present"))
adapter_wiring_clean = bool(comparison_data.get("adapter_wiring_clean"))
diagnostic_success_rate = metric.get("diagnostic_success_rate")
reward_sum_total = metric.get("reward_sum_total")
current_strategies = metric.get("action_adapter_strategies") or []
source_has_strategy_hook = "adapt_policy_action_to_env_action" in source_text

if not audit_ready:
    stop_reasons.append("Action-interface audit is not ready for adapter-strategy diagnosis.")
if not comparison_ready:
    stop_reasons.append("Zero-action comparison is not ready for adapter-strategy diagnosis.")
if not adapter_metadata_present:
    stop_reasons.append("Adapter metadata is missing from the reduced-scope metric summary.")
if not source_has_strategy_hook:
    stop_reasons.append("Rollout bridge does not use the explicit action adapter hook.")

policy = {
    "planning_only": True,
    "downloads_performed": False,
    "installs_performed": False,
    "heavy_model_imports_performed": False,
    "model_load_performed": False,
    "model_inference_performed": False,
    "simulator_environment_created": False,
    "rollouts_performed": False,
    "benchmark_rollouts_performed": False,
    "gpu_jobs_performed": False,
    "training_performed": False,
    "openvla_oft_executed": False,
    "tokens_read_or_written": False,
    "paper_grade_claims_made": False,
}
claims = {
    "standard_success_claimed": False,
    "benchmark_success_claimed": False,
    "counterfactual_robustness_claimed": False,
    "sota_claimed": False,
    "paper_grade_claim_made": False,
}

diagnostic_plan = {
    "evidence_label": "adapter_strategy_action_scale_diagnostic_plan",
    "max_tasks": 1,
    "max_steps_per_variant": 10,
    "max_variants_first_runner": 3,
    "expected_runtime_minutes": 15,
    "expected_vram_gb": 0,
    "first_gripper_strategy_variants": [
        "policy_6d_delta_pose_plus_gripper_zero_hold",
        "policy_6d_delta_pose_plus_gripper_open",
        "policy_6d_delta_pose_plus_gripper_close",
    ],
    "later_action_scale_variants": [0.5, 1.0, 1.5],
    "later_prompt_checks": [
        "raw_bddl_stem_prompt",
        "scene_prefix_stripped_prompt",
        "lowercase_natural_language_prompt",
    ],
    "later_camera_checks": [
        "current_alias_mapping",
        "camera3_eye_in_hand_variant_if_supported",
    ],
    "acceptance_checks": [
        "no downloads",
        "no training",
        "no GPU jobs",
        "no OpenVLA-OFT",
        "one task only for first diagnostic runner",
        "at most 10 steps per variant",
        "adapter metadata recorded for every variant",
        "results labeled diagnostic only",
    ],
}

ready = bool(not stop_reasons)
decision = "proceed" if ready else "stop"
reason = (
    "Adapter metadata is present, zero reward remains, and a bounded strategy/action-scale diagnostic plan is ready."
    if ready
    else "Adapter-strategy/action-scale diagnostic prerequisites are not satisfied."
)
report = {
    "adapter_strategy_action_scale_diagnostics_plan_passed": ready,
    "decision": decision,
    "reason": reason,
    "source_reports": {
        "action_interface_audit": str(AUDIT),
        "reduced_scope_metric_summary": str(METRIC),
        "zero_action_comparison": str(COMPARISON),
        "rollout_bridge_source": str(SOURCE),
    },
    "policy": policy,
    "claims": claims,
    "inputs": {
        "audit_ready_for_adapter_strategy_diagnosis": audit_ready,
        "comparison_ready_for_adapter_strategy_diagnosis": comparison_ready,
        "adapter_metadata_present": adapter_metadata_present,
        "adapter_wiring_clean": adapter_wiring_clean,
        "action_adapter_strategies": current_strategies,
        "state_adapters": metric.get("state_adapters") or [],
        "image_source_keys": metric.get("image_source_keys") or {},
        "diagnostic_success_rate": diagnostic_success_rate,
        "reward_sum_total": reward_sum_total,
        "last_env_action_max_abs": metric.get("last_env_action_max_abs"),
        "last_env_action_gripper_component": metric.get("last_env_action_gripper_component"),
        "source_has_strategy_hook": source_has_strategy_hook,
    },
    "risk_assessment": {
        "task": "adapter-strategy/action-scale diagnostics planning",
        "command": "future separately gated diagnostic runner",
        "source": "local SmolVLA checkpoint and local LIBERO/RoboSuite WSL simulator topology",
        "expected_size_gb": 0,
        "expected_runtime_minutes": diagnostic_plan["expected_runtime_minutes"],
        "expected_ram_gb": 8,
        "expected_vram_gb": diagnostic_plan["expected_vram_gb"],
        "task_count": diagnostic_plan["max_tasks"],
        "max_steps_per_variant": diagnostic_plan["max_steps_per_variant"],
        "token_login_license_payment_needed": False,
        "simulator_will_run_in_future_runner": True,
        "learned_policy_inference_will_run_in_future_runner": True,
        "training_will_run": False,
        "openvla_oft_will_run": False,
        "paper_claim_will_be_made": False,
    },
    "diagnostic_plan": diagnostic_plan,
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_adapter_strategy_diagnostic_runner_implementation": ready,
    "ready_for_rollout_scaling": False,
    "recommended_next_step": (
        "Implement a separately gated one-task adapter-strategy diagnostic runner; do not scale rollout or make claims."
        if ready
        else "Fix missing adapter-aware diagnostic inputs before implementing a runner."
    ),
}
write_outputs(report)
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    exit $exitCode
}

