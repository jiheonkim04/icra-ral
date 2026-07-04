param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ZeroActionReportPath = "reports\bounded_libero_robosuite_diagnostic_rollout_report.json",
    [string]$LearnedPolicySummaryPath = "reports\reduced_scope_rollout_metric_summary_report.json",
    [string]$ActionAuditReportPath = "reports\action_interface_metadata_audit_report.json",
    [string]$JsonReportPath = "reports\zero_action_policy_diagnostic_comparison_report.json",
    [string]$MarkdownReportPath = "reports\zero_action_policy_diagnostic_comparison_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Zero-action versus SmolVLA-action diagnostic comparison"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads existing diagnostic reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$zeroFullPath = Resolve-RepoPath -Path $ZeroActionReportPath
$learnedFullPath = Resolve-RepoPath -Path $LearnedPolicySummaryPath
$auditFullPath = Resolve-RepoPath -Path $ActionAuditReportPath
$jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
$markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath

$env:TCA_MAP_ZERO_POLICY_ZERO_REPORT = $zeroFullPath
$env:TCA_MAP_ZERO_POLICY_LEARNED_REPORT = $learnedFullPath
$env:TCA_MAP_ZERO_POLICY_AUDIT_REPORT = $auditFullPath
$env:TCA_MAP_ZERO_POLICY_JSON = $jsonFullPath
$env:TCA_MAP_ZERO_POLICY_MARKDOWN = $markdownFullPath
$env:TCA_MAP_ZERO_POLICY_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import math
import os
from pathlib import Path

ZERO = Path(os.environ["TCA_MAP_ZERO_POLICY_ZERO_REPORT"])
LEARNED = Path(os.environ["TCA_MAP_ZERO_POLICY_LEARNED_REPORT"])
AUDIT = Path(os.environ["TCA_MAP_ZERO_POLICY_AUDIT_REPORT"])
JSON_OUT = Path(os.environ["TCA_MAP_ZERO_POLICY_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_ZERO_POLICY_MARKDOWN"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_ZERO_POLICY_GATES", "").split(";") if item]

def load_json(path):
    if not path.exists():
        return None, f"Missing input report: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:
        return None, f"Could not read input report {path}: {exc}"

def as_bool(value):
    return bool(value) if value is not None else False

def as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def rounded(value):
    if value is None:
        return None
    return round(float(value), 6)

def zero_summary(report):
    rollout = (report or {}).get("rollout_result") or {}
    tasks = rollout.get("task_summaries") or []
    steps = sum(int(as_float(task.get("steps_performed"), 0)) for task in tasks if isinstance(task, dict))
    reward = sum(as_float(task.get("reward_sum"), 0.0) for task in tasks if isinstance(task, dict))
    success = sum(1 for task in tasks if isinstance(task, dict) and as_bool(task.get("success_check")))
    task_names = [task.get("task_name") for task in tasks if isinstance(task, dict)]
    return {
        "source_runner_passed": bool((report or {}).get("bounded_libero_robosuite_diagnostic_rollout_passed")),
        "tasks_observed": len(tasks),
        "tasks_completed": int(as_float(rollout.get("tasks_completed"), len(tasks))),
        "total_steps": steps,
        "diagnostic_success_count": success,
        "diagnostic_success_rate": rounded(success / len(tasks)) if tasks else None,
        "reward_sum_total": rounded(reward),
        "task_names": task_names,
        "simulator_env_created": bool((report or {}).get("policy", {}).get("simulator_environment_created")),
        "learned_policy_inference_performed": bool((report or {}).get("policy", {}).get("learned_policy_inference_performed")),
        "zero_action_policy_only": bool((report or {}).get("policy", {}).get("zero_action_policy_only")),
    }

def learned_summary(report):
    metric = (report or {}).get("metric_summary") or {}
    last_action = metric.get("last_env_action_preview") or []
    values = [as_float(value, 0.0) for value in last_action]
    return {
        "source_runner_passed": bool(metric.get("source_runner_passed")),
        "tasks_observed": int(as_float(metric.get("tasks_observed"), 0)),
        "tasks_completed": int(as_float(metric.get("tasks_completed"), 0)),
        "total_steps": int(as_float(metric.get("total_steps"), 0)),
        "policy_calls": int(as_float(metric.get("policy_calls"), 0)),
        "diagnostic_success_count": int(as_float(metric.get("diagnostic_success_count"), 0)),
        "diagnostic_success_rate": metric.get("diagnostic_success_rate"),
        "reward_sum_total": metric.get("reward_sum_total"),
        "mean_policy_latency_sec": metric.get("mean_policy_latency_sec"),
        "policy_action_shapes": metric.get("policy_action_shapes") or [],
        "env_action_dims": metric.get("env_action_dims") or [],
        "last_env_action_preview": [rounded(value) for value in values],
        "last_env_action_max_abs": metric.get("last_env_action_max_abs"),
        "last_env_action_l2": metric.get("last_env_action_l2") if metric.get("last_env_action_l2") is not None else rounded(math.sqrt(sum(value * value for value in values))) if values else None,
        "last_env_action_gripper_component": metric.get("last_env_action_gripper_component"),
        "adapter_metadata_present": bool(metric.get("adapter_metadata_present")),
        "action_adapter_strategies": metric.get("action_adapter_strategies") or [],
        "state_adapters": metric.get("state_adapters") or [],
        "image_source_keys": metric.get("image_source_keys") or {},
        "action_adapter_implicit_padding_performed": bool(metric.get("action_adapter_implicit_padding_performed")),
        "action_adapter_truncation_performed": bool(metric.get("action_adapter_truncation_performed")),
        "state_adapter_implicit_padding_performed": bool(metric.get("state_adapter_implicit_padding_performed")),
        "state_adapter_silent_truncation_performed": bool(metric.get("state_adapter_silent_truncation_performed")),
        "image_zero_fallback_performed": bool(metric.get("image_zero_fallback_performed")),
        "failure_modes": metric.get("failure_modes") or [],
    }

def write_outputs(report):
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Zero-Action Versus SmolVLA-Action Diagnostic Comparison Report",
        "",
        f"- decision: {report['decision']}",
        f"- comparison passed: {report['zero_action_policy_diagnostic_comparison_passed']}",
        f"- zero-action success rate: {report['comparison']['zero_action']['diagnostic_success_rate']}",
        f"- learned-policy success rate: {report['comparison']['learned_policy']['diagnostic_success_rate']}",
        f"- reward delta learned-minus-zero: {report['comparison']['reward_delta_learned_minus_zero']}",
        f"- policy action nontrivial: {report['comparison']['policy_action_nontrivial']}",
        f"- explicit adapter metadata present: {report['comparison']['explicit_adapter_metadata_present']}",
        f"- learned policy outperformed zero-action: {report['comparison']['learned_policy_outperformed_zero_action']}",
        f"- ready for action/state adapter patch plan: {report['ready_for_action_state_adapter_patch_plan']}",
        f"- ready for adapter strategy diagnosis: {report['ready_for_adapter_strategy_diagnosis']}",
        f"- ready for rollout scaling: {report['ready_for_rollout_scaling']}",
        f"- standard success claimed: {report['claims']['standard_success_claimed']}",
        f"- paper-grade claim made: {report['claims']['paper_grade_claim_made']}",
        "",
        report["recommended_next_step"],
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

policy = {
    "summary_only": True,
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

zero, zero_error = load_json(ZERO)
learned, learned_error = load_json(LEARNED)
audit, audit_error = load_json(AUDIT)

stop_reasons = []
warnings = []
if SET_GATES:
    stop_reasons.append("Refusing comparison while execution gates are set: " + ", ".join(SET_GATES))
if zero_error:
    stop_reasons.append(zero_error)
if learned_error:
    stop_reasons.append(learned_error)
if audit_error:
    warnings.append(audit_error)

zero_stats = zero_summary(zero)
learned_stats = learned_summary(learned)
audit_findings = (audit or {}).get("high_priority_findings") or []

reward_delta = None
if zero_stats["reward_sum_total"] is not None and learned_stats["reward_sum_total"] is not None:
    reward_delta = rounded(as_float(learned_stats["reward_sum_total"]) - as_float(zero_stats["reward_sum_total"]))

success_delta = None
if zero_stats["diagnostic_success_rate"] is not None and learned_stats["diagnostic_success_rate"] is not None:
    success_delta = rounded(as_float(learned_stats["diagnostic_success_rate"]) - as_float(zero_stats["diagnostic_success_rate"]))

policy_action_nontrivial = as_float(learned_stats.get("last_env_action_max_abs"), 0.0) > 0.05
learned_outperformed = (success_delta is not None and success_delta > 0.0) or (reward_delta is not None and reward_delta > 0.0)
both_zero_reward = zero_stats.get("reward_sum_total") == 0.0 and learned_stats.get("reward_sum_total") == 0.0
both_no_success = zero_stats.get("diagnostic_success_count") == 0 and learned_stats.get("diagnostic_success_count") == 0
env_plumbing_passed = zero_stats.get("source_runner_passed") and zero_stats.get("simulator_env_created") and zero_stats.get("total_steps", 0) > 0
adapter_metadata_present = bool(learned_stats.get("adapter_metadata_present"))
adapter_wiring_clean = bool(
    adapter_metadata_present
    and not learned_stats.get("action_adapter_implicit_padding_performed")
    and not learned_stats.get("action_adapter_truncation_performed")
    and not learned_stats.get("state_adapter_implicit_padding_performed")
    and not learned_stats.get("state_adapter_silent_truncation_performed")
    and not learned_stats.get("image_zero_fallback_performed")
)

findings = []
if env_plumbing_passed:
    findings.append({
        "name": "zero_action_simulator_plumbing_passed",
        "severity": "info",
        "evidence": {"zero_action_steps": zero_stats.get("total_steps"), "simulator_env_created": zero_stats.get("simulator_env_created")},
        "recommendation": "Treat basic reset/step/render plumbing as less likely than policy-interface mismatch.",
    })
if policy_action_nontrivial and both_zero_reward and both_no_success:
    findings.append({
        "name": "nontrivial_policy_actions_do_not_improve_over_zero_action",
        "severity": "high",
        "evidence": {
            "policy_action_max_abs": learned_stats.get("last_env_action_max_abs"),
            "reward_delta_learned_minus_zero": reward_delta,
            "success_delta_learned_minus_zero": success_delta,
        },
        "recommendation": (
            "Inspect adapter strategy, action scale, gripper semantics, prompt, and camera mapping before rollout scaling."
            if adapter_metadata_present
            else "Create an explicit action/state adapter patch plan before further rollout scaling."
        ),
    })
if adapter_metadata_present:
    findings.append({
        "name": "explicit_adapter_metadata_present_but_zero_reward",
        "severity": "high" if both_zero_reward and both_no_success else "medium",
        "evidence": {
            "action_adapter_strategies": learned_stats.get("action_adapter_strategies"),
            "state_adapters": learned_stats.get("state_adapters"),
            "image_source_keys": learned_stats.get("image_source_keys"),
            "adapter_wiring_clean": adapter_wiring_clean,
        },
        "recommendation": "Treat the next step as adapter-strategy and action-scale diagnosis, not another pure wiring patch.",
    })
if (not adapter_metadata_present) and ("action_dim_mismatch" in audit_findings or "gripper_constant_zero" in audit_findings):
    findings.append({
        "name": "audit_supports_action_adapter_work",
        "severity": "high",
        "evidence": {"audit_high_priority_findings": audit_findings},
        "recommendation": "Resolve 6D policy action to 7D environment action mapping and gripper semantics.",
    })

passed = bool(not stop_reasons and zero_stats["source_runner_passed"] and learned_stats["source_runner_passed"] and env_plumbing_passed)
decision = "proceed" if passed else "stop"
reason = "Compared existing zero-action and SmolVLA-action diagnostics without executing new rollout." if passed else "Comparison prerequisites are not satisfied."
ready_for_patch_plan = passed and any(item["severity"] == "high" for item in findings)
ready_for_patch_plan = bool(ready_for_patch_plan and not adapter_metadata_present)
ready_for_adapter_strategy_diagnosis = bool(passed and adapter_metadata_present and not learned_outperformed)
ready_for_rollout_scaling = bool(passed and learned_outperformed and not ready_for_patch_plan and not ready_for_adapter_strategy_diagnosis)

comparison = {
    "zero_action": zero_stats,
    "learned_policy": learned_stats,
    "same_task_names": sorted(set(zero_stats.get("task_names") or []) & {item.get("task_name") for item in learned_stats.get("failure_modes") or [] if isinstance(item, dict)}),
    "reward_delta_learned_minus_zero": reward_delta,
    "success_delta_learned_minus_zero": success_delta,
    "policy_action_nontrivial": policy_action_nontrivial,
    "explicit_adapter_metadata_present": adapter_metadata_present,
    "adapter_wiring_clean": adapter_wiring_clean,
    "learned_policy_outperformed_zero_action": learned_outperformed,
    "both_zero_reward": both_zero_reward,
    "both_no_success": both_no_success,
    "zero_action_env_plumbing_passed": env_plumbing_passed,
}

report = {
    "zero_action_policy_diagnostic_comparison_passed": passed,
    "decision": decision,
    "reason": reason,
    "source_reports": {
        "zero_action_report": str(ZERO),
        "learned_policy_summary": str(LEARNED),
        "action_interface_audit": str(AUDIT),
    },
    "evidence_label": "zero_action_vs_smolvla_action_diagnostic_comparison",
    "policy": policy,
    "claims": claims,
    "comparison": comparison,
    "findings": findings,
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_action_state_adapter_patch_plan": ready_for_patch_plan,
    "ready_for_adapter_strategy_diagnosis": ready_for_adapter_strategy_diagnosis,
    "ready_for_rollout_scaling": ready_for_rollout_scaling,
    "recommended_next_step": (
        "Run adapter-strategy/action-scale diagnostics before rollout scaling."
        if ready_for_adapter_strategy_diagnosis
        else (
            "Create an explicit action/state adapter patch plan before rollout scaling."
            if ready_for_patch_plan
            else "Fix missing comparison inputs before planning action/state adapter changes."
        )
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
