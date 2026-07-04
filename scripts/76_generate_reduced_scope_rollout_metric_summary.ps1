param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$InputReportPath = "reports\bounded_reduced_scope_learned_policy_rollout_report.json",
    [string]$JsonReportPath = "reports\reduced_scope_rollout_metric_summary_report.json",
    [string]$MarkdownReportPath = "reports\reduced_scope_rollout_metric_summary_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Reduced-scope rollout metric summary"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads an existing reduced-scope rollout report only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$inputFullPath = if ([System.IO.Path]::IsPathRooted($InputReportPath)) { $InputReportPath } else { Join-Path $RepoRoot $InputReportPath }
$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }

$env:TCA_MAP_REDUCED_SCOPE_SUMMARY_INPUT = $inputFullPath
$env:TCA_MAP_REDUCED_SCOPE_SUMMARY_JSON = $jsonFullPath
$env:TCA_MAP_REDUCED_SCOPE_SUMMARY_MARKDOWN = $markdownFullPath
$env:TCA_MAP_REDUCED_SCOPE_SUMMARY_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import math
import os
import statistics
from pathlib import Path

INPUT = Path(os.environ["TCA_MAP_REDUCED_SCOPE_SUMMARY_INPUT"])
JSON_OUT = Path(os.environ["TCA_MAP_REDUCED_SCOPE_SUMMARY_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_REDUCED_SCOPE_SUMMARY_MARKDOWN"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_REDUCED_SCOPE_SUMMARY_GATES", "").split(";") if item]

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

def load_source(path):
    if not path.exists():
        return None, f"Missing input report: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:
        return None, f"Could not read input report: {exc}"

def write_outputs(report):
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = report["metric_summary"]
    lines = [
        "# Reduced-Scope Rollout Metric Summary Report",
        "",
        f"- decision: {report['decision']}",
        f"- evidence label: {report['evidence_label']}",
        f"- source runner passed: {s.get('source_runner_passed')}",
        f"- tasks completed: {s.get('tasks_completed')}",
        f"- total steps: {s.get('total_steps')}",
        f"- policy calls: {s.get('policy_calls')}",
        f"- diagnostic success count: {s.get('diagnostic_success_count')}",
        f"- diagnostic success rate: {s.get('diagnostic_success_rate')}",
        f"- reward sum: {s.get('reward_sum_total')}",
        f"- mean policy latency sec: {s.get('mean_policy_latency_sec')}",
        f"- action max abs: {s.get('last_env_action_max_abs')}",
        f"- action L2: {s.get('last_env_action_l2')}",
        f"- gripper component: {s.get('last_env_action_gripper_component')}",
        f"- failure modes: {s.get('failure_modes')}",
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

source, load_error = load_source(INPUT)
stop_reasons = []
decision = "proceed"
reason = "Reduced-scope learned-policy diagnostic metrics summarized from the existing rollout report."

if SET_GATES:
    decision = "stop"
    reason = "Refusing summary generation while execution gates are set: " + ", ".join(SET_GATES)
    stop_reasons.append(reason)
if load_error:
    decision = "stop"
    reason = load_error
    stop_reasons.append(load_error)

source_runner_passed = bool(source.get("bounded_reduced_scope_learned_policy_rollout_passed")) if isinstance(source, dict) else False
inner = (source or {}).get("rollout_result") or {}
tasks = inner.get("tasks") or []
result = inner.get("result") or {}

policy_calls = sum(int(as_float(task.get("policy_calls"), 0)) for task in tasks if isinstance(task, dict))
steps = sum(int(as_float(task.get("steps_performed"), 0)) for task in tasks if isinstance(task, dict))
reward_sum = sum(as_float(task.get("reward_sum"), 0.0) for task in tasks if isinstance(task, dict))
success_count = sum(1 for task in tasks if isinstance(task, dict) and as_bool(task.get("success_check")))
latencies = [as_float(task.get("last_inference_sec"), None) for task in tasks if isinstance(task, dict) and task.get("last_inference_sec") is not None]
policy_action_shapes = sorted({tuple(task.get("last_policy_action_shape") or []) for task in tasks if isinstance(task, dict)})
env_action_dims = sorted({int(as_float(task.get("action_dim"), 0)) for task in tasks if isinstance(task, dict) and task.get("action_dim") is not None})
last_actions = [task.get("last_env_action_preview") for task in tasks if isinstance(task, dict) and isinstance(task.get("last_env_action_preview"), list)]
last_action = last_actions[-1] if last_actions else []
last_action_values = [as_float(value, 0.0) for value in last_action]
last_action_max_abs = max((abs(value) for value in last_action_values), default=None)
last_action_l2 = math.sqrt(sum(value * value for value in last_action_values)) if last_action_values else None
last_gripper = last_action_values[-1] if last_action_values else None

failure_modes = []
for task in tasks:
    if not isinstance(task, dict):
        continue
    if task.get("error"):
        failure_modes.append({"task_name": task.get("task_name"), "failure": str(task.get("error"))})
    elif not as_bool(task.get("success_check")):
        failure_modes.append({"task_name": task.get("task_name"), "failure": "diagnostic_success_check_false"})

tasks_completed = int(as_float(result.get("tasks_completed"), len(tasks)))
summary_passed = bool(decision == "proceed" and source_runner_passed and tasks and policy_calls > 0 and steps > 0)
if decision == "proceed" and not summary_passed:
    decision = "stop"
    reason = "Existing reduced-scope rollout report is not sufficient for a passed metric summary."
    stop_reasons.append(reason)

metric_summary = {
    "source_report_exists": INPUT.exists(),
    "source_runner_passed": source_runner_passed,
    "tasks_observed": len(tasks),
    "tasks_completed": tasks_completed,
    "total_steps": steps,
    "policy_calls": policy_calls,
    "diagnostic_success_count": success_count,
    "diagnostic_success_rate": rounded(success_count / len(tasks)) if tasks else None,
    "reward_sum_total": rounded(reward_sum),
    "mean_policy_latency_sec": rounded(statistics.mean(latencies)) if latencies else None,
    "max_policy_latency_sec": rounded(max(latencies)) if latencies else None,
    "policy_action_shapes": [list(shape) for shape in policy_action_shapes],
    "env_action_dims": env_action_dims,
    "last_env_action_preview": [rounded(value) for value in last_action_values],
    "last_env_action_max_abs": rounded(last_action_max_abs) if last_action_max_abs is not None else None,
    "last_env_action_l2": rounded(last_action_l2) if last_action_l2 is not None else None,
    "last_env_action_gripper_component": rounded(last_gripper) if last_gripper is not None else None,
    "failure_modes": failure_modes,
}

report = {
    "reduced_scope_rollout_metric_summary_passed": summary_passed,
    "decision": decision,
    "reason": reason,
    "source_report_path": str(INPUT),
    "evidence_label": "reduced_scope_learned_policy_diagnostic",
    "policy": policy,
    "claims": claims,
    "metric_summary": metric_summary,
    "stop_reasons": stop_reasons,
    "recommended_next_step": (
        "Run action-interface diagnostics before scaling: action normalization, gripper mapping, observation/state mapping, language prompt, and camera mapping."
        if summary_passed
        else "Rerun the reduced-scope learned-policy rollout or inspect the input report before planning action-interface diagnostics."
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
