param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ActionScaleReportPath = "reports\action_scale_diagnostic_report.json",
    [string]$RolloutBridgeSourcePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$JsonReportPath = "reports\prompt_format_diagnostic_plan_report.json",
    [string]$MarkdownReportPath = "reports\prompt_format_diagnostic_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Prompt-format diagnostic planner"
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
    "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
    "ALLOW_ACTION_SCALE_DIAGNOSTIC",
    "ALLOW_PROMPT_FORMAT_DIAGNOSTIC",
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

$env:TCA_MAP_PROMPT_ACTION_SCALE_REPORT = Resolve-RepoPath -Path $ActionScaleReportPath
$env:TCA_MAP_PROMPT_SOURCE = Resolve-RepoPath -Path $RolloutBridgeSourcePath
$env:TCA_MAP_PROMPT_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_PROMPT_MARKDOWN = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_PROMPT_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

ACTION_SCALE_REPORT = Path(os.environ["TCA_MAP_PROMPT_ACTION_SCALE_REPORT"])
SOURCE = Path(os.environ["TCA_MAP_PROMPT_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_PROMPT_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_PROMPT_MARKDOWN"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_PROMPT_GATES", "").split(";") if item]

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
        "# Prompt-Format Diagnostic Plan Report",
        "",
        f"- decision: {report['decision']}",
        f"- planner passed: {report['prompt_format_diagnostic_plan_passed']}",
        f"- ready for runner: {report['ready_for_prompt_format_diagnostic_runner']}",
        f"- ready for rollout scaling: {report['ready_for_rollout_scaling']}",
        f"- source action-scale passed: {report['inputs']['action_scale_diagnostic_passed']}",
        f"- planned prompt variants: {report['diagnostic_plan']['prompt_strategy_variants']}",
        "",
        report["recommended_next_step"],
        "",
        "This is planning evidence only. It is not benchmark success, standard success, SOTA evidence, or paper-grade evidence.",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

scale_report, scale_error = load_json(ACTION_SCALE_REPORT)
source_text = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""
stop_reasons = []
if SET_GATES:
    stop_reasons.append("Refusing prompt-format planning while execution gates are set: " + ", ".join(SET_GATES))
if scale_error:
    stop_reasons.append(scale_error)
if not SOURCE.exists():
    stop_reasons.append(f"Missing rollout bridge source: {SOURCE}")

action_scale_passed = bool((scale_report or {}).get("action_scale_diagnostic_passed"))
result = (scale_report or {}).get("result") or {}
source_has_prompt_strategy = "--prompt-strategy" in source_text and "args.prompt_strategy" in source_text
source_has_bddl_language = "PROMPT_STRATEGY_BDDL_LANGUAGE" in source_text and "_read_bddl_language" in source_text

if not action_scale_passed:
    stop_reasons.append("Action-scale diagnostic has not passed.")
if not source_has_prompt_strategy:
    stop_reasons.append("Rollout bridge does not expose the prompt-strategy CLI and task-language hook.")
if not source_has_bddl_language:
    stop_reasons.append("Rollout bridge does not expose BDDL language prompt parsing.")

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
    "evidence_label": "prompt_format_diagnostic_plan",
    "max_tasks": 1,
    "max_steps_per_variant": 10,
    "max_variants_first_runner": 3,
    "expected_runtime_minutes": 15,
    "expected_vram_gb": 0,
    "action_adapter_strategy": "policy_6d_delta_pose_plus_gripper_zero_hold",
    "action_scale": 1.0,
    "prompt_strategy_variants": ["stem_spaces", "bddl_language", "bddl_language_period"],
    "acceptance_checks": [
        "no downloads",
        "no installs",
        "no training",
        "no GPU jobs",
        "no OpenVLA-OFT",
        "one task only",
        "at most 10 steps per variant",
        "prompt_strategy and language recorded for every variant",
        "results labeled diagnostic only",
    ],
}

ready = bool(not stop_reasons)
report = {
    "prompt_format_diagnostic_plan_passed": ready,
    "decision": "proceed" if ready else "stop",
    "reason": (
        "Action-scale diagnostic passed with zero reward; bounded prompt-format diagnostic runner is ready."
        if ready
        else "Prompt-format diagnostic prerequisites are not satisfied."
    ),
    "source_reports": {
        "action_scale_diagnostic": str(ACTION_SCALE_REPORT),
        "rollout_bridge_source": str(SOURCE),
    },
    "policy": policy,
    "claims": claims,
    "inputs": {
        "action_scale_diagnostic_passed": action_scale_passed,
        "variants_completed": result.get("variants_completed"),
        "best_action_scale": result.get("best_action_scale"),
        "best_diagnostic_success_rate": result.get("best_diagnostic_success_rate"),
        "best_reward_sum": result.get("best_reward_sum"),
        "source_has_prompt_strategy": source_has_prompt_strategy,
        "source_has_bddl_language": source_has_bddl_language,
    },
    "risk_assessment": {
        "task": "bounded prompt-format diagnostic",
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
    "stop_reasons": stop_reasons,
    "ready_for_prompt_format_diagnostic_runner": ready,
    "ready_for_rollout_scaling": False,
    "recommended_next_step": (
        "Run a separately gated one-task prompt-format diagnostic runner; do not scale rollout or make claims."
        if ready
        else "Fix missing prompt-format diagnostic inputs before running a bounded prompt-format diagnostic."
    ),
}
write_outputs(report)
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
