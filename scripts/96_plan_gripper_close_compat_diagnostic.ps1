param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$OfflineReproductionReportPath = "reports\offline_adapter_reproduction_check_report.json",
    [string]$PreviousAdapterStrategyReportPath = "reports\adapter_strategy_action_scale_diagnostic_report.json",
    [string]$RolloutBridgeSourcePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$JsonReportPath = "reports\gripper_close_compat_diagnostic_plan_report.json",
    [string]$MarkdownReportPath = "reports\gripper_close_compat_diagnostic_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Gripper-close compatibility diagnostic planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This planner reads local reports and source files only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
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
    "ALLOW_CAMERA_SOURCE_DIAGNOSTIC",
    "ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC",
    "ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC",
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

$env:TCA_MAP_GRIPPER_CLOSE_REPRO = Resolve-RepoPath -Path $OfflineReproductionReportPath
$env:TCA_MAP_GRIPPER_CLOSE_PREVIOUS = Resolve-RepoPath -Path $PreviousAdapterStrategyReportPath
$env:TCA_MAP_GRIPPER_CLOSE_SOURCE = Resolve-RepoPath -Path $RolloutBridgeSourcePath
$env:TCA_MAP_GRIPPER_CLOSE_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_GRIPPER_CLOSE_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_GRIPPER_CLOSE_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

REPRO = Path(os.environ["TCA_MAP_GRIPPER_CLOSE_REPRO"])
PREVIOUS = Path(os.environ["TCA_MAP_GRIPPER_CLOSE_PREVIOUS"])
SOURCE = Path(os.environ["TCA_MAP_GRIPPER_CLOSE_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_GRIPPER_CLOSE_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_GRIPPER_CLOSE_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_GRIPPER_CLOSE_GATES", "").split(";") if item]

CLOSE = "policy_6d_delta_pose_plus_gripper_close"
ZERO_HOLD = "policy_6d_delta_pose_plus_gripper_zero_hold"


def load_json(path: Path, *, required: bool = True):
    if not path.exists():
        return None, f"Missing input report: {path}" if required else None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parse issue belongs in report.
        return None, f"Could not read {path}: {exc}"


def find_strategy_variant(previous: dict | None, strategy: str) -> dict | None:
    if not previous:
        return None
    variants = previous.get("variants") or []
    for item in variants:
        if item.get("strategy") == strategy:
            return item
    return None


def write_outputs(report: dict) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gripper-Close Compatibility Diagnostic Plan Report",
        "",
        f"- decision: {report['decision']}",
        f"- plan passed: {report['gripper_close_compat_plan_passed']}",
        f"- ready for bounded runner: {report['ready_for_gripper_close_compat_diagnostic_runner']}",
        f"- offline best strategy: {report['offline_evidence']['best_strategy']}",
        f"- close first-action L1: {report['offline_evidence']['close_l1_to_demo_first_action']}",
        f"- zero-hold first-action L1: {report['offline_evidence']['zero_hold_l1_to_demo_first_action']}",
        f"- previous close diagnostic found: {report['previous_diagnostic']['close_variant_found']}",
        f"- previous close reward sum: {report['previous_diagnostic']['close_reward_sum']}",
        f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
        f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
        "",
        report["recommended_next_step"],
        "",
        "This is a planning report only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


repro, repro_error = load_json(REPRO, required=True)
previous, previous_error = load_json(PREVIOUS, required=False)
source_text = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""

stop_reasons: list[str] = []
warnings: list[str] = []
if SET_GATES:
    stop_reasons.append("Refusing gripper-close compatibility planning while execution gates are set: " + ", ".join(SET_GATES))
if repro_error:
    stop_reasons.append(repro_error)
if previous_error:
    warnings.append(previous_error)
if not SOURCE.exists():
    stop_reasons.append(f"Missing rollout bridge source: {SOURCE}")

repro = repro or {}
reproductions = ((repro.get("reproduction") or {}).get("action_reproductions") or {})
close_repro = reproductions.get(CLOSE) or {}
zero_repro = reproductions.get(ZERO_HOLD) or {}
best_strategy = (repro.get("reproduction") or {}).get("best_action_adapter_strategy_for_first_demo_action")
close_l1 = close_repro.get("l1_to_demo_first_action")
zero_l1 = zero_repro.get("l1_to_demo_first_action")

source_has_strategy_arg = "--action-adapter-strategy" in source_text
source_has_close_choice = CLOSE in source_text or "ACTION_STRATEGY_GRIPPER_CLOSE" in source_text
offline_passed = bool(repro.get("offline_adapter_reproduction_check_passed"))
offline_close_exact = isinstance(close_l1, (int, float)) and close_l1 <= 1e-6
offline_close_better = (
    isinstance(close_l1, (int, float))
    and isinstance(zero_l1, (int, float))
    and close_l1 < zero_l1
)

if not offline_passed:
    stop_reasons.append("Offline adapter reproduction check did not pass.")
if best_strategy != CLOSE:
    stop_reasons.append("Offline adapter reproduction did not select the gripper-close strategy as best.")
if not offline_close_better:
    stop_reasons.append("Gripper-close did not improve over zero-hold in the offline first-action reproduction check.")
if not source_has_strategy_arg or not source_has_close_choice:
    stop_reasons.append("Rollout bridge does not expose the gripper-close action-adapter strategy.")

previous_close = find_strategy_variant(previous, CLOSE)
previous_close_found = previous_close is not None
previous_close_success = None
previous_close_reward = None
previous_close_passed = None
if previous_close_found:
    previous_close_success = previous_close.get("diagnostic_success_rate")
    previous_close_reward = previous_close.get("reward_sum")
    previous_close_passed = previous_close.get("passed")

duplicate_zero_signal = (
    previous_close_found
    and bool(previous_close_passed)
    and isinstance(previous_close_success, (int, float))
    and isinstance(previous_close_reward, (int, float))
    and previous_close_success <= 0.0
    and previous_close_reward <= 0.0
)
if duplicate_zero_signal:
    warnings.append(
        "A prior gripper-close bounded diagnostic already executed cleanly but produced zero diagnostic success and zero reward."
    )

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

if stop_reasons:
    decision = "stop"
    plan_passed = False
    ready_runner = False
    reason = "Required offline evidence or rollout-bridge support is missing."
    next_step = "Fix the missing report/source prerequisites before any gripper-close compatibility diagnostic."
elif duplicate_zero_signal:
    decision = "reduce_scope"
    plan_passed = True
    ready_runner = False
    reason = "Offline evidence supports gripper-close, but an equivalent close-strategy rollout diagnostic already produced zero signal."
    next_step = (
        "Do not rerun the identical close-strategy diagnostic. Plan a narrower HDF5-aligned compatibility check "
        "that compares task selection, initial-state convention, and replay/action-sign assumptions before another rollout."
    )
else:
    decision = "proceed"
    plan_passed = True
    ready_runner = True
    reason = "Offline HDF5 evidence identifies gripper-close as a specific bounded compatibility hypothesis."
    next_step = (
        "Run a separately gated one-task gripper-close compatibility diagnostic only under "
        "ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC=1, capped at one task and 10 steps, diagnostic evidence only."
    )

report = {
    "evidence_label": "gripper_close_compat_diagnostic_plan",
    "gripper_close_compat_plan_passed": plan_passed,
    "decision": decision,
    "reason": reason,
    "source_reports": {
        "offline_adapter_reproduction": str(REPRO),
        "previous_adapter_strategy_diagnostic": str(PREVIOUS),
        "rollout_bridge_source": str(SOURCE),
    },
    "policy": policy,
    "claims": claims,
    "offline_evidence": {
        "offline_adapter_reproduction_check_passed": offline_passed,
        "best_strategy": best_strategy,
        "close_l1_to_demo_first_action": close_l1,
        "zero_hold_l1_to_demo_first_action": zero_l1,
        "close_exact_first_action": offline_close_exact,
        "close_better_than_zero_hold": offline_close_better,
        "hdf5_path": (repro.get("paths") or {}).get("hdf5_path"),
    },
    "previous_diagnostic": {
        "close_variant_found": previous_close_found,
        "close_variant_passed": previous_close_passed,
        "close_diagnostic_success_rate": previous_close_success,
        "close_reward_sum": previous_close_reward,
        "duplicate_zero_signal": duplicate_zero_signal,
    },
    "risk_assessment": {
        "task": "bounded gripper-close compatibility diagnostic planning",
        "future_command": "future separately gated one-task runner",
        "source": "local offline LIBERO HDF5 reproduction plus local rollout bridge source",
        "expected_size_gb": 0,
        "expected_runtime_minutes_if_run": 15,
        "expected_ram_gb_if_run": 8,
        "expected_vram_gb_if_run": 0,
        "task_count_if_run": 1,
        "max_steps_if_run": 10,
        "token_login_license_payment_needed": False,
        "simulator_will_run_in_future_runner": True,
        "learned_policy_inference_will_run_in_future_runner": True,
        "training_will_run": False,
        "gpu_job_will_run": False,
        "openvla_oft_will_run": False,
        "paper_claim_will_be_made": False,
        "decision": decision,
        "reason": reason,
    },
    "runner_plan": {
        "task_local_gate": "ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC=1",
        "max_tasks": 1,
        "max_steps": 10,
        "action_adapter_strategy": CLOSE,
        "evidence_label": "bounded_gripper_close_compat_diagnostic",
        "acceptance_checks": [
            "no downloads",
            "no installs",
            "no training",
            "no GPU jobs",
            "no OpenVLA-OFT",
            "no multi-seed evaluation",
            "diagnostic label only",
            "report adapter metadata and reward/success separately from wrapper pass",
        ],
    },
    "source_support": {
        "source_has_action_adapter_strategy_argument": source_has_strategy_arg,
        "source_has_gripper_close_choice": source_has_close_choice,
    },
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_gripper_close_compat_diagnostic_runner": ready_runner,
    "ready_for_rollout_scaling": False,
    "ready_for_paper_claim": False,
    "recommended_next_step": next_step,
}
write_outputs(report)
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
