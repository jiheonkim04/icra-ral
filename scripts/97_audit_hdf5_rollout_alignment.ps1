param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$GripperClosePlanReportPath = "reports\gripper_close_compat_diagnostic_plan_report.json",
    [string]$OfflineReproductionReportPath = "reports\offline_adapter_reproduction_check_report.json",
    [string]$PreviousAdapterStrategyReportPath = "reports\adapter_strategy_action_scale_diagnostic_report.json",
    [string]$Hdf5AuditReportPath = "reports\libero_hdf5_interface_audit_report.json",
    [string]$RolloutBridgeSourcePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$JsonReportPath = "reports\hdf5_rollout_alignment_audit_report.json",
    [string]$MarkdownReportPath = "reports\hdf5_rollout_alignment_audit_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "HDF5-to-rollout alignment audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This audit reads local reports, one local HDF5 demonstration, and source files only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
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

$env:TCA_MAP_HDF5_ALIGN_GRIPPER_PLAN = Resolve-RepoPath -Path $GripperClosePlanReportPath
$env:TCA_MAP_HDF5_ALIGN_REPRO = Resolve-RepoPath -Path $OfflineReproductionReportPath
$env:TCA_MAP_HDF5_ALIGN_PREVIOUS = Resolve-RepoPath -Path $PreviousAdapterStrategyReportPath
$env:TCA_MAP_HDF5_ALIGN_AUDIT = Resolve-RepoPath -Path $Hdf5AuditReportPath
$env:TCA_MAP_HDF5_ALIGN_SOURCE = Resolve-RepoPath -Path $RolloutBridgeSourcePath
$env:TCA_MAP_HDF5_ALIGN_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_HDF5_ALIGN_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_HDF5_ALIGN_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

GRIPPER_PLAN = Path(os.environ["TCA_MAP_HDF5_ALIGN_GRIPPER_PLAN"])
REPRO = Path(os.environ["TCA_MAP_HDF5_ALIGN_REPRO"])
PREVIOUS = Path(os.environ["TCA_MAP_HDF5_ALIGN_PREVIOUS"])
HDF5_AUDIT = Path(os.environ["TCA_MAP_HDF5_ALIGN_AUDIT"])
SOURCE = Path(os.environ["TCA_MAP_HDF5_ALIGN_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_HDF5_ALIGN_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_HDF5_ALIGN_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_HDF5_ALIGN_GATES", "").split(";") if item]

CLOSE = "policy_6d_delta_pose_plus_gripper_close"


def load_json(path: Path, *, required: bool = True):
    if not path.exists():
        return None, f"Missing input report: {path}" if required else None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parse issue belongs in report.
        return None, f"Could not read {path}: {exc}"


def find_close_variant(previous: dict | None) -> dict | None:
    if not previous:
        return None
    for item in previous.get("variants") or []:
        if item.get("strategy") == CLOSE:
            return item
    return None


def first_task_from_variant(variant: dict | None) -> dict:
    if not variant:
        return {}
    inner = variant.get("inner_report") or {}
    tasks = inner.get("tasks") or []
    return tasks[0] if tasks else {}


def hdf5_task_stem(path: Path) -> str:
    name = path.name
    for suffix in ("_demo.hdf5", ".hdf5"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def write_outputs(report: dict) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HDF5-to-Rollout Alignment Audit Report",
        "",
        f"- decision: {report['decision']}",
        f"- audit passed: {report['hdf5_rollout_alignment_audit_passed']}",
        f"- task name matches: {report['alignment']['task_name_matches']}",
        f"- HDF5 init state present: {report['hdf5_demo']['init_state_present']}",
        f"- rollout sets HDF5 initial state: {report['rollout_bridge']['source_sets_hdf5_initial_state']}",
        f"- previous close duplicate zero signal: {report['previous_close_diagnostic']['duplicate_zero_signal']}",
        f"- ready for HDF5 initial-state replay plan: {report['ready_for_hdf5_initial_state_replay_plan']}",
        f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
        f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
        "",
        report["recommended_next_step"],
        "",
        "This is a report-only alignment audit. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


stop_reasons: list[str] = []
warnings: list[str] = []
if SET_GATES:
    stop_reasons.append("Refusing HDF5/rollout alignment audit while execution gates are set: " + ", ".join(SET_GATES))

try:
    import h5py
except Exception as exc:  # noqa: BLE001
    h5py = None
    stop_reasons.append(f"h5py is required for report-only HDF5 inspection: {exc}")

gripper_plan, gripper_error = load_json(GRIPPER_PLAN, required=False)
repro, repro_error = load_json(REPRO, required=True)
previous, previous_error = load_json(PREVIOUS, required=True)
hdf5_audit, hdf5_audit_error = load_json(HDF5_AUDIT, required=True)
source_text = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""

if gripper_error:
    warnings.append(gripper_error)
for error in (repro_error, previous_error, hdf5_audit_error):
    if error:
        stop_reasons.append(error)
if not SOURCE.exists():
    stop_reasons.append(f"Missing rollout bridge source: {SOURCE}")

repro = repro or {}
previous = previous or {}
hdf5_audit = hdf5_audit or {}
gripper_plan = gripper_plan or {}

hdf5_path = Path((repro.get("paths") or {}).get("hdf5_path") or "")
if not str(hdf5_path) or not hdf5_path.exists():
    stop_reasons.append(f"Missing HDF5 demonstration file from offline reproduction report: {hdf5_path}")

close_variant = find_close_variant(previous)
close_task = first_task_from_variant(close_variant)
rollout_task_name = close_task.get("task_name")
rollout_bddl_file = close_task.get("bddl_file")
hdf5_stem = hdf5_task_stem(hdf5_path) if str(hdf5_path) else None
task_name_matches = bool(hdf5_stem and rollout_task_name and hdf5_stem == rollout_task_name)
if hdf5_stem and rollout_task_name and not task_name_matches:
    stop_reasons.append("Offline HDF5 task and previous close diagnostic rollout task do not match.")

hdf5_demo = {
    "hdf5_path": str(hdf5_path) if str(hdf5_path) else None,
    "task_stem": hdf5_stem,
    "demo_name": None,
    "init_state_present": False,
    "init_state_shape": None,
    "states_present": False,
    "states_shape": None,
    "actions_shape": None,
    "first_action": None,
    "first_gripper_action": None,
    "obs_keys": [],
    "model_file_attr_present": False,
}
if not stop_reasons and h5py is not None:
    with h5py.File(hdf5_path, "r") as handle:
        demo_name = sorted(handle["data"].keys())[0]
        demo = handle["data"][demo_name]
        init_state = demo.attrs.get("init_state")
        states = demo.get("states")
        actions = demo.get("actions")
        obs = demo.get("obs")
        first_action = actions[0].tolist() if actions is not None and actions.shape[0] > 0 else None
        hdf5_demo.update(
            {
                "demo_name": demo_name,
                "init_state_present": init_state is not None,
                "init_state_shape": list(getattr(init_state, "shape", [])) if init_state is not None else None,
                "states_present": states is not None,
                "states_shape": list(states.shape) if states is not None else None,
                "actions_shape": list(actions.shape) if actions is not None else None,
                "first_action": [float(x) for x in first_action] if first_action is not None else None,
                "first_gripper_action": float(first_action[-1]) if first_action is not None else None,
                "obs_keys": sorted(list(obs.keys())) if obs is not None else [],
                "model_file_attr_present": "model_file" in demo.attrs,
            }
        )

source_sets_hdf5_initial_state = any(
    token in source_text
    for token in (
        "set_init_state",
        "set_initial_state",
        "init_state",
        "set_state_from_flattened",
        "sim.set_state",
    )
)
source_uses_plain_reset = "env.reset()" in source_text
previous_close_passed = bool(close_variant.get("passed")) if close_variant else False
previous_close_success = close_variant.get("diagnostic_success_rate") if close_variant else None
previous_close_reward = close_variant.get("reward_sum") if close_variant else None
duplicate_zero_signal = (
    previous_close_passed
    and isinstance(previous_close_success, (int, float))
    and isinstance(previous_close_reward, (int, float))
    and previous_close_success <= 0.0
    and previous_close_reward <= 0.0
)
init_state_gap = bool(hdf5_demo["init_state_present"] and not source_sets_hdf5_initial_state)

issues = []
if task_name_matches:
    issues.append(
        {
            "severity": "low",
            "axis": "task_selection",
            "finding": "The offline HDF5 demonstration task matches the previous close-strategy rollout task.",
            "recommendation": "Treat task selection as less likely than initial-state/control-convention mismatch for this specific diagnostic.",
        }
    )
if duplicate_zero_signal:
    issues.append(
        {
            "severity": "high",
            "axis": "duplicate_close_rollout",
            "finding": "A gripper-close rollout variant already executed but still produced zero reward and zero diagnostic success.",
            "recommendation": "Do not rerun the same close variant without changing the compatibility hypothesis.",
        }
    )
if init_state_gap:
    issues.append(
        {
            "severity": "high",
            "axis": "hdf5_initial_state_alignment",
            "finding": "The HDF5 demonstration has an init_state/states trajectory, but the rollout bridge uses reset without evidence of setting the HDF5 initial state.",
            "recommendation": "Plan a bounded HDF5 initial-state or first-action replay diagnostic before another learned-policy rollout.",
        }
    )

policy = {
    "report_only": True,
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
    passed = False
    ready_replay_plan = False
    reason = "Required HDF5/rollout alignment inputs are missing or inconsistent."
    next_step = "Fix report/HDF5 task alignment prerequisites before any replay or rollout diagnostic."
elif init_state_gap or duplicate_zero_signal:
    decision = "reduce_scope"
    passed = True
    ready_replay_plan = True
    reason = "The task appears aligned, but HDF5 initial-state/replay alignment is not established and the close rollout already had zero signal."
    next_step = (
        "Create a planning-only HDF5 initial-state/first-action replay diagnostic. It should inspect whether LIBERO can safely "
        "set the demo init_state and step the first demonstration action under a task-local gate before learned-policy rollout scaling."
    )
else:
    decision = "proceed"
    passed = True
    ready_replay_plan = False
    reason = "No HDF5/rollout alignment blocker was found in report-only inputs."
    next_step = "Proceed only to a separately risk-assessed diagnostic runner; keep rollout scaling blocked until positive task signal appears."

report = {
    "evidence_label": "hdf5_rollout_alignment_audit",
    "hdf5_rollout_alignment_audit_passed": passed,
    "decision": decision,
    "reason": reason,
    "source_reports": {
        "gripper_close_plan": str(GRIPPER_PLAN),
        "offline_adapter_reproduction": str(REPRO),
        "previous_adapter_strategy_diagnostic": str(PREVIOUS),
        "hdf5_interface_audit": str(HDF5_AUDIT),
        "rollout_bridge_source": str(SOURCE),
    },
    "policy": policy,
    "claims": claims,
    "alignment": {
        "hdf5_task_stem": hdf5_stem,
        "rollout_task_name": rollout_task_name,
        "rollout_bddl_file": rollout_bddl_file,
        "task_name_matches": task_name_matches,
        "gripper_close_plan_decision": gripper_plan.get("decision"),
        "hdf5_audit_decision": hdf5_audit.get("decision"),
    },
    "hdf5_demo": hdf5_demo,
    "previous_close_diagnostic": {
        "close_variant_found": close_variant is not None,
        "close_variant_passed": previous_close_passed,
        "close_diagnostic_success_rate": previous_close_success,
        "close_reward_sum": previous_close_reward,
        "duplicate_zero_signal": duplicate_zero_signal,
        "task_summary_has_initial_state_set": "initial_state_set" in close_task,
    },
    "rollout_bridge": {
        "source_uses_plain_reset": source_uses_plain_reset,
        "source_sets_hdf5_initial_state": source_sets_hdf5_initial_state,
    },
    "issues": issues,
    "risk_assessment": {
        "task": "HDF5-to-rollout alignment audit",
        "command": "scripts/97_audit_hdf5_rollout_alignment.ps1",
        "source": "local ignored reports plus one local LIBERO HDF5 demonstration",
        "expected_size_gb": 0,
        "expected_runtime_minutes": 1,
        "expected_ram_gb": 1,
        "expected_vram_gb": 0,
        "token_login_license_payment_needed": False,
        "simulator_will_run": False,
        "rollout_will_run": False,
        "training_will_run": False,
        "openvla_oft_will_run": False,
        "paper_claim_will_be_made": False,
        "decision": decision,
        "reason": reason,
    },
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_hdf5_initial_state_replay_plan": ready_replay_plan,
    "ready_for_rollout_scaling": False,
    "ready_for_paper_claim": False,
    "recommended_next_step": next_step,
}
write_outputs(report)
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
