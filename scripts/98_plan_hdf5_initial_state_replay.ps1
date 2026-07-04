param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$Hdf5AlignmentAuditReportPath = "reports\hdf5_rollout_alignment_audit_report.json",
    [string]$OfflineReproductionReportPath = "reports\offline_adapter_reproduction_check_report.json",
    [string]$LiberoEnvWrapperPath = "C:\assets\repos\LIBERO\libero\libero\envs\env_wrapper.py",
    [string]$LiberoReadmePath = "C:\assets\repos\LIBERO\README.md",
    [string]$RoboSuitePlaybackPath = "C:\assets\repos\robosuite\robosuite\scripts\playback_demonstrations_from_hdf5.py",
    [string]$JsonReportPath = "reports\hdf5_initial_state_replay_plan_report.json",
    [string]$MarkdownReportPath = "reports\hdf5_initial_state_replay_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "HDF5 initial-state replay planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This planner reads local reports, one local HDF5 demonstration, and local LIBERO/RoboSuite source files only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$env:TCA_MAP_HDF5_REPLAY_ALIGN = Resolve-RepoPath -Path $Hdf5AlignmentAuditReportPath
$env:TCA_MAP_HDF5_REPLAY_REPRO = Resolve-RepoPath -Path $OfflineReproductionReportPath
$env:TCA_MAP_HDF5_REPLAY_LIBERO_ENV = $LiberoEnvWrapperPath
$env:TCA_MAP_HDF5_REPLAY_LIBERO_README = $LiberoReadmePath
$env:TCA_MAP_HDF5_REPLAY_ROBOSUITE_PLAYBACK = $RoboSuitePlaybackPath
$env:TCA_MAP_HDF5_REPLAY_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_HDF5_REPLAY_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_HDF5_REPLAY_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

ALIGN = Path(os.environ["TCA_MAP_HDF5_REPLAY_ALIGN"])
REPRO = Path(os.environ["TCA_MAP_HDF5_REPLAY_REPRO"])
LIBERO_ENV = Path(os.environ["TCA_MAP_HDF5_REPLAY_LIBERO_ENV"])
LIBERO_README = Path(os.environ["TCA_MAP_HDF5_REPLAY_LIBERO_README"])
ROBOSUITE_PLAYBACK = Path(os.environ["TCA_MAP_HDF5_REPLAY_ROBOSUITE_PLAYBACK"])
JSON_OUT = Path(os.environ["TCA_MAP_HDF5_REPLAY_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_HDF5_REPLAY_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_HDF5_REPLAY_GATES", "").split(";") if item]


def load_json(path: Path, *, required: bool = True):
    if not path.exists():
        return None, f"Missing input report: {path}" if required else None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read {path}: {exc}"


def read_text(path: Path):
    if not path.exists():
        return "", f"Missing source file: {path}"
    return path.read_text(encoding="utf-8", errors="replace"), None


def write_outputs(report: dict) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HDF5 Initial-State Replay Plan Report",
        "",
        f"- decision: {report['decision']}",
        f"- plan passed: {report['hdf5_initial_state_replay_plan_passed']}",
        f"- ready for bounded replay runner: {report['ready_for_bounded_hdf5_replay_runner']}",
        f"- HDF5 init state present: {report['hdf5_inputs']['init_state_present']}",
        f"- HDF5 model file attr present: {report['hdf5_inputs']['model_file_attr_present']}",
        f"- LIBERO set_init_state support: {report['source_support']['libero_env_set_init_state']}",
        f"- RoboSuite playback support: {report['source_support']['robosuite_playback_sets_state']}",
        f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
        f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
        "",
        report["recommended_next_step"],
        "",
        "This is a planning report only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


stop_reasons: list[str] = []
warnings: list[str] = []
if SET_GATES:
    stop_reasons.append("Refusing HDF5 initial-state replay planning while execution gates are set: " + ", ".join(SET_GATES))

try:
    import h5py
except Exception as exc:  # noqa: BLE001
    h5py = None
    stop_reasons.append(f"h5py is required for report-only HDF5 inspection: {exc}")

align, align_error = load_json(ALIGN, required=True)
repro, repro_error = load_json(REPRO, required=True)
for error in (align_error, repro_error):
    if error:
        stop_reasons.append(error)
align = align or {}
repro = repro or {}

libero_env_text, libero_env_error = read_text(LIBERO_ENV)
libero_readme_text, libero_readme_error = read_text(LIBERO_README)
robosuite_playback_text, robosuite_playback_error = read_text(ROBOSUITE_PLAYBACK)
for error in (libero_env_error, libero_readme_error, robosuite_playback_error):
    if error:
        stop_reasons.append(error)

if align.get("ready_for_hdf5_initial_state_replay_plan") is not True:
    stop_reasons.append("HDF5 rollout alignment audit did not authorize an initial-state replay plan.")

hdf5_path = Path((repro.get("paths") or {}).get("hdf5_path") or "")
if not str(hdf5_path) or not hdf5_path.exists():
    stop_reasons.append(f"Missing HDF5 demonstration file from offline reproduction report: {hdf5_path}")

hdf5_inputs = {
    "hdf5_path": str(hdf5_path) if str(hdf5_path) else None,
    "demo_name": None,
    "init_state_present": False,
    "init_state_shape": None,
    "states_present": False,
    "states_shape": None,
    "actions_shape": None,
    "first_action": None,
    "first_gripper_action": None,
    "model_file_attr_present": False,
}
if not stop_reasons and h5py is not None:
    with h5py.File(hdf5_path, "r") as handle:
        demo_name = sorted(handle["data"].keys())[0]
        demo = handle["data"][demo_name]
        init_state = demo.attrs.get("init_state")
        states = demo.get("states")
        actions = demo.get("actions")
        first_action = actions[0].tolist() if actions is not None and actions.shape[0] > 0 else None
        hdf5_inputs.update(
            {
                "demo_name": demo_name,
                "init_state_present": init_state is not None,
                "init_state_shape": list(getattr(init_state, "shape", [])) if init_state is not None else None,
                "states_present": states is not None,
                "states_shape": list(states.shape) if states is not None else None,
                "actions_shape": list(actions.shape) if actions is not None else None,
                "first_action": [float(x) for x in first_action] if first_action is not None else None,
                "first_gripper_action": float(first_action[-1]) if first_action is not None else None,
                "model_file_attr_present": "model_file" in demo.attrs,
            }
        )

if not hdf5_inputs["init_state_present"]:
    stop_reasons.append("HDF5 demonstration does not expose an init_state attribute.")
if not hdf5_inputs["states_present"]:
    stop_reasons.append("HDF5 demonstration does not expose a states trajectory.")
if not hdf5_inputs["actions_shape"] or hdf5_inputs["actions_shape"][-1] != 7:
    stop_reasons.append("HDF5 demonstration actions are not the expected 7D LIBERO action format.")

source_support = {
    "libero_env_set_init_state": "def set_init_state" in libero_env_text and "regenerate_obs_from_state" in libero_env_text,
    "libero_readme_documents_set_init_state": "env.set_init_state" in libero_readme_text,
    "robosuite_playback_resets_from_xml": "reset_from_xml_string" in robosuite_playback_text,
    "robosuite_playback_sets_state": "set_state_from_flattened" in robosuite_playback_text,
}
if not source_support["libero_env_set_init_state"]:
    stop_reasons.append("LIBERO env wrapper source does not expose set_init_state support.")
if not source_support["robosuite_playback_sets_state"]:
    stop_reasons.append("RoboSuite playback source does not expose flattened-state replay support.")

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
    passed = False
    ready_runner = False
    reason = "HDF5 initial-state replay prerequisites are missing or unsafe."
    next_step = "Fix missing HDF5/source prerequisites before planning any replay runner."
else:
    decision = "proceed"
    passed = True
    ready_runner = True
    reason = "Local HDF5 and source evidence support a separately gated bounded replay diagnostic plan."
    next_step = (
        "Implement a separately gated bounded HDF5 initial-state/first-action replay runner. It should use one task, one demo, "
        "set the HDF5 init_state or flattened state if supported, step only the first demonstration action or a tiny prefix, "
        "avoid learned-policy inference, and keep evidence labeled diagnostic only."
    )

report = {
    "evidence_label": "hdf5_initial_state_replay_plan",
    "hdf5_initial_state_replay_plan_passed": passed,
    "decision": decision,
    "reason": reason,
    "source_reports": {
        "hdf5_rollout_alignment_audit": str(ALIGN),
        "offline_adapter_reproduction": str(REPRO),
    },
    "source_files": {
        "libero_env_wrapper": str(LIBERO_ENV),
        "libero_readme": str(LIBERO_README),
        "robosuite_playback": str(ROBOSUITE_PLAYBACK),
    },
    "policy": policy,
    "claims": claims,
    "hdf5_inputs": hdf5_inputs,
    "source_support": source_support,
    "runner_plan": {
        "task_local_gate": "ALLOW_HDF5_REPLAY_DIAGNOSTIC=1",
        "max_tasks": 1,
        "max_demos": 1,
        "max_replay_steps_first_runner": 1,
        "max_replay_steps_later_runner": 5,
        "policy_inference_allowed": False,
        "training_allowed": False,
        "gpu_allowed": False,
        "openvla_oft_allowed": False,
        "evidence_label": "bounded_hdf5_initial_state_replay_diagnostic",
        "acceptance_checks": [
            "no downloads",
            "no installs",
            "no training",
            "no GPU jobs",
            "no model load or learned-policy inference",
            "no OpenVLA-OFT",
            "one HDF5 demo only",
            "one first-action replay step in the first runner",
            "diagnostic label only",
        ],
    },
    "risk_assessment": {
        "task": "HDF5 initial-state replay planning",
        "future_command": "future separately gated one-demo replay runner",
        "source": "local HDF5 demonstration plus local LIBERO/RoboSuite source code",
        "expected_size_gb": 0,
        "expected_runtime_minutes_if_run": 5,
        "expected_ram_gb_if_run": 4,
        "expected_vram_gb_if_run": 0,
        "token_login_license_payment_needed": False,
        "simulator_will_run_in_future_runner": True,
        "rollout_will_run_in_future_runner": False,
        "replay_diagnostic_will_run_in_future_runner": True,
        "learned_policy_inference_will_run": False,
        "training_will_run": False,
        "gpu_job_will_run": False,
        "openvla_oft_will_run": False,
        "paper_claim_will_be_made": False,
        "decision": decision,
        "reason": reason,
    },
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_bounded_hdf5_replay_runner": ready_runner,
    "ready_for_rollout_scaling": False,
    "ready_for_paper_claim": False,
    "recommended_next_step": next_step,
}
write_outputs(report)
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
