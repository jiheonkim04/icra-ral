param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmolVLACkpt = "",
    [string]$VlmActionAuditPath = "reports\vlm_loading_policy_action_normalization_audit_report.json",
    [string]$OfflineDecodingReportPath = "reports\offline_demo_action_decoding_report.json",
    [int]$MaxTimesteps = 3,
    [string]$JsonReportPath = "reports\repeated_offline_demo_action_decoding_plan_report.json",
    [string]$MarkdownReportPath = "reports\repeated_offline_demo_action_decoding_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Repeated offline demonstration action-decoding plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It reads local reports and HDF5 metadata only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
    "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
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

if ([string]::IsNullOrWhiteSpace($SmolVLACkpt)) { $SmolVLACkpt = $env:SMOLVLA_CKPT }
if ([string]::IsNullOrWhiteSpace($SmolVLACkpt)) { $SmolVLACkpt = "C:\assets\checkpoints\smolvla" }

$env:TCA_MAP_REPEAT_PLAN_SMOLVLA = $SmolVLACkpt
$env:TCA_MAP_REPEAT_PLAN_AUDIT = Resolve-RepoPath -Path $VlmActionAuditPath
$env:TCA_MAP_REPEAT_PLAN_OFFLINE = Resolve-RepoPath -Path $OfflineDecodingReportPath
$env:TCA_MAP_REPEAT_PLAN_MAX_TIMESTEPS = [string]$MaxTimesteps
$env:TCA_MAP_REPEAT_PLAN_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_REPEAT_PLAN_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_REPEAT_PLAN_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
import shutil
from pathlib import Path

import h5py

SMOLVLA = Path(os.environ["TCA_MAP_REPEAT_PLAN_SMOLVLA"])
AUDIT_PATH = Path(os.environ["TCA_MAP_REPEAT_PLAN_AUDIT"])
OFFLINE_PATH = Path(os.environ["TCA_MAP_REPEAT_PLAN_OFFLINE"])
MAX_TIMESTEPS = max(1, min(3, int(os.environ.get("TCA_MAP_REPEAT_PLAN_MAX_TIMESTEPS", "3"))))
JSON_OUT = Path(os.environ["TCA_MAP_REPEAT_PLAN_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_REPEAT_PLAN_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_REPEAT_PLAN_GATES", "").split(";") if item]


def read_json(path: Path):
    if not path.exists():
        return None, f"Missing JSON file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parse issue is useful.
        return None, f"Could not parse {path}: {exc}"


def first_existing(path: Path, names: list[str]) -> list[str]:
    return [name for name in names if (path / name).exists()]


def find_demo_group(handle):
    candidates = []
    if "data" in handle:
        root = handle["data"]
        for name, value in root.items():
            if hasattr(value, "keys") and "actions" in value:
                candidates.append((name, value))
    for name, value in handle.items():
        if hasattr(value, "keys") and "actions" in value:
            candidates.append((name, value))
    return candidates[0] if candidates else (None, None)


def sample_hdf5(path: Path):
    with h5py.File(path, "r") as handle:
        demo_name, demo = find_demo_group(handle)
        if demo is None:
            raise RuntimeError("No demo group with an actions dataset was found.")
        actions = demo["actions"]
        action_shape = list(actions.shape)
        timestep_count = int(action_shape[0]) if action_shape else 0
        if timestep_count <= 0:
            raise RuntimeError("Actions dataset has no timesteps.")
        selected = sorted(set([0, timestep_count // 2, timestep_count - 1]))[:MAX_TIMESTEPS]
        if len(selected) > MAX_TIMESTEPS:
            selected = selected[:MAX_TIMESTEPS]
        obs_keys = []
        if "obs" in demo:
            obs_keys = sorted(demo["obs"].keys())
        return {
            "demo_name": demo_name,
            "action_shape": action_shape,
            "timestep_count": timestep_count,
            "selected_timesteps": selected,
            "obs_keys": obs_keys,
            "has_ee_states": bool("obs" in demo and "ee_states" in demo["obs"]),
            "has_agentview_rgb": bool("obs" in demo and ("agentview_rgb" in demo["obs"] or "agentview_image" in demo["obs"])),
            "has_eye_in_hand_rgb": bool("obs" in demo and ("eye_in_hand_rgb" in demo["obs"] or "robot0_eye_in_hand_image" in demo["obs"])),
            "has_states": bool("states" in demo),
            "has_init_state_attr": bool("init_state" in demo.attrs),
        }


stop_reasons: list[str] = []
if SET_GATES:
    stop_reasons.append("Planning-only repeated offline decoding gate refuses execution gates: " + ", ".join(SET_GATES))

audit, audit_error = read_json(AUDIT_PATH)
offline, offline_error = read_json(OFFLINE_PATH)
for error in (audit_error, offline_error):
    if error:
        stop_reasons.append(error)
audit = audit or {}
offline = offline or {}

if not audit.get("vlm_loading_policy_action_normalization_audit_passed"):
    stop_reasons.append("VLM/action-normalization audit did not pass.")
if not audit.get("ready_for_repeated_offline_decoding_plan"):
    stop_reasons.append("VLM/action-normalization audit did not authorize repeated offline decoding planning.")

hdf5_value = (
    (((audit.get("offline_alignment_summary") or {}).get("sample") or {}).get("hdf5_path"))
    or ((offline.get("sample") or {}).get("hdf5_path"))
)
hdf5_path = Path(hdf5_value) if hdf5_value else None
if not hdf5_path or not hdf5_path.exists():
    stop_reasons.append(f"Selected HDF5 path is missing: {hdf5_path}")

required_checkpoint_files = first_existing(
    SMOLVLA,
    [
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "model.safetensors",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ],
)
if "config.json" not in required_checkpoint_files:
    stop_reasons.append(f"SmolVLA config.json is missing under {SMOLVLA}")
if "policy_preprocessor.json" not in required_checkpoint_files:
    stop_reasons.append(f"SmolVLA policy_preprocessor.json is missing under {SMOLVLA}")
if "policy_postprocessor.json" not in required_checkpoint_files:
    stop_reasons.append(f"SmolVLA policy_postprocessor.json is missing under {SMOLVLA}")
if "model.safetensors" not in required_checkpoint_files:
    stop_reasons.append(f"SmolVLA model.safetensors is missing under {SMOLVLA}")

hdf5_summary = None
if not stop_reasons:
    try:
        hdf5_summary = sample_hdf5(hdf5_path)
    except Exception as exc:  # noqa: BLE001 - local HDF5 shape issue.
        stop_reasons.append(f"Could not inspect selected HDF5 file: {exc}")

disk = shutil.disk_usage(Path.cwd().anchor or str(Path.cwd()))
free_gb = round(disk.free / (1024**3), 3)
passed = not stop_reasons
planned_inference_calls = len((hdf5_summary or {}).get("selected_timesteps", [])) if passed else 0

report = {
    "evidence_label": "repeated_offline_demo_action_decoding_plan",
    "repeated_offline_demo_action_decoding_plan_passed": passed,
    "decision": "proceed" if passed else "stop",
    "ready_for_bounded_repeated_offline_demo_action_decoding_runner": passed,
    "ready_for_rollout_scaling": False,
    "ready_for_benchmark_claim": False,
    "ready_for_paper_claim": False,
    "policy": {
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
    },
    "claims": {
        "standard_success_claimed": False,
        "benchmark_success_claimed": False,
        "counterfactual_robustness_claimed": False,
        "sota_claimed": False,
        "paper_grade_claim_made": False,
    },
    "risk_assessment": {
        "task": "tiny repeated offline demonstration action decoding runner",
        "future_command": "powershell -ExecutionPolicy Bypass -File scripts\\110_bounded_repeated_offline_demo_action_decoding.ps1",
        "future_runner_gate": "ALLOW_REPEATED_OFFLINE_DEMO_DECODING=1",
        "source_path": str(hdf5_path) if hdf5_path else None,
        "checkpoint_path": str(SMOLVLA),
        "target_output_paths": [
            "reports\\repeated_offline_demo_action_decoding_report.json",
            "reports\\repeated_offline_demo_action_decoding_report.md",
        ],
        "expected_runtime_minutes": 15,
        "expected_new_disk_gb": 0,
        "expected_ram_gb": 8,
        "expected_vram_gb": 0,
        "current_free_disk_gb": free_gb,
        "max_hdf5_timesteps_to_read": MAX_TIMESTEPS,
        "max_policy_inference_calls": MAX_TIMESTEPS,
        "planned_policy_inference_calls": planned_inference_calls,
        "model_inference_in_this_planner": False,
        "future_runner_model_inference": True,
        "simulator_will_run": False,
        "rollout_will_run": False,
        "training_will_run": False,
        "token_login_license_payment_required": False,
        "stop_condition": "Stop if the future runner needs downloads, tokens, GPU-only execution, simulator rollout, OpenVLA-OFT, more than three policy calls, or runtime beyond budget.",
        "fallback_plan": "If repeated local inference is not safe, keep the audit result and prepare a cloud or VLM-loading handoff plan.",
        "decision": "proceed" if passed else "stop",
        "reason": "The audit is green for repeated offline planning, checkpoint files are present, and the selected HDF5 file exposes enough timesteps." if passed else "; ".join(stop_reasons),
    },
    "inputs": {
        "audit_report": str(AUDIT_PATH),
        "offline_decoding_report": str(OFFLINE_PATH),
        "smolvla_ckpt": str(SMOLVLA),
        "hdf5_path": str(hdf5_path) if hdf5_path else None,
        "checkpoint_files_present": required_checkpoint_files,
    },
    "planned_sample": {
        "hdf5": hdf5_summary,
        "selected_task_text": (((audit.get("offline_alignment_summary") or {}).get("sample") or {}).get("task")),
        "load_vlm_weights": (audit.get("checkpoint_summary") or {}).get("observed_load_vlm_weights"),
        "config_load_vlm_weights": (audit.get("checkpoint_summary") or {}).get("config_load_vlm_weights"),
        "action_normalization": ((audit.get("checkpoint_summary") or {}).get("config_normalization_mapping") or {}).get("ACTION"),
        "adapter_strategy": (((audit.get("offline_alignment_summary") or {}).get("action_adapter_metadata") or {}).get("strategy")),
        "log_required_fields": [
            "load_vlm_weights",
            "policy_action_preview",
            "adapted_action_preview",
            "expert_action_preview",
            "action_l1_to_expert",
            "action_mse_to_expert",
            "policy6_l1_to_expert_first6",
            "clipped_values",
            "gripper_strategy",
            "image_sources",
        ],
    },
    "stop_reasons": stop_reasons,
    "recommended_next_step": (
        "Implement the separately gated repeated offline decoding runner. It may load local SmolVLA on CPU and run at most three HDF5 timestep action decodes, but it must not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, or make paper claims."
        if passed
        else "Resolve missing planning prerequisites before implementing the repeated offline decoding runner."
    ),
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Repeated Offline Demonstration Action Decoding Plan Report",
    "",
    f"- decision: {report['decision']}",
    f"- plan passed: {report['repeated_offline_demo_action_decoding_plan_passed']}",
    f"- runner ready: {report['ready_for_bounded_repeated_offline_demo_action_decoding_runner']}",
    f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
    f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
    f"- HDF5 path: {report['inputs']['hdf5_path']}",
    f"- selected timesteps: {report['planned_sample']['hdf5']['selected_timesteps'] if report['planned_sample']['hdf5'] else None}",
    f"- future max policy calls: {report['risk_assessment']['max_policy_inference_calls']}",
    "",
    "## Risk Assessment",
    "",
    f"- expected runtime minutes: {report['risk_assessment']['expected_runtime_minutes']}",
    f"- expected new disk GB: {report['risk_assessment']['expected_new_disk_gb']}",
    f"- expected RAM GB: {report['risk_assessment']['expected_ram_gb']}",
    f"- expected VRAM GB: {report['risk_assessment']['expected_vram_gb']}",
    f"- current free disk GB: {report['risk_assessment']['current_free_disk_gb']}",
    f"- simulator will run: {report['risk_assessment']['simulator_will_run']}",
    f"- rollout will run: {report['risk_assessment']['rollout_will_run']}",
    "",
    "## Recommended Next Step",
    "",
    report["recommended_next_step"],
    "",
    "This plan is diagnostic-only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
]
MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
