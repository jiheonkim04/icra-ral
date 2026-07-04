param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmolVLACkpt = "",
    [string]$LiberoDataRoot = "",
    [string]$AlignmentAuditPath = "reports\smolvla_libero_checkpoint_task_alignment_audit_report.json",
    [string]$Hdf5AuditPath = "reports\libero_hdf5_interface_audit_report.json",
    [string]$OfflineAdapterReportPath = "reports\offline_adapter_reproduction_check_report.json",
    [string]$JsonReportPath = "reports\offline_demo_conditioned_action_decoding_plan_report.json",
    [string]$MarkdownReportPath = "reports\offline_demo_conditioned_action_decoding_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Offline demonstration-conditioned action decoding plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT"
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
if ([string]::IsNullOrWhiteSpace($LiberoDataRoot)) { $LiberoDataRoot = $env:LIBERO_DATA_ROOT }
if ([string]::IsNullOrWhiteSpace($LiberoDataRoot)) { $LiberoDataRoot = "C:\assets\data\libero" }

$env:TCA_MAP_OFFLINE_DECODE_SMOLVLA = $SmolVLACkpt
$env:TCA_MAP_OFFLINE_DECODE_LIBERO_DATA = $LiberoDataRoot
$env:TCA_MAP_OFFLINE_DECODE_ALIGNMENT = Resolve-RepoPath -Path $AlignmentAuditPath
$env:TCA_MAP_OFFLINE_DECODE_HDF5 = Resolve-RepoPath -Path $Hdf5AuditPath
$env:TCA_MAP_OFFLINE_DECODE_ADAPTER = Resolve-RepoPath -Path $OfflineAdapterReportPath
$env:TCA_MAP_OFFLINE_DECODE_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_OFFLINE_DECODE_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_OFFLINE_DECODE_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
import shutil
from pathlib import Path

SMOLVLA = Path(os.environ["TCA_MAP_OFFLINE_DECODE_SMOLVLA"])
LIBERO_DATA = Path(os.environ["TCA_MAP_OFFLINE_DECODE_LIBERO_DATA"])
ALIGNMENT = Path(os.environ["TCA_MAP_OFFLINE_DECODE_ALIGNMENT"])
HDF5_AUDIT = Path(os.environ["TCA_MAP_OFFLINE_DECODE_HDF5"])
OFFLINE_ADAPTER = Path(os.environ["TCA_MAP_OFFLINE_DECODE_ADAPTER"])
JSON_OUT = Path(os.environ["TCA_MAP_OFFLINE_DECODE_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_OFFLINE_DECODE_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_OFFLINE_DECODE_GATES", "").split(";") if item]


def read_json(path: Path):
    if not path.exists():
        return None, f"Missing JSON file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parsing issue.
        return None, f"Could not parse {path}: {exc}"


def first_existing(path: Path, patterns: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for item in sorted(path.glob(pattern)):
            if item.name not in seen:
                found.append(item.name)
                seen.add(item.name)
    return found


stop_reasons: list[str] = []
if SET_GATES:
    stop_reasons.append("Planning-only offline decoding gate refuses execution gates: " + ", ".join(SET_GATES))

alignment, alignment_error = read_json(ALIGNMENT)
hdf5_audit, hdf5_error = read_json(HDF5_AUDIT)
offline_adapter, offline_error = read_json(OFFLINE_ADAPTER)
for error in (alignment_error, hdf5_error, offline_error):
    if error:
        stop_reasons.append(error)

alignment = alignment or {}
hdf5_audit = hdf5_audit or {}
offline_adapter = offline_adapter or {}

hdf5_path_value = (
    (alignment.get("task_summary") or {}).get("hdf5_path")
    or (offline_adapter.get("paths") or {}).get("hdf5_path")
    or (hdf5_audit.get("paths") or {}).get("hdf5_path")
)
hdf5_path = Path(hdf5_path_value) if hdf5_path_value else None
config_files = first_existing(SMOLVLA, ["config.json"])
preprocessor_files = first_existing(SMOLVLA, ["policy_preprocessor.json"])
postprocessor_files = first_existing(SMOLVLA, ["policy_postprocessor.json"])
weight_files = first_existing(SMOLVLA, ["model.safetensors", "pytorch_model.bin", "*.safetensors", "*.bin"])

if not alignment.get("smolvla_libero_checkpoint_task_alignment_audit_passed"):
    stop_reasons.append("Checkpoint-task alignment audit did not pass.")
if not alignment.get("ready_for_offline_demonstration_conditioned_action_decoding_plan"):
    stop_reasons.append("Checkpoint-task alignment audit did not authorize offline decoding planning.")
if not hdf5_path or not hdf5_path.exists():
    stop_reasons.append(f"Selected HDF5 path is missing: {hdf5_path}")
if not config_files:
    stop_reasons.append(f"SmolVLA config.json is missing under {SMOLVLA}")
if not preprocessor_files:
    stop_reasons.append(f"SmolVLA policy_preprocessor.json is missing under {SMOLVLA}")
if not postprocessor_files:
    stop_reasons.append(f"SmolVLA policy_postprocessor.json is missing under {SMOLVLA}")
if not weight_files:
    stop_reasons.append(f"SmolVLA weight files are missing under {SMOLVLA}")

disk_root = Path.cwd().anchor or str(Path.cwd())
disk = shutil.disk_usage(disk_root)
free_gb = round(disk.free / (1024**3), 3)

selected_task = (alignment.get("task_summary") or {}).get("selected_task_name")
selected_language = (alignment.get("task_summary") or {}).get("selected_bddl_language")
policy_action_shape = (alignment.get("checkpoint_summary") or {}).get("action_shape")
hdf5_action_dim = (alignment.get("evidence_summary") or {}).get("hdf5_action_dim")
best_strategy = (alignment.get("evidence_summary") or {}).get("best_gripper_strategy_for_first_demo_action") or (
    offline_adapter.get("reproduction") or {}
).get("best_action_adapter_strategy_for_first_demo_action")

passed = not stop_reasons
report = {
    "evidence_label": "offline_demo_conditioned_action_decoding_plan",
    "offline_demo_conditioned_action_decoding_plan_passed": passed,
    "decision": "proceed" if passed else "stop",
    "ready_for_bounded_offline_demo_action_decoding_runner": passed,
    "ready_for_rollout_scaling": False,
    "ready_for_paper_claim": False,
    "risk_assessment": {
        "task": "one-sample offline demonstration-conditioned SmolVLA action decoding runner",
        "future_command": "powershell -ExecutionPolicy Bypass -File scripts\\106_bounded_offline_demo_action_decoding.ps1",
        "source_path": str(hdf5_path) if hdf5_path else None,
        "checkpoint_path": str(SMOLVLA),
        "target_output_paths": [
            "reports\\offline_demo_action_decoding_report.json",
            "reports\\offline_demo_action_decoding_report.md",
        ],
        "expected_runtime_minutes": 10,
        "expected_new_disk_gb": 0,
        "expected_ram_gb": 8,
        "expected_vram_gb": 0,
        "current_free_disk_gb": free_gb,
        "token_login_license_payment_required": False,
        "simulator_will_run": False,
        "rollout_will_run": False,
        "training_will_run": False,
        "model_inference_in_this_planner": False,
        "future_runner_model_inference": True,
        "future_runner_gate": "ALLOW_OFFLINE_DEMO_ACTION_DECODING=1",
        "stop_condition": "Stop if model loading would require downloads, tokens, simulator rollout, GPU-only execution, OpenVLA-OFT, or runtime beyond budget.",
        "fallback_plan": "If bounded model inference is not safe, keep analysis report-only and move to cloud handoff or checkpoint-provenance investigation.",
        "decision": "proceed" if passed else "stop",
        "reason": "All local file/report prerequisites are present for a separately gated one-sample offline decoding runner." if passed else "; ".join(stop_reasons),
    },
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
    "inputs": {
        "alignment_audit": str(ALIGNMENT),
        "hdf5_audit": str(HDF5_AUDIT),
        "offline_adapter_report": str(OFFLINE_ADAPTER),
        "smolvla_ckpt": str(SMOLVLA),
        "libero_data_root": str(LIBERO_DATA),
        "hdf5_path": str(hdf5_path) if hdf5_path else None,
        "checkpoint_files": {
            "config": config_files,
            "preprocessor": preprocessor_files,
            "postprocessor": postprocessor_files,
            "weights": weight_files,
        },
    },
    "planned_sample": {
        "selected_task_name": selected_task,
        "selected_language": selected_language,
        "policy_action_shape": policy_action_shape,
        "hdf5_action_dim": hdf5_action_dim,
        "expert_adapter_strategy": best_strategy,
        "max_hdf5_timesteps_to_read": 1,
        "max_policy_inference_calls_in_future_runner": 1,
    },
    "stop_reasons": stop_reasons,
    "recommended_next_step": (
        "Implement a separately gated one-sample offline action-decoding runner. It may set ALLOW_OFFLINE_DEMO_ACTION_DECODING=1 only inside the task, load local SmolVLA on CPU, read one HDF5 observation/action pair, run one select_action call, compare to expert action, and write diagnostic metrics. It must not create a simulator environment, rollout, train, download, use OpenVLA-OFT, or make paper claims."
        if passed
        else "Resolve missing planning prerequisites before implementing the offline decoding runner."
    ),
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Offline Demonstration-Conditioned Action Decoding Plan Report",
    "",
    f"- decision: {report['decision']}",
    f"- plan passed: {report['offline_demo_conditioned_action_decoding_plan_passed']}",
    f"- runner ready: {report['ready_for_bounded_offline_demo_action_decoding_runner']}",
    f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
    f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
    f"- selected task: {report['planned_sample']['selected_task_name']}",
    f"- HDF5 path: {report['inputs']['hdf5_path']}",
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
    "This plan is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
]
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
