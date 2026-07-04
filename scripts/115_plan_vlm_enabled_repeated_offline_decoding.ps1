param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$VlmLoadSmokeReportPath = "reports\vlm_enabled_load_smoke_report.json",
    [string]$PreviousRepeatedReportPath = "reports\repeated_offline_demo_action_decoding_report.json",
    [string]$PreviousRepeatedPlanPath = "reports\repeated_offline_demo_action_decoding_plan_report.json",
    [int]$MaxTimesteps = 3,
    [string]$JsonReportPath = "reports\vlm_enabled_repeated_offline_decoding_plan_report.json",
    [string]$MarkdownReportPath = "reports\vlm_enabled_repeated_offline_decoding_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "VLM-enabled repeated offline decoding recheck plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not load models, infer, train, rollout, download, install, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    "ALLOW_VLM_ENABLED_LOAD_SMOKE",
    "ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING",
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

$env:TCA_MAP_VLM_RECHECK_LOAD = Resolve-RepoPath -Path $VlmLoadSmokeReportPath
$env:TCA_MAP_VLM_RECHECK_PREVIOUS = Resolve-RepoPath -Path $PreviousRepeatedReportPath
$env:TCA_MAP_VLM_RECHECK_PLAN = Resolve-RepoPath -Path $PreviousRepeatedPlanPath
$env:TCA_MAP_VLM_RECHECK_MAX_TIMESTEPS = [string]$MaxTimesteps
$env:TCA_MAP_VLM_RECHECK_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_VLM_RECHECK_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_VLM_RECHECK_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
import shutil
from pathlib import Path

LOAD_PATH = Path(os.environ["TCA_MAP_VLM_RECHECK_LOAD"])
PREVIOUS_PATH = Path(os.environ["TCA_MAP_VLM_RECHECK_PREVIOUS"])
PLAN_PATH = Path(os.environ["TCA_MAP_VLM_RECHECK_PLAN"])
MAX_TIMESTEPS = max(1, min(3, int(os.environ.get("TCA_MAP_VLM_RECHECK_MAX_TIMESTEPS", "3"))))
JSON_OUT = Path(os.environ["TCA_MAP_VLM_RECHECK_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_VLM_RECHECK_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_VLM_RECHECK_GATES", "").split(";") if item]


def read_json(path: Path):
    if not path.exists():
        return None, f"Missing JSON file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not parse {path}: {exc}"


stop_reasons: list[str] = []
if SET_GATES:
    stop_reasons.append("Planning-only VLM-enabled repeated offline gate refuses execution gates: " + ", ".join(SET_GATES))

load, load_error = read_json(LOAD_PATH)
previous, previous_error = read_json(PREVIOUS_PATH)
plan, plan_error = read_json(PLAN_PATH)
for error in (load_error, previous_error, plan_error):
    if error:
        stop_reasons.append(error)
load = load or {}
previous = previous or {}
plan = plan or {}

if not load.get("vlm_enabled_load_smoke_passed"):
    stop_reasons.append("VLM-enabled load smoke report did not pass.")
if (load.get("load") or {}).get("load_vlm_weights") is not True:
    stop_reasons.append("VLM-enabled load report does not show load_vlm_weights=true.")
if (load.get("policy") or {}).get("model_inference_performed"):
    stop_reasons.append("VLM-enabled load smoke unexpectedly performed inference.")
if not previous.get("repeated_offline_demo_action_decoding_passed"):
    stop_reasons.append("Previous repeated offline decoding report did not pass.")

previous_metrics = previous.get("metrics") or {}
previous_signal = previous_metrics.get("offline_alignment_signal")
if previous_signal != "weak":
    stop_reasons.append(f"Previous repeated offline alignment signal is not weak; recheck is not currently prioritized: {previous_signal}")
if previous_metrics.get("load_vlm_weights") is not False:
    stop_reasons.append("Previous repeated offline report was expected to use load_vlm_weights=false.")

plan_ready = bool(plan.get("ready_for_bounded_repeated_offline_demo_action_decoding_runner"))
if not plan_ready:
    stop_reasons.append("Previous repeated offline decoding plan is not runner-ready.")

planned = ((plan.get("planned_sample") or {}).get("hdf5") or {})
timesteps = [int(item) for item in planned.get("selected_timesteps", [])[:MAX_TIMESTEPS]]
hdf5_path = Path((plan.get("inputs") or {}).get("hdf5_path") or "")
if not timesteps:
    stop_reasons.append("No selected timesteps found in previous repeated offline plan.")
if not hdf5_path.exists():
    stop_reasons.append(f"Selected HDF5 file is missing: {hdf5_path}")

disk = shutil.disk_usage(Path.cwd().anchor or str(Path.cwd()))
free_gb = round(disk.free / (1024**3), 3)
passed = not stop_reasons

report = {
    "evidence_label": "vlm_enabled_repeated_offline_decoding_plan",
    "vlm_enabled_repeated_offline_decoding_plan_passed": passed,
    "decision": "proceed" if passed else "stop",
    "ready_for_bounded_vlm_enabled_repeated_offline_decoding_runner": passed,
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
    "inputs": {
        "vlm_enabled_load_smoke_report": str(LOAD_PATH),
        "previous_repeated_report": str(PREVIOUS_PATH),
        "previous_repeated_plan": str(PLAN_PATH),
        "hdf5_path": str(hdf5_path),
        "selected_timesteps": timesteps,
    },
    "baseline_to_compare": {
        "previous_load_vlm_weights": previous_metrics.get("load_vlm_weights"),
        "previous_offline_alignment_signal": previous_signal,
        "previous_mean_action_l1_to_expert": previous_metrics.get("mean_action_l1_to_expert"),
        "previous_mean_action_mse_to_expert": previous_metrics.get("mean_action_mse_to_expert"),
        "previous_clipped_values_total": previous_metrics.get("clipped_values_total"),
    },
    "vlm_load_summary": {
        "load_vlm_weights": (load.get("load") or {}).get("load_vlm_weights"),
        "device": (load.get("load") or {}).get("device"),
        "parameter_count": (load.get("load") or {}).get("parameter_count"),
        "cuda_max_allocated_mb": (load.get("load") or {}).get("cuda_max_allocated_mb"),
        "load_elapsed_sec": (load.get("load") or {}).get("load_elapsed_sec"),
    },
    "risk_assessment": {
        "task": "Bounded VLM-enabled repeated offline demonstration action decoding recheck",
        "future_command": "scripts\\116_bounded_vlm_enabled_repeated_offline_decoding.ps1",
        "future_runner_gates": ["ALLOW_HEAVY_IMPORT=1", "ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING=1"],
        "source_path": str(hdf5_path),
        "target_output_paths": [
            "reports\\vlm_enabled_repeated_offline_decoding_report.json",
            "reports\\vlm_enabled_repeated_offline_decoding_report.md",
        ],
        "expected_runtime_minutes": 20,
        "expected_new_disk_gb": 0,
        "expected_ram_gb": 18,
        "expected_vram_gb": 0,
        "current_free_disk_gb": free_gb,
        "max_policy_inference_calls": MAX_TIMESTEPS,
        "planned_policy_inference_calls": len(timesteps) if passed else 0,
        "simulator_will_run": False,
        "rollout_will_run": False,
        "training_will_run": False,
        "token_login_license_payment_required": False,
        "stop_condition": "Stop if VLM-enabled inference needs GPU-only execution, exceeds three policy calls, requires rollout/simulator/training/download/OpenVLA-OFT/tokens, or exceeds runtime/RAM budget.",
        "fallback_plan": "If VLM-enabled repeated decoding is unsafe or still weak, keep rollout scaling blocked and pivot to action normalization/provenance analysis.",
        "decision": "proceed" if passed else "stop",
        "reason": "VLM-enabled load-only passed and prior load_vlm_weights=false repeated offline alignment was weak, so a bounded offline recheck is the next informative diagnostic." if passed else "; ".join(stop_reasons),
    },
    "stop_reasons": stop_reasons,
    "recommended_next_step": (
        "Implement the separately gated VLM-enabled repeated offline decoding runner. It may load local SmolVLA with VLM weights on CPU and decode at most three HDF5 timesteps, but must not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."
        if passed
        else "Resolve the listed blockers before implementing the VLM-enabled repeated offline decoding runner."
    ),
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# VLM-Enabled Repeated Offline Decoding Plan Report",
    "",
    f"- decision: {report['decision']}",
    f"- plan passed: {report['vlm_enabled_repeated_offline_decoding_plan_passed']}",
    f"- runner ready: {report['ready_for_bounded_vlm_enabled_repeated_offline_decoding_runner']}",
    f"- previous alignment signal: {previous_signal}",
    f"- selected timesteps: {timesteps}",
    f"- future gates: {', '.join(report['risk_assessment']['future_runner_gates'])}",
    "",
    "This planner is diagnostic-only. It did not load models, infer, train, rollout, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.",
    "",
    "## Recommended Next Step",
    "",
    report["recommended_next_step"],
]
MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
