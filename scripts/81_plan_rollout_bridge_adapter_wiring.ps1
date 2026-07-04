param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PatchPlanReportPath = "reports\action_state_adapter_patch_plan_report.json",
    [string]$SingleSampleReportPath = "reports\smolvla_single_sample_interface_report.json",
    [string]$RolloutBridgeSourcePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$AdapterSourcePath = "tca_map\smolvla\interface_adapters.py",
    [string]$JsonReportPath = "reports\rollout_bridge_adapter_wiring_plan_report.json",
    [string]$MarkdownReportPath = "reports\rollout_bridge_adapter_wiring_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Rollout bridge adapter wiring planner"
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

$env:TCA_MAP_ROLLOUT_WIRING_PATCH_PLAN = Resolve-RepoPath -Path $PatchPlanReportPath
$env:TCA_MAP_ROLLOUT_WIRING_SINGLE_SAMPLE = Resolve-RepoPath -Path $SingleSampleReportPath
$env:TCA_MAP_ROLLOUT_WIRING_BRIDGE_SOURCE = Resolve-RepoPath -Path $RolloutBridgeSourcePath
$env:TCA_MAP_ROLLOUT_WIRING_ADAPTER_SOURCE = Resolve-RepoPath -Path $AdapterSourcePath
$env:TCA_MAP_ROLLOUT_WIRING_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_ROLLOUT_WIRING_MARKDOWN = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_ROLLOUT_WIRING_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

PATCH_PLAN = Path(os.environ["TCA_MAP_ROLLOUT_WIRING_PATCH_PLAN"])
SINGLE_SAMPLE = Path(os.environ["TCA_MAP_ROLLOUT_WIRING_SINGLE_SAMPLE"])
BRIDGE_SOURCE = Path(os.environ["TCA_MAP_ROLLOUT_WIRING_BRIDGE_SOURCE"])
ADAPTER_SOURCE = Path(os.environ["TCA_MAP_ROLLOUT_WIRING_ADAPTER_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_ROLLOUT_WIRING_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_ROLLOUT_WIRING_MARKDOWN"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_ROLLOUT_WIRING_GATES", "").split(";") if item]

def load_json(path):
    if not path.exists():
        return None, f"Missing input report: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:
        return None, f"Could not read {path}: {exc}"

def read_text(path):
    if not path.exists():
        return "", f"Missing source file: {path}"
    return path.read_text(encoding="utf-8", errors="replace"), None

def write_outputs(report):
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Rollout Bridge Adapter Wiring Plan Report",
        "",
        f"- decision: {report['decision']}",
        f"- planner passed: {report['rollout_bridge_adapter_wiring_plan_passed']}",
        f"- ready for rollout bridge adapter wiring: {report['ready_for_rollout_bridge_adapter_wiring']}",
        f"- ready for rollout execution: {report['ready_for_rollout_execution']}",
        f"- adapter metadata recorded: {report['inputs']['single_sample_adapter_metadata_recorded']}",
        f"- bridge needs action adapter: {report['source_audit']['bridge_needs_action_adapter']}",
        f"- bridge needs state adapter: {report['source_audit']['bridge_needs_state_adapter']}",
        f"- bridge needs image adapter: {report['source_audit']['bridge_needs_image_adapter']}",
        f"- paper-grade claim made: {report['claims']['paper_grade_claim_made']}",
        "",
        report["recommended_next_step"],
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

patch_plan, patch_error = load_json(PATCH_PLAN)
single_sample, sample_error = load_json(SINGLE_SAMPLE)
bridge_text, bridge_error = read_text(BRIDGE_SOURCE)
adapter_text, adapter_error = read_text(ADAPTER_SOURCE)

stop_reasons = []
warnings = []
if SET_GATES:
    stop_reasons.append("Refusing rollout bridge wiring planning while execution gates are set: " + ", ".join(SET_GATES))
for error in [patch_error, sample_error, bridge_error, adapter_error]:
    if error:
        stop_reasons.append(error)

patch_passed = bool((patch_plan or {}).get("action_state_adapter_patch_plan_passed"))
pure_ready = bool((patch_plan or {}).get("ready_for_pure_adapter_implementation"))
sample_policy = (single_sample or {}).get("policy") or {}
sample_interface = (single_sample or {}).get("interface") or {}
sample_adapter_metadata = sample_interface.get("adapter_metadata") or {}
adapter_metadata_recorded = bool(sample_policy.get("adapter_metadata_recorded") and sample_adapter_metadata)

adapter_has_action = "def adapt_policy_action_to_env_action" in adapter_text
adapter_has_state = "def adapt_observation_state" in adapter_text
adapter_has_image = "def select_image_source" in adapter_text
bridge_uses_action_adapter = "adapt_policy_action_to_env_action" in bridge_text
bridge_uses_state_adapter = "adapt_observation_state" in bridge_text
bridge_uses_image_adapter = "select_image_source" in bridge_text
bridge_has_implicit_padding = "values.extend([0.0]" in bridge_text and "return values[:action_dim]" in bridge_text
bridge_has_state_truncation = "values = values[:dim]" in bridge_text
bridge_has_fallback_image_selector = "def _select_image_array" in bridge_text

source_audit = {
    "adapter_has_action_adapter": adapter_has_action,
    "adapter_has_state_adapter": adapter_has_state,
    "adapter_has_image_adapter": adapter_has_image,
    "bridge_uses_action_adapter": bridge_uses_action_adapter,
    "bridge_uses_state_adapter": bridge_uses_state_adapter,
    "bridge_uses_image_adapter": bridge_uses_image_adapter,
    "bridge_has_implicit_action_padding": bridge_has_implicit_padding,
    "bridge_has_state_truncation": bridge_has_state_truncation,
    "bridge_has_fallback_image_selector": bridge_has_fallback_image_selector,
    "bridge_needs_action_adapter": adapter_has_action and not bridge_uses_action_adapter and bridge_has_implicit_padding,
    "bridge_needs_state_adapter": adapter_has_state and not bridge_uses_state_adapter and bridge_has_state_truncation,
    "bridge_needs_image_adapter": adapter_has_image and not bridge_uses_image_adapter and bridge_has_fallback_image_selector,
}

if not patch_passed:
    stop_reasons.append("Action/state adapter patch plan has not passed.")
if not pure_ready:
    stop_reasons.append("Pure adapter implementation is not marked ready by the patch plan.")
if not adapter_metadata_recorded:
    stop_reasons.append("Single-sample adapter metadata has not been recorded.")
if not (adapter_has_action and adapter_has_state and adapter_has_image):
    stop_reasons.append("Pure adapter helper source is incomplete.")

ready_for_wiring = bool(
    not stop_reasons
    and source_audit["bridge_needs_action_adapter"]
    and source_audit["bridge_needs_state_adapter"]
    and source_audit["bridge_needs_image_adapter"]
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

wiring_plan = {
    "implementation_scope": [
        "replace _policy_action_to_env_action with adapt_policy_action_to_env_action",
        "replace _state_tensor internals with adapt_observation_state and explicit state fields",
        "replace _select_image_array alias logic with select_image_source while keeping tensor resize behavior local",
        "record adapter metadata per task in rollout reports",
        "keep benchmark/paper claim flags false",
    ],
    "must_not_do_in_wiring_branch": [
        "run rollout",
        "load model",
        "run simulator",
        "change adapter strategy based on privileged simulator state",
        "download assets",
        "execute OpenVLA-OFT",
    ],
    "post_wiring_validation": [
        "unit tests for bridge helper calls",
        "safe runner",
        "explicit pytest",
        "separate bounded diagnostic rollout risk gate before execution",
    ],
}

decision = "proceed" if ready_for_wiring else "stop"
reason = "Rollout bridge is ready for adapter wiring implementation, but not rollout execution." if ready_for_wiring else "Rollout bridge adapter wiring prerequisites are not satisfied."
report = {
    "rollout_bridge_adapter_wiring_plan_passed": ready_for_wiring,
    "decision": decision,
    "reason": reason,
    "source_reports": {
        "action_state_adapter_patch_plan": str(PATCH_PLAN),
        "single_sample_interface_report": str(SINGLE_SAMPLE),
        "rollout_bridge_source": str(BRIDGE_SOURCE),
        "adapter_source": str(ADAPTER_SOURCE),
    },
    "evidence_label": "rollout_bridge_adapter_wiring_plan",
    "policy": policy,
    "claims": claims,
    "inputs": {
        "patch_plan_passed": patch_passed,
        "pure_adapter_ready": pure_ready,
        "single_sample_adapter_metadata_recorded": adapter_metadata_recorded,
        "single_sample_action_adapter": (sample_adapter_metadata.get("action_adapter") or {}).get("adapter_mode"),
        "single_sample_state_adapter": (sample_adapter_metadata.get("state_adapter") or {}).get("adapter"),
    },
    "source_audit": source_audit,
    "wiring_plan": wiring_plan,
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_rollout_bridge_adapter_wiring": ready_for_wiring,
    "ready_for_rollout_execution": False,
    "recommended_next_step": (
        "Implement rollout bridge adapter wiring with unit tests only; do not run rollout until a separate bounded diagnostic gate is green."
        if ready_for_wiring
        else "Fix missing inputs before rollout bridge adapter wiring."
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
