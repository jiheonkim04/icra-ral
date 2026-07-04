param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ActionAuditReportPath = "reports\action_interface_metadata_audit_report.json",
    [string]$ZeroPolicyComparisonPath = "reports\zero_action_policy_diagnostic_comparison_report.json",
    [string]$SmolVlaCkptPath = "C:\assets\checkpoints\smolvla",
    [string]$SourceFilePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$JsonReportPath = "reports\action_state_adapter_patch_plan_report.json",
    [string]$MarkdownReportPath = "reports\action_state_adapter_patch_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Action/state adapter patch planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This planner reads reports and metadata only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$auditFullPath = Resolve-RepoPath -Path $ActionAuditReportPath
$comparisonFullPath = Resolve-RepoPath -Path $ZeroPolicyComparisonPath
$sourceFullPath = Resolve-RepoPath -Path $SourceFilePath
$configPath = Join-Path $SmolVlaCkptPath "config.json"
$preprocessorPath = Join-Path $SmolVlaCkptPath "policy_preprocessor.json"
$postprocessorPath = Join-Path $SmolVlaCkptPath "policy_postprocessor.json"
$jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
$markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath

$env:TCA_MAP_ADAPTER_PLAN_AUDIT = $auditFullPath
$env:TCA_MAP_ADAPTER_PLAN_COMPARISON = $comparisonFullPath
$env:TCA_MAP_ADAPTER_PLAN_CONFIG = $configPath
$env:TCA_MAP_ADAPTER_PLAN_PREPROCESSOR = $preprocessorPath
$env:TCA_MAP_ADAPTER_PLAN_POSTPROCESSOR = $postprocessorPath
$env:TCA_MAP_ADAPTER_PLAN_SOURCE = $sourceFullPath
$env:TCA_MAP_ADAPTER_PLAN_JSON = $jsonFullPath
$env:TCA_MAP_ADAPTER_PLAN_MARKDOWN = $markdownFullPath
$env:TCA_MAP_ADAPTER_PLAN_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
import re
from pathlib import Path

AUDIT = Path(os.environ["TCA_MAP_ADAPTER_PLAN_AUDIT"])
COMPARISON = Path(os.environ["TCA_MAP_ADAPTER_PLAN_COMPARISON"])
CONFIG = Path(os.environ["TCA_MAP_ADAPTER_PLAN_CONFIG"])
PRE = Path(os.environ["TCA_MAP_ADAPTER_PLAN_PREPROCESSOR"])
POST = Path(os.environ["TCA_MAP_ADAPTER_PLAN_POSTPROCESSOR"])
SOURCE = Path(os.environ["TCA_MAP_ADAPTER_PLAN_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_ADAPTER_PLAN_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_ADAPTER_PLAN_MARKDOWN"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_ADAPTER_PLAN_GATES", "").split(";") if item]

def load_json(path):
    if not path.exists():
        return None, f"Missing input: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:
        return None, f"Could not read {path}: {exc}"

def shape(feature_map, key):
    value = feature_map.get(key) if isinstance(feature_map, dict) else None
    if isinstance(value, dict):
        return value.get("shape")
    return None

def processor_features(processor):
    if not isinstance(processor, dict):
        return {}
    for step in processor.get("steps", []):
        if not isinstance(step, dict):
            continue
        config = step.get("config") or {}
        features = config.get("features")
        if isinstance(features, dict):
            return features
    return {}

def write_outputs(report):
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Action/State Adapter Patch Plan Report",
        "",
        f"- decision: {report['decision']}",
        f"- planner passed: {report['action_state_adapter_patch_plan_passed']}",
        f"- ready for pure adapter implementation: {report['ready_for_pure_adapter_implementation']}",
        f"- ready for rollout scaling: {report['ready_for_rollout_scaling']}",
        f"- action patch required: {report['patch_plan']['action_adapter']['required']}",
        f"- state patch required: {report['patch_plan']['state_adapter']['required']}",
        f"- camera alias patch required: {report['patch_plan']['camera_adapter']['required']}",
        f"- standard success claimed: {report['claims']['standard_success_claimed']}",
        f"- paper-grade claim made: {report['claims']['paper_grade_claim_made']}",
        "",
        report["recommended_next_step"],
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

audit, audit_error = load_json(AUDIT)
comparison, comparison_error = load_json(COMPARISON)
config, config_error = load_json(CONFIG)
pre, pre_error = load_json(PRE)
post, post_error = load_json(POST)
source_text = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""

stop_reasons = []
warnings = []
if SET_GATES:
    stop_reasons.append("Refusing patch planning while execution gates are set: " + ", ".join(SET_GATES))
for label, error, required in [
    ("action-interface metadata audit", audit_error, True),
    ("zero-action policy comparison", comparison_error, True),
    ("SmolVLA config", config_error, False),
    ("SmolVLA preprocessor", pre_error, False),
    ("SmolVLA postprocessor", post_error, False),
]:
    if error and required:
        stop_reasons.append(error)
    elif error:
        warnings.append(error)
if not SOURCE.exists():
    warnings.append(f"Missing source file: {SOURCE}")

audit_findings = set((audit or {}).get("high_priority_findings") or [])
comparison_ready = bool((comparison or {}).get("ready_for_action_state_adapter_patch_plan"))
comparison_data = (comparison or {}).get("comparison") or {}
learned = comparison_data.get("learned_policy") or {}
zero = comparison_data.get("zero_action") or {}

input_features = (config or {}).get("input_features") or {}
output_features = (config or {}).get("output_features") or {}
pre_features = processor_features(pre or {})
post_features = processor_features(post or {})

config_action_shape = shape(output_features, "action")
post_action_shape = shape(post_features, "action")
config_state_shape = shape(input_features, "observation.state")
pre_state_shape = shape(pre_features, "observation.state")
config_images = sorted(k for k, v in input_features.items() if isinstance(v, dict) and v.get("type") == "VISUAL")
pre_images = sorted(k for k, v in pre_features.items() if isinstance(v, dict) and v.get("type") == "VISUAL")

has_implicit_action_padding = "values.extend([0.0]" in source_text and "return values[:action_dim]" in source_text
has_state_truncation = "values = values[:dim]" in source_text
has_image_alias_fallback = "_select_image_array" in source_text and "agentview_image" in source_text

action_required = (
    "action_dim_mismatch" in audit_findings
    or "gripper_constant_zero" in audit_findings
    or has_implicit_action_padding
)
state_required = "state_truncation_risk" in audit_findings or has_state_truncation
camera_required = "camera_feature_name_mismatch" in audit_findings or (config_images and pre_images and config_images != pre_images)

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

patch_plan = {
    "action_adapter": {
        "required": action_required,
        "current_problem": "Current bridge silently pads/truncates policy action to environment action dimension.",
        "current_code_signal": {
            "implicit_zero_padding_detected": has_implicit_action_padding,
            "policy_action_shapes": learned.get("policy_action_shapes"),
            "env_action_dims": learned.get("env_action_dims"),
            "last_gripper_component": learned.get("last_env_action_gripper_component"),
        },
        "required_change": "Introduce an explicit action adapter and remove silent padding/truncation from rollout code.",
        "initial_strategy": "policy_6d_delta_pose_plus_explicit_gripper_hold",
        "allowed_diagnostic_strategies": [
            "policy_6d_delta_pose_plus_gripper_zero_hold",
            "policy_6d_delta_pose_plus_gripper_open",
            "policy_6d_delta_pose_plus_gripper_close",
        ],
        "forbidden_shortcuts": [
            "silent zero padding without report metadata",
            "changing environment action dimension to hide mismatch",
            "using privileged simulator state at inference",
            "claiming benchmark success from adapter diagnostics",
        ],
        "acceptance_checks": [
            "adapter output length equals env action_dim",
            "gripper strategy is named in runtime report",
            "action clipping range is explicit",
            "unit tests cover 6D-to-7D mapping and refusal of unsupported dimensions",
        ],
    },
    "state_adapter": {
        "required": state_required,
        "current_problem": "Current state builder flattens multiple observation fields and truncates to policy state dimension.",
        "current_code_signal": {
            "silent_truncation_detected": has_state_truncation,
            "config_state_shape": config_state_shape,
            "preprocessor_state_shape": pre_state_shape,
        },
        "required_change": "Introduce an explicit state adapter with named key order, transforms, and no silent truncation.",
        "candidate_inputs_to_audit": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
        ],
        "forbidden_shortcuts": [
            "values[:dim] truncation without an adapter name",
            "padding missing proprioception without reporting missing keys",
            "using simulator privileged target/object state at inference",
        ],
        "acceptance_checks": [
            "adapter report lists source observation keys",
            "state vector length equals policy observation.state shape",
            "missing keys are reported explicitly",
            "unit tests cover missing-key and no-silent-truncation behavior",
        ],
    },
    "camera_adapter": {
        "required": camera_required,
        "current_problem": "Config image feature names and preprocessor image feature names differ.",
        "current_code_signal": {
            "config_image_features": config_images,
            "preprocessor_image_features": pre_images,
            "fallback_selector_detected": has_image_alias_fallback,
        },
        "required_change": "Create an explicit image key alias table and report selected RoboSuite observation source for each policy feature.",
        "candidate_aliases": {
            "observation.images.camera1": ["agentview_image", "agentview_rgb"],
            "observation.images.camera2": ["robot0_eye_in_hand_image", "eye_in_hand_image"],
            "observation.images.camera3": ["agentview_image", "robot0_eye_in_hand_image"],
        },
        "acceptance_checks": [
            "adapter report lists feature-to-source image mapping",
            "single-sample smoke records image feature aliases",
            "bounded diagnostic rollout records selected source keys",
        ],
    },
    "implementation_sequence": [
        "Add pure action/state/image adapter helpers plus unit tests; no simulator, model load, GPU, or rollout.",
        "Wire adapters into synthetic single-sample interface smoke and report adapter metadata.",
        "Run one bounded diagnostic rollout with task-local gate and adapter metadata after pure tests pass.",
        "Compare adapter diagnostic against zero-action and legacy SmolVLA-action diagnostics before scaling.",
    ],
}

passed = bool(not stop_reasons and (comparison_ready or action_required or state_required or camera_required))
decision = "proceed" if passed else "stop"
reason = "Action/state adapter patch plan generated from existing diagnostics." if passed else "Patch plan prerequisites are not satisfied."
ready_for_pure_adapter = bool(passed and (action_required or state_required or camera_required))
ready_for_rollout_scaling = False

report = {
    "action_state_adapter_patch_plan_passed": passed,
    "decision": decision,
    "reason": reason,
    "source_reports": {
        "action_interface_metadata_audit": str(AUDIT),
        "zero_action_policy_comparison": str(COMPARISON),
        "smolvla_config": str(CONFIG),
        "smolvla_preprocessor": str(PRE),
        "smolvla_postprocessor": str(POST),
        "rollout_bridge_source": str(SOURCE),
    },
    "evidence_label": "action_state_adapter_patch_plan",
    "policy": policy,
    "claims": claims,
    "diagnostic_context": {
        "audit_high_priority_findings": sorted(audit_findings),
        "comparison_ready_for_patch_plan": comparison_ready,
        "zero_action_reward_sum": zero.get("reward_sum_total"),
        "learned_policy_reward_sum": learned.get("reward_sum_total"),
        "learned_policy_action_nontrivial": comparison_data.get("policy_action_nontrivial"),
        "learned_policy_outperformed_zero_action": comparison_data.get("learned_policy_outperformed_zero_action"),
    },
    "metadata_context": {
        "config_action_shape": config_action_shape,
        "postprocessor_action_shape": post_action_shape,
        "config_state_shape": config_state_shape,
        "preprocessor_state_shape": pre_state_shape,
        "config_image_features": config_images,
        "preprocessor_image_features": pre_images,
    },
    "patch_plan": patch_plan,
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_pure_adapter_implementation": ready_for_pure_adapter,
    "ready_for_rollout_scaling": ready_for_rollout_scaling,
    "recommended_next_step": (
        "Implement pure action/state/image adapter helpers with unit tests; do not run rollout until adapter tests and single-sample smoke pass."
        if ready_for_pure_adapter
        else "Fix missing diagnostic inputs before implementing adapter helpers."
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
