param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PlanReportPath = "reports\action_interface_diagnostic_plan_report.json",
    [string]$MetricSummaryReportPath = "reports\reduced_scope_rollout_metric_summary_report.json",
    [string]$SmolVlaCkptPath = "C:\assets\checkpoints\smolvla",
    [string]$SourceFilePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$JsonReportPath = "reports\action_interface_metadata_audit_report.json",
    [string]$MarkdownReportPath = "reports\action_interface_metadata_audit_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Action-interface metadata audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads metadata and existing reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$planFullPath = Resolve-RepoPath -Path $PlanReportPath
$metricFullPath = Resolve-RepoPath -Path $MetricSummaryReportPath
$sourceFullPath = Resolve-RepoPath -Path $SourceFilePath
$jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
$markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath
$configPath = Join-Path $SmolVlaCkptPath "config.json"
$preprocessorPath = Join-Path $SmolVlaCkptPath "policy_preprocessor.json"
$postprocessorPath = Join-Path $SmolVlaCkptPath "policy_postprocessor.json"

$env:TCA_MAP_ACTION_AUDIT_PLAN = $planFullPath
$env:TCA_MAP_ACTION_AUDIT_METRIC = $metricFullPath
$env:TCA_MAP_ACTION_AUDIT_CONFIG = $configPath
$env:TCA_MAP_ACTION_AUDIT_PREPROCESSOR = $preprocessorPath
$env:TCA_MAP_ACTION_AUDIT_POSTPROCESSOR = $postprocessorPath
$env:TCA_MAP_ACTION_AUDIT_SOURCE = $sourceFullPath
$env:TCA_MAP_ACTION_AUDIT_JSON = $jsonFullPath
$env:TCA_MAP_ACTION_AUDIT_MARKDOWN = $markdownFullPath
$env:TCA_MAP_ACTION_AUDIT_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
import re
from pathlib import Path

PLAN = Path(os.environ["TCA_MAP_ACTION_AUDIT_PLAN"])
METRIC = Path(os.environ["TCA_MAP_ACTION_AUDIT_METRIC"])
CONFIG = Path(os.environ["TCA_MAP_ACTION_AUDIT_CONFIG"])
PRE = Path(os.environ["TCA_MAP_ACTION_AUDIT_PREPROCESSOR"])
POST = Path(os.environ["TCA_MAP_ACTION_AUDIT_POSTPROCESSOR"])
SOURCE = Path(os.environ["TCA_MAP_ACTION_AUDIT_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_ACTION_AUDIT_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_ACTION_AUDIT_MARKDOWN"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_ACTION_AUDIT_GATES", "").split(";") if item]

def load_json(path):
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:
        return None, str(exc)

def shape_from_feature_map(feature_map, key):
    if not isinstance(feature_map, dict):
        return None
    value = feature_map.get(key)
    if isinstance(value, dict):
        return value.get("shape")
    return None

def features_from_processor(processor):
    steps = processor.get("steps", []) if isinstance(processor, dict) else []
    for step in steps:
        config = step.get("config", {}) if isinstance(step, dict) else {}
        features = config.get("features")
        if isinstance(features, dict):
            return features
    return {}

def processor_steps(processor):
    if not isinstance(processor, dict):
        return []
    return [step.get("registry_name") for step in processor.get("steps", []) if isinstance(step, dict)]

plan, plan_error = load_json(PLAN)
metric, metric_error = load_json(METRIC)
config, config_error = load_json(CONFIG)
pre, pre_error = load_json(PRE)
post, post_error = load_json(POST)
source_text = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""

stop_reasons = []
warnings = []
if SET_GATES:
    stop_reasons.append("Execution gates are set during metadata audit: " + ", ".join(SET_GATES))
for label, error, required in [
    ("action-interface plan", plan_error, True),
    ("reduced-scope metric summary", metric_error, True),
    ("SmolVLA config", config_error, True),
    ("SmolVLA preprocessor", pre_error, False),
    ("SmolVLA postprocessor", post_error, False),
]:
    if error and required:
        stop_reasons.append(f"Could not read {label}: {error}")
    elif error:
        warnings.append(f"Could not read {label}: {error}")
if not SOURCE.exists():
    warnings.append(f"Rollout bridge source file missing: {SOURCE}")

metric_summary = (metric or {}).get("metric_summary", {})
observed = (plan or {}).get("observed_signals", {})
config_inputs = (config or {}).get("input_features", {})
config_outputs = (config or {}).get("output_features", {})
pre_features = features_from_processor(pre or {})
post_features = features_from_processor(post or {})

config_action_shape = shape_from_feature_map(config_outputs, "action")
pre_action_shape = shape_from_feature_map(pre_features, "action")
post_action_shape = shape_from_feature_map(post_features, "action")
config_state_shape = shape_from_feature_map(config_inputs, "observation.state")
pre_state_shape = shape_from_feature_map(pre_features, "observation.state")
config_image_shapes = {
    key: value.get("shape")
    for key, value in config_inputs.items()
    if isinstance(value, dict) and value.get("type") == "VISUAL"
}
pre_image_shapes = {
    key: value.get("shape")
    for key, value in pre_features.items()
    if isinstance(value, dict) and value.get("type") == "VISUAL"
}

policy_action_dim = observed.get("policy_action_dim")
env_action_dim = observed.get("env_action_dim")
if policy_action_dim is None and config_action_shape:
    policy_action_dim = config_action_shape[-1]
gripper_component = observed.get("gripper_component")
success_rate = observed.get("diagnostic_success_rate")
reward_sum = observed.get("reward_sum_total")
action_max_abs = observed.get("action_max_abs")
action_l2 = observed.get("action_l2")

expected_state_keys = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
]
state_builder_unique = [key for key in expected_state_keys if key in source_text]
state_truncates = "values = values[:dim]" in source_text
state_dim = config_state_shape[-1] if config_state_shape else None
bridge_uses_action_adapter = "adapt_policy_action_to_env_action" in source_text
bridge_uses_state_adapter = "adapt_observation_state" in source_text
bridge_uses_image_adapter = "select_image_source" in source_text
bridge_has_implicit_action_padding = "values.extend([0.0]" in source_text or "return values[:action_dim]" in source_text

findings = []
def add_finding(name, severity, evidence, recommendation):
    findings.append({
        "name": name,
        "severity": severity,
        "evidence": evidence,
        "recommendation": recommendation,
    })

action_dim_mismatch = policy_action_dim is not None and env_action_dim is not None and int(policy_action_dim) != int(env_action_dim)
if action_dim_mismatch:
    if bridge_uses_action_adapter and not bridge_has_implicit_action_padding:
        add_finding(
            "action_dim_mismatch_explicit_adapter_in_use",
            "high",
            {
                "policy_action_dim": policy_action_dim,
                "env_action_dim": env_action_dim,
                "bridge_uses_action_adapter": bridge_uses_action_adapter,
                "bridge_has_implicit_action_padding": bridge_has_implicit_action_padding,
            },
            "Diagnose adapter strategy, action scale, and gripper semantics before rollout scaling.",
        )
    else:
        add_finding(
            "action_dim_mismatch",
            "high",
            {"policy_action_dim": policy_action_dim, "env_action_dim": env_action_dim},
            "Create an explicit action adapter policy instead of implicit padding/truncation.",
        )
if action_dim_mismatch and gripper_component == 0.0:
    if bridge_uses_action_adapter and not bridge_has_implicit_action_padding:
        add_finding(
            "gripper_zero_hold_strategy_requires_validation",
            "high",
            {
                "gripper_component": gripper_component,
                "bridge_uses_action_adapter": bridge_uses_action_adapter,
            },
            "Compare diagnostic gripper strategies and action scaling before rollout scaling.",
        )
    else:
        add_finding(
            "gripper_constant_zero",
            "high",
            {"gripper_component": gripper_component},
            "Audit whether gripper should be held, opened, closed, copied from dataset, or controlled by a separate head.",
        )
if state_dim == 6 and state_truncates and len(state_builder_unique) > 2:
    add_finding(
        "state_truncation_risk",
        "high",
        {"state_dim": state_dim, "state_builder_keys": state_builder_unique, "state_truncates": state_truncates},
        "Replace truncation with an explicit state adapter matching the policy training convention.",
    )
if config_action_shape and post_action_shape and config_action_shape != post_action_shape:
    add_finding(
        "postprocessor_action_shape_mismatch",
        "high",
        {"config_action_shape": config_action_shape, "postprocessor_action_shape": post_action_shape},
        "Do not run further policy rollouts until action postprocessor shape is resolved.",
    )
else:
    add_finding(
        "action_normalization_metadata_present",
        "info",
        {"config_action_shape": config_action_shape, "postprocessor_action_shape": post_action_shape, "postprocessor_steps": processor_steps(post or {})},
        "Use this metadata in the action adapter audit; do not bypass unnormalization.",
    )
if config_image_shapes and pre_image_shapes and set(config_image_shapes) != set(pre_image_shapes):
    add_finding(
        "camera_feature_name_mismatch",
        "medium",
        {"config_image_features": config_image_shapes, "preprocessor_image_features": pre_image_shapes},
        "Audit camera key aliases and confirm the loaded policy config uses the expected image feature names.",
    )
if success_rate == 0.0 and reward_sum == 0.0 and action_max_abs and float(action_max_abs) > 0.05:
    add_finding(
        "nontrivial_actions_zero_reward",
        "high",
        {"success_rate": success_rate, "reward_sum": reward_sum, "action_max_abs": action_max_abs, "action_l2": action_l2},
        "Run a zero-action versus SmolVLA-action comparison and log object distance/failure observations before scaling.",
    )

decision = "stop" if stop_reasons else "proceed"
reason = "Action-interface metadata audit completed; fix interface risks before rollout scaling." if not stop_reasons else "Metadata audit prerequisites are not satisfied."
high_findings = [item for item in findings if item["severity"] == "high"]

report = {
    "action_interface_metadata_audit_passed": decision == "proceed",
    "decision": decision,
    "reason": reason,
    "policy": {
        "metadata_only": True,
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
    "evidence_policy": {
        "evidence_label": "action_interface_metadata_audit",
        "standard_success_claimed": False,
        "benchmark_success_claimed": False,
        "paper_grade_claim_made": False,
    },
    "inputs": {
        "plan_report": str(PLAN),
        "metric_report": str(METRIC),
        "config": str(CONFIG),
        "preprocessor": str(PRE),
        "postprocessor": str(POST),
        "source_file": str(SOURCE),
    },
    "observed": {
        "policy_action_dim": policy_action_dim,
        "env_action_dim": env_action_dim,
        "gripper_component": gripper_component,
        "diagnostic_success_rate": success_rate,
        "reward_sum_total": reward_sum,
        "action_max_abs": action_max_abs,
        "action_l2": action_l2,
    },
    "metadata": {
        "config_action_shape": config_action_shape,
        "preprocessor_action_shape": pre_action_shape,
        "postprocessor_action_shape": post_action_shape,
        "config_state_shape": config_state_shape,
        "preprocessor_state_shape": pre_state_shape,
        "config_image_shapes": config_image_shapes,
        "preprocessor_image_shapes": pre_image_shapes,
        "normalization_mapping": (config or {}).get("normalization_mapping"),
        "preprocessor_steps": processor_steps(pre or {}),
        "postprocessor_steps": processor_steps(post or {}),
        "state_builder_keys": state_builder_unique,
        "state_builder_truncates_to_dim": state_truncates,
        "bridge_uses_action_adapter": bridge_uses_action_adapter,
        "bridge_uses_state_adapter": bridge_uses_state_adapter,
        "bridge_uses_image_adapter": bridge_uses_image_adapter,
        "bridge_has_implicit_action_padding": bridge_has_implicit_action_padding,
    },
    "findings": findings,
    "high_priority_findings": [item["name"] for item in high_findings],
    "warnings": warnings,
    "stop_reasons": stop_reasons,
    "ready_for_zero_action_vs_policy_action_diagnostic": decision == "proceed",
    "ready_for_action_adapter_patch_plan": (
        decision == "proceed" and bool(high_findings) and not (bridge_uses_action_adapter and not bridge_has_implicit_action_padding)
    ),
    "ready_for_adapter_strategy_diagnosis": (
        decision == "proceed" and bool(high_findings) and bridge_uses_action_adapter and not bridge_has_implicit_action_padding
    ),
    "recommended_next_step": (
        (
            "Run adapter-strategy/action-scale diagnostics before rollout scaling."
            if bridge_uses_action_adapter and not bridge_has_implicit_action_padding
            else "Create a bounded zero-action versus SmolVLA-action diagnostic and an explicit action/state adapter patch plan before rollout scaling."
        )
        if decision == "proceed"
        else "Fix missing metadata/report inputs before action-interface audit."
    ),
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Action-Interface Metadata Audit Report",
    "",
    f"- decision: {decision}",
    f"- reason: {reason}",
    f"- high-priority findings: {report['high_priority_findings']}",
    f"- policy action dim: {policy_action_dim}",
    f"- env action dim: {env_action_dim}",
    f"- gripper component: {gripper_component}",
    f"- diagnostic success rate: {success_rate}",
    f"- reward sum: {reward_sum}",
    f"- ready for zero-action vs policy diagnostic: {report['ready_for_zero_action_vs_policy_action_diagnostic']}",
    f"- ready for action adapter patch plan: {report['ready_for_action_adapter_patch_plan']}",
    f"- ready for adapter strategy diagnosis: {report['ready_for_adapter_strategy_diagnosis']}",
    "",
    report["recommended_next_step"],
    "",
]
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    exit $exitCode
}
