param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmolVLACkpt = "",
    [string]$OfflineDecodingReportPath = "reports\offline_demo_action_decoding_report.json",
    [string]$OfflineSummaryPath = "reports\offline_demo_action_decoding_summary_report.json",
    [string]$LoadOnlyReportPath = "reports\smolvla_load_only_smoke_report.json",
    [string]$JsonReportPath = "reports\vlm_loading_policy_action_normalization_audit_report.json",
    [string]$MarkdownReportPath = "reports\vlm_loading_policy_action_normalization_audit_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "VLM loading policy and action-normalization audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads local configs and existing diagnostic reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
    "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
    "ALLOW_ACTION_SCALE_DIAGNOSTIC",
    "ALLOW_PROMPT_FORMAT_DIAGNOSTIC",
    "ALLOW_CAMERA_SOURCE_DIAGNOSTIC",
    "ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC",
    "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
    "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK",
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

$env:TCA_MAP_VLM_AUDIT_SMOLVLA = $SmolVLACkpt
$env:TCA_MAP_VLM_AUDIT_OFFLINE = Resolve-RepoPath -Path $OfflineDecodingReportPath
$env:TCA_MAP_VLM_AUDIT_SUMMARY = Resolve-RepoPath -Path $OfflineSummaryPath
$env:TCA_MAP_VLM_AUDIT_LOAD_ONLY = Resolve-RepoPath -Path $LoadOnlyReportPath
$env:TCA_MAP_VLM_AUDIT_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_VLM_AUDIT_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_VLM_AUDIT_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

SMOLVLA = Path(os.environ["TCA_MAP_VLM_AUDIT_SMOLVLA"])
OFFLINE = Path(os.environ["TCA_MAP_VLM_AUDIT_OFFLINE"])
SUMMARY = Path(os.environ["TCA_MAP_VLM_AUDIT_SUMMARY"])
LOAD_ONLY = Path(os.environ["TCA_MAP_VLM_AUDIT_LOAD_ONLY"])
JSON_OUT = Path(os.environ["TCA_MAP_VLM_AUDIT_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_VLM_AUDIT_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_VLM_AUDIT_GATES", "").split(";") if item]


def read_json(path: Path, *, required: bool = True):
    if not path.exists():
        return None, f"Missing JSON file: {path}" if required else None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parse issue is useful here.
        return None, f"Could not parse {path}: {exc}"


def nested(mapping, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def feature_shape(config: dict, key: str):
    for group in ("input_features", "output_features"):
        feature = (config.get(group) or {}).get(key)
        if isinstance(feature, dict):
            return feature.get("shape")
    return None


def issue(axis: str, severity: str, finding: str, evidence: dict, recommendation: str) -> dict:
    return {
        "axis": axis,
        "severity": severity,
        "finding": finding,
        "evidence": evidence,
        "recommendation": recommendation,
    }


stop_reasons: list[str] = []
if SET_GATES:
    stop_reasons.append("Refusing report-only audit while execution gates are set: " + ", ".join(SET_GATES))

config, config_error = read_json(SMOLVLA / "config.json")
preprocessor, preprocessor_error = read_json(SMOLVLA / "policy_preprocessor.json")
postprocessor, postprocessor_error = read_json(SMOLVLA / "policy_postprocessor.json")
offline, offline_error = read_json(OFFLINE)
summary, summary_error = read_json(SUMMARY)
load_only, load_error = read_json(LOAD_ONLY, required=False)
for error in (config_error, preprocessor_error, postprocessor_error, offline_error, summary_error):
    if error:
        stop_reasons.append(error)
if load_error:
    stop_reasons.append(load_error)

config = config or {}
preprocessor = preprocessor or {}
postprocessor = postprocessor or {}
offline = offline or {}
summary = summary or {}
load_only = load_only or {}

preprocessor_steps = preprocessor.get("steps") or []
postprocessor_steps = postprocessor.get("steps") or []
tokenizer_name = None
pre_norm_map = {}
post_norm_map = {}
pre_state_file = None
post_state_file = None
pre_features = {}
post_features = {}
for step in preprocessor_steps:
    if not isinstance(step, dict):
        continue
    step_config = step.get("config") or {}
    if isinstance(step_config, dict) and step_config.get("tokenizer_name"):
        tokenizer_name = step_config.get("tokenizer_name")
    if isinstance(step_config, dict) and step_config.get("norm_map"):
        pre_norm_map = step_config.get("norm_map") or {}
        pre_features = step_config.get("features") or {}
        pre_state_file = step.get("state_file")
for step in postprocessor_steps:
    if not isinstance(step, dict):
        continue
    step_config = step.get("config") or {}
    if isinstance(step_config, dict) and step_config.get("norm_map"):
        post_norm_map = step_config.get("norm_map") or {}
        post_features = step_config.get("features") or {}
        post_state_file = step.get("state_file")

policy_action_shape = feature_shape(config, "action")
policy_state_shape = feature_shape(config, "observation.state")
policy_image_shapes = {
    key: value.get("shape")
    for key, value in (config.get("input_features") or {}).items()
    if isinstance(value, dict) and value.get("type") == "VISUAL"
}
config_norm_map = config.get("normalization_mapping") or {}
config_load_vlm_weights = config.get("load_vlm_weights")
offline_metrics = offline.get("metrics") or {}
offline_policy = offline.get("policy") or {}
offline_files = offline.get("files") or {}
offline_ext_dep = offline_files.get("external_tokenizer_dependency") or {}
offline_sample = offline.get("sample") or {}
offline_action_adapter = offline_metrics.get("action_adapter_metadata") or {}
summary_metrics = summary.get("metrics") or {}
observed_load_vlm_weights = offline_metrics.get("load_vlm_weights")
if observed_load_vlm_weights is None:
    observed_load_vlm_weights = nested(load_only, "load", "load_vlm_weights")

action_l1 = summary_metrics.get("action_l1_to_expert", offline_metrics.get("action_l1_to_expert"))
action_mse = summary_metrics.get("action_mse_to_expert", offline_metrics.get("action_mse_to_expert"))
policy6_l1 = summary_metrics.get("policy6_l1_to_expert_first6", offline_metrics.get("policy6_l1_to_expert_first6"))
alignment_signal = summary_metrics.get("offline_alignment_signal", "unknown")
policy_action_preview = offline_metrics.get("policy_action_preview") or []
adapted_action_preview = offline_metrics.get("adapted_action_preview") or []
expert_action_preview = offline_metrics.get("expert_action_preview") or []
clipped_values = offline_action_adapter.get("clipped_values")
adapter_strategy = offline_action_adapter.get("strategy")
gripper_value = offline_action_adapter.get("gripper_value")
image_sources = nested(offline_metrics, "batch_metadata", "image_sources", default={}) or {}
image_adapters = nested(offline_metrics, "batch_metadata", "image_adapters", default={}) or {}
resized_images = [
    key
    for key, value in image_adapters.items()
    if isinstance(value, dict) and value.get("resized")
]

external_tokenizer_only = bool(offline_ext_dep.get("found")) and not any(
    str(name).endswith((".safetensors", ".bin"))
    for name in offline_ext_dep.get("files_found", [])
)

issues: list[dict] = []
if config_load_vlm_weights is True and observed_load_vlm_weights is False:
    issues.append(
        issue(
            "vlm_loading_policy",
            "high",
            "The checkpoint config requests VLM weights, but the local diagnostic loaded with VLM weights disabled.",
            {
                "config_load_vlm_weights": config_load_vlm_weights,
                "observed_load_vlm_weights": observed_load_vlm_weights,
                "vlm_model_name": config.get("vlm_model_name"),
                "external_tokenizer_dependency_found": offline_ext_dep.get("found"),
                "external_dependency_appears_tokenizer_only": external_tokenizer_only,
            },
            "Do not scale rollouts from disabled-VLM diagnostics. Either keep evidence as interface diagnostics or create a separate risk assessment for VLM-enabled load if official weights, size, and memory are green.",
        )
    )
else:
    issues.append(
        issue(
            "vlm_loading_policy",
            "medium",
            "The observed VLM loading policy still needs to be recorded for any later diagnostic comparison.",
            {
                "config_load_vlm_weights": config_load_vlm_weights,
                "observed_load_vlm_weights": observed_load_vlm_weights,
                "vlm_model_name": config.get("vlm_model_name"),
            },
            "Keep load policy explicit in every offline decoding and rollout report.",
        )
    )

if config_norm_map.get("ACTION") == "MEAN_STD" and post_norm_map.get("ACTION") == "MEAN_STD":
    issues.append(
        issue(
            "action_normalization",
            "high" if alignment_signal == "weak" else "medium",
            "Action outputs use MEAN_STD unnormalization, while the one-sample offline action remained far from the expert action.",
            {
                "config_normalization_mapping": config_norm_map,
                "preprocessor_norm_map": pre_norm_map,
                "postprocessor_norm_map": post_norm_map,
                "preprocessor_state_file": pre_state_file,
                "postprocessor_state_file": post_state_file,
                "action_l1_to_expert": action_l1,
                "action_mse_to_expert": action_mse,
                "policy6_l1_to_expert_first6": policy6_l1,
                "offline_alignment_signal": alignment_signal,
            },
            "Before another rollout, inspect action scale/unnormalization assumptions with a repeated offline HDF5 decoding check.",
        )
    )

if clipped_values and clipped_values > 0:
    issues.append(
        issue(
            "action_adapter_clipping",
            "high",
            "The adapted 7D action clipped at least one value, which can erase a meaningful continuous control magnitude.",
            {
                "policy_action_preview": policy_action_preview,
                "adapted_action_preview": adapted_action_preview,
                "expert_action_preview": expert_action_preview,
                "clipped_values": clipped_values,
                "adapter_strategy": adapter_strategy,
                "gripper_value": gripper_value,
            },
            "Test bounded offline action-scale/adapter alternatives against expert actions before simulator rollout scaling.",
        )
    )

if policy_action_shape and offline_sample.get("expert_action_shape") and policy_action_shape[-1] != offline_sample.get("expert_action_shape")[-1]:
    issues.append(
        issue(
            "action_dimension_convention",
            "high",
            "The policy emits 6D actions while the LIBERO expert action target is 7D.",
            {
                "policy_action_shape": policy_action_shape,
                "expert_action_shape": offline_sample.get("expert_action_shape"),
                "adapter_strategy": adapter_strategy,
            },
            "Keep action adapter strategy explicit and compare against expert actions offline before additional learned-policy rollout.",
        )
    )

if resized_images or len(policy_image_shapes) == 3:
    issues.append(
        issue(
            "observation_image_convention",
            "medium",
            "The offline diagnostic resized HDF5 images and duplicated/aliased camera inputs to satisfy three policy image slots.",
            {
                "policy_image_shapes": policy_image_shapes,
                "offline_image_sources": image_sources,
                "resized_image_keys": resized_images,
            },
            "Keep image aliasing in the report and prefer offline ablations before assigning failure to policy competence.",
        )
    )

if not config.get("repo_id"):
    issues.append(
        issue(
            "checkpoint_provenance",
            "medium",
            "The local checkpoint config does not record a repo_id or task-suite provenance.",
            {
                "repo_id": config.get("repo_id"),
                "license": config.get("license"),
                "push_to_hub": config.get("push_to_hub"),
            },
            "Record provenance as unknown and avoid treating the selected LIBERO task as in-distribution evidence.",
        )
    )

high_count = sum(1 for item in issues if item["severity"] == "high")
passed = not stop_reasons
decision = "no_go_rollout_scaling" if passed else "stop"
ready_for_repeated_offline_decoding_plan = bool(passed and high_count > 0)

report = {
    "evidence_label": "vlm_loading_policy_action_normalization_audit",
    "vlm_loading_policy_action_normalization_audit_passed": passed,
    "decision": decision,
    "ready_for_rollout_scaling": False,
    "ready_for_benchmark_claim": False,
    "ready_for_paper_claim": False,
    "ready_for_repeated_offline_decoding_plan": ready_for_repeated_offline_decoding_plan,
    "high_severity_issue_count": high_count,
    "policy": {
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
    },
    "claims": {
        "standard_success_claimed": False,
        "benchmark_success_claimed": False,
        "counterfactual_robustness_claimed": False,
        "sota_claimed": False,
        "paper_grade_claim_made": False,
    },
    "paths": {
        "smolvla_ckpt": str(SMOLVLA),
        "offline_decoding_report": str(OFFLINE),
        "offline_summary_report": str(SUMMARY),
        "load_only_report": str(LOAD_ONLY),
    },
    "checkpoint_summary": {
        "type": config.get("type"),
        "repo_id": config.get("repo_id"),
        "license": config.get("license"),
        "push_to_hub": config.get("push_to_hub"),
        "vlm_model_name": config.get("vlm_model_name"),
        "config_load_vlm_weights": config_load_vlm_weights,
        "observed_load_vlm_weights": observed_load_vlm_weights,
        "tokenizer_name": tokenizer_name,
        "external_tokenizer_dependency": offline_ext_dep,
        "external_dependency_appears_tokenizer_only": external_tokenizer_only,
        "policy_state_shape": policy_state_shape,
        "policy_action_shape": policy_action_shape,
        "policy_image_shapes": policy_image_shapes,
        "config_normalization_mapping": config_norm_map,
        "preprocessor_norm_map": pre_norm_map,
        "postprocessor_norm_map": post_norm_map,
        "preprocessor_state_file": pre_state_file,
        "postprocessor_state_file": post_state_file,
        "preprocessor_feature_keys": sorted(pre_features.keys()),
        "postprocessor_feature_keys": sorted(post_features.keys()),
    },
    "offline_alignment_summary": {
        "offline_alignment_signal": alignment_signal,
        "source_diagnostic_passed": offline.get("offline_demo_action_decoding_passed"),
        "action_l1_to_expert": action_l1,
        "action_mse_to_expert": action_mse,
        "policy6_l1_to_expert_first6": policy6_l1,
        "policy_action_shape": offline_metrics.get("policy_action_shape"),
        "expert_action_shape": offline_metrics.get("expert_action_shape"),
        "policy_action_preview": policy_action_preview,
        "adapted_action_preview": adapted_action_preview,
        "expert_action_preview": expert_action_preview,
        "action_adapter_metadata": offline_action_adapter,
        "sample": offline_sample,
    },
    "issues": issues,
    "stop_reasons": stop_reasons,
    "recommended_next_step": (
        "Do not run another learned-policy rollout yet. Plan a tiny repeated offline demonstration action-decoding diagnostic over a few HDF5 timesteps, explicitly logging load_vlm_weights, action unnormalization, clipping, gripper strategy, and image aliases. Treat VLM-enabled loading or full SmolVLM2 weight acquisition as a separate risk-assessed task."
        if passed
        else "Resolve missing audit inputs before continuing."
    ),
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# VLM Loading Policy and Action-Normalization Audit Report",
    "",
    f"- decision: {report['decision']}",
    f"- audit passed: {report['vlm_loading_policy_action_normalization_audit_passed']}",
    f"- high severity issues: {report['high_severity_issue_count']}",
    f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
    f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
    f"- repeated offline decoding plan ready: {report['ready_for_repeated_offline_decoding_plan']}",
    "",
    "## Key Evidence",
    "",
    f"- config load_vlm_weights: {report['checkpoint_summary']['config_load_vlm_weights']}",
    f"- observed load_vlm_weights: {report['checkpoint_summary']['observed_load_vlm_weights']}",
    f"- VLM model name: {report['checkpoint_summary']['vlm_model_name']}",
    f"- external dependency appears tokenizer-only: {report['checkpoint_summary']['external_dependency_appears_tokenizer_only']}",
    f"- action normalization: {report['checkpoint_summary']['config_normalization_mapping'].get('ACTION')}",
    f"- policy action shape: {report['checkpoint_summary']['policy_action_shape']}",
    f"- expert action shape: {report['offline_alignment_summary']['expert_action_shape']}",
    f"- offline alignment signal: {report['offline_alignment_summary']['offline_alignment_signal']}",
    f"- action L1 to expert: {report['offline_alignment_summary']['action_l1_to_expert']}",
    "",
    "## Issues",
    "",
]
for item in issues:
    lines.append(f"- {item['severity']} / {item['axis']}: {item['finding']}")
lines.extend(
    [
        "",
        "## Recommended Next Step",
        "",
        report["recommended_next_step"],
        "",
        "This audit is report-only diagnostic evidence. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
    ]
)
MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
