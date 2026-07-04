param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmolVLACkpt = "",
    [string]$LiberoRoot = "",
    [string]$LiberoDataRoot = "",
    [string]$InitStateSummaryPath = "reports\init_state_recheck_metric_summary_report.json",
    [string]$Hdf5AuditPath = "reports\libero_hdf5_interface_audit_report.json",
    [string]$OfflineAdapterReportPath = "reports\offline_adapter_reproduction_check_report.json",
    [string]$LoadOnlyReportPath = "reports\smolvla_load_only_smoke_report.json",
    [string]$JsonReportPath = "reports\smolvla_libero_checkpoint_task_alignment_audit_report.json",
    [string]$MarkdownReportPath = "reports\smolvla_libero_checkpoint_task_alignment_audit_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "SmolVLA/LIBERO checkpoint-task alignment audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads local configs, BDDL names, and existing reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
    "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK",
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

if ([string]::IsNullOrWhiteSpace($SmolVLACkpt)) { $SmolVLACkpt = $env:SMOLVLA_CKPT }
if ([string]::IsNullOrWhiteSpace($SmolVLACkpt)) { $SmolVLACkpt = "C:\assets\checkpoints\smolvla" }
if ([string]::IsNullOrWhiteSpace($LiberoRoot)) { $LiberoRoot = $env:LIBERO_ROOT }
if ([string]::IsNullOrWhiteSpace($LiberoRoot)) { $LiberoRoot = "C:\assets\repos\LIBERO" }
if ([string]::IsNullOrWhiteSpace($LiberoDataRoot)) { $LiberoDataRoot = $env:LIBERO_DATA_ROOT }
if ([string]::IsNullOrWhiteSpace($LiberoDataRoot)) { $LiberoDataRoot = "C:\assets\data\libero" }

$env:TCA_MAP_ALIGN_SMOLVLA = $SmolVLACkpt
$env:TCA_MAP_ALIGN_LIBERO = $LiberoRoot
$env:TCA_MAP_ALIGN_LIBERO_DATA = $LiberoDataRoot
$env:TCA_MAP_ALIGN_INIT_SUMMARY = Resolve-RepoPath -Path $InitStateSummaryPath
$env:TCA_MAP_ALIGN_HDF5_AUDIT = Resolve-RepoPath -Path $Hdf5AuditPath
$env:TCA_MAP_ALIGN_OFFLINE_ADAPTER = Resolve-RepoPath -Path $OfflineAdapterReportPath
$env:TCA_MAP_ALIGN_LOAD_ONLY = Resolve-RepoPath -Path $LoadOnlyReportPath
$env:TCA_MAP_ALIGN_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_ALIGN_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_ALIGN_GATES = ($setExecutionGates -join ";")

$script = @'
import glob
import json
import os
import re
from pathlib import Path

SMOLVLA = Path(os.environ["TCA_MAP_ALIGN_SMOLVLA"])
LIBERO = Path(os.environ["TCA_MAP_ALIGN_LIBERO"])
LIBERO_DATA = Path(os.environ["TCA_MAP_ALIGN_LIBERO_DATA"])
INIT_SUMMARY = Path(os.environ["TCA_MAP_ALIGN_INIT_SUMMARY"])
HDF5_AUDIT = Path(os.environ["TCA_MAP_ALIGN_HDF5_AUDIT"])
OFFLINE_ADAPTER = Path(os.environ["TCA_MAP_ALIGN_OFFLINE_ADAPTER"])
LOAD_ONLY = Path(os.environ["TCA_MAP_ALIGN_LOAD_ONLY"])
JSON_OUT = Path(os.environ["TCA_MAP_ALIGN_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_ALIGN_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_ALIGN_GATES", "").split(";") if item]


def read_json(path: Path, *, required: bool = True):
    if not path.exists():
        return None, f"Missing JSON file: {path}" if required else None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parsing issue.
        return None, f"Could not parse {path}: {exc}"


def read_bddl_language(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(:language"):
            value = stripped[len("(:language") :].strip()
            if value.endswith(")"):
                value = value[:-1].strip()
            return " ".join(value.split())
    return None


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"(_demo|\.hdf5|\.bddl)$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def shape_of(config: dict, key: str):
    feature = (config.get("input_features") or {}).get(key) or (config.get("output_features") or {}).get(key)
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
    stop_reasons.append("Refusing checkpoint-task alignment audit while execution gates are set: " + ", ".join(SET_GATES))

config, config_error = read_json(SMOLVLA / "config.json")
preprocessor, preprocessor_error = read_json(SMOLVLA / "policy_preprocessor.json")
init_summary, init_error = read_json(INIT_SUMMARY)
hdf5_audit, hdf5_error = read_json(HDF5_AUDIT)
offline_adapter, offline_error = read_json(OFFLINE_ADAPTER)
load_only, load_error = read_json(LOAD_ONLY, required=False)
for error in (config_error, preprocessor_error, init_error, hdf5_error, offline_error):
    if error:
        stop_reasons.append(error)
if load_error:
    stop_reasons.append(load_error)

config = config or {}
preprocessor = preprocessor or {}
init_summary = init_summary or {}
hdf5_audit = hdf5_audit or {}
offline_adapter = offline_adapter or {}
load_only = load_only or {}

tokenizer_name = None
preprocessor_features = {}
for step in preprocessor.get("steps", []):
    step_config = step.get("config") if isinstance(step, dict) else {}
    if isinstance(step_config, dict) and step_config.get("tokenizer_name"):
        tokenizer_name = step_config.get("tokenizer_name")
    if isinstance(step_config, dict) and isinstance(step_config.get("features"), dict):
        preprocessor_features.update(step_config.get("features"))

input_features = config.get("input_features") or {}
output_features = config.get("output_features") or {}
policy_image_shapes = {
    key: value.get("shape")
    for key, value in input_features.items()
    if isinstance(value, dict) and value.get("type") == "VISUAL"
}
preprocessor_image_keys = sorted(
    key
    for key, value in preprocessor_features.items()
    if isinstance(value, dict) and value.get("type") == "VISUAL"
)
policy_state_shape = shape_of(config, "observation.state")
policy_action_shape = shape_of(config, "action")

scenarios = (init_summary.get("metric_summary") or {}).get("scenarios") or []
selected_task_name = None
for scenario in scenarios:
    if scenario.get("task_name"):
        selected_task_name = scenario.get("task_name")
        break
if not selected_task_name:
    hdf5_path = ((offline_adapter.get("paths") or {}).get("hdf5_path") or (hdf5_audit.get("paths") or {}).get("hdf5_path"))
    selected_task_name = Path(hdf5_path).name.replace("_demo.hdf5", "") if hdf5_path else None

bddl_root = LIBERO / "libero" / "libero" / "bddl_files" / "libero_10"
bddl_files = sorted(Path(p) for p in glob.glob(str(bddl_root / "*.bddl")))
selected_bddl = None
if selected_task_name:
    expected = bddl_root / f"{selected_task_name}.bddl"
    if expected.exists():
        selected_bddl = expected
if selected_bddl is None and selected_task_name:
    selected_norm = normalize_name(selected_task_name)
    for candidate in bddl_files:
        if normalize_name(candidate.name) == selected_norm:
            selected_bddl = candidate
            break
if selected_bddl is None and bddl_files:
    selected_bddl = bddl_files[0]
selected_language = read_bddl_language(selected_bddl) if selected_bddl else None
task_matches_bddl = bool(
    selected_task_name
    and selected_bddl
    and normalize_name(selected_task_name) == normalize_name(selected_bddl.name)
)

config_provenance_fields = {
    key: value
    for key, value in config.items()
    if any(fragment in key.lower() for fragment in ("repo", "dataset", "task", "suite", "env", "libero"))
}
explicit_libero_provenance = any(
    "libero" in str(value).lower()
    for value in config_provenance_fields.values()
)
repo_id = config.get("repo_id")
config_records_task_source = bool(explicit_libero_provenance or repo_id)

load_only_policy = load_only.get("load") or {}
load_only_load_vlm_weights = load_only_policy.get("load_vlm_weights")
config_load_vlm_weights = config.get("load_vlm_weights")

hdf5_summary = hdf5_audit.get("hdf5_summary") or {}
hdf5_action_dim = hdf5_summary.get("action_dim")
hdf5_image_shapes = ((hdf5_audit.get("hdf5_summary") or {}).get("obs_shapes") or {})
offline_reproduction = offline_adapter.get("reproduction") or {}
best_gripper_strategy = offline_reproduction.get("best_action_adapter_strategy_for_first_demo_action")
positive_signal = bool((init_summary.get("metric_summary") or {}).get("positive_diagnostic_signal_found"))
hdf5_init_state_set = bool((init_summary.get("metric_summary") or {}).get("hdf5_init_state_set_in_environment"))

issues: list[dict] = []
if not config_records_task_source:
    issues.append(
        issue(
            "checkpoint_provenance",
            "high",
            "The local SmolVLA checkpoint config does not record a confirmed LIBERO dataset/task-suite provenance.",
            {
                "repo_id": repo_id,
                "config_provenance_fields": config_provenance_fields,
                "explicit_libero_provenance": explicit_libero_provenance,
            },
            "Do not treat zero-reward LIBERO rollouts as method evidence until checkpoint/task provenance is established or an offline action-decoding check is performed.",
        )
    )
else:
    issues.append(
        issue(
            "checkpoint_provenance",
            "medium",
            "The config contains provenance-like fields, but a direct match to the selected LIBERO diagnostic task still needs verification.",
            {
                "repo_id": repo_id,
                "config_provenance_fields": config_provenance_fields,
                "explicit_libero_provenance": explicit_libero_provenance,
            },
            "Use the provenance fields only as a routing clue until task-level behavior is verified.",
        )
    )

task_severity = "low" if task_matches_bddl else "medium"
issues.append(
    issue(
        "selected_task_alignment",
        task_severity,
        "The selected rollout/HDF5 task can be matched to a local LIBERO BDDL file." if task_matches_bddl else "The selected task could not be cleanly matched to a local LIBERO BDDL filename.",
        {
            "selected_task_name": selected_task_name,
            "selected_bddl_file": selected_bddl.name if selected_bddl else None,
            "selected_bddl_language": selected_language,
            "task_matches_bddl_filename": task_matches_bddl,
            "libero_10_bddl_count": len(bddl_files),
        },
        "Keep using exact BDDL language for diagnostics, but do not infer checkpoint competence from filename alignment alone.",
    )
)

vlm_severity = "high" if config_load_vlm_weights and load_only_load_vlm_weights is False else "medium"
issues.append(
    issue(
        "vlm_loading_policy",
        vlm_severity,
        "The checkpoint config expects VLM weights, while the bounded local diagnostics loaded with VLM weights disabled." if vlm_severity == "high" else "The VLM loading policy remains a compatibility variable for local diagnostics.",
        {
            "config_load_vlm_weights": config_load_vlm_weights,
            "local_load_only_load_vlm_weights": load_only_load_vlm_weights,
            "vlm_model_name": config.get("vlm_model_name"),
            "preprocessor_tokenizer_name": tokenizer_name,
        },
        "Before more learned-policy rollouts, plan either an offline action-decoding check that records the load policy or a one-task VLM-enabled risk assessment if memory budget permits.",
    )
)

if policy_action_shape and hdf5_action_dim and hdf5_action_dim != policy_action_shape[-1]:
    issues.append(
        issue(
            "action_decoding_convention",
            "high",
            "The policy action shape is 6D while local LIBERO demonstrations and environments use 7D actions.",
            {
                "policy_action_shape": policy_action_shape,
                "hdf5_action_dim": hdf5_action_dim,
                "best_gripper_strategy_for_first_demo_action": best_gripper_strategy,
            },
            "Treat the next step as offline demonstration-conditioned action decoding or adapter-target comparison, not rollout scaling.",
        )
    )

if len(policy_image_shapes) != len([key for key in hdf5_image_shapes if key.endswith("_rgb")]) or preprocessor_image_keys:
    issues.append(
        issue(
            "observation_camera_convention",
            "medium",
            "The policy/preprocessor image keys do not directly mirror the local HDF5 camera streams.",
            {
                "policy_image_shapes": policy_image_shapes,
                "preprocessor_image_keys": preprocessor_image_keys,
                "hdf5_image_keys": sorted(key for key in hdf5_image_shapes if key.endswith("_rgb")),
            },
            "Keep camera aliasing labeled and prefer offline demonstration-conditioned decoding before larger rollouts.",
        )
    )

issues.append(
    issue(
        "init_state_recheck_result",
        "high",
        "Reset-only and HDF5-init-state learned-policy diagnostics all produced zero reward and no diagnostic success.",
        {
            "positive_diagnostic_signal_found": positive_signal,
            "hdf5_init_state_set_in_environment": hdf5_init_state_set,
            "init_state_vs_reset_reward_delta": (init_summary.get("metric_summary") or {}).get("init_state_vs_reset_3_step_reward_delta"),
            "init_summary_decision": init_summary.get("decision"),
        },
        "Keep rollout scaling blocked until a non-rollout or one-task compatibility check explains the zero signal.",
    )
)

issues.append(
    issue(
        "offline_demonstration_conditioned_action_decoding",
        "medium",
        "The next informative check is whether local SmolVLA action decoding from a real LIBERO demonstration observation is closer to expert action than rollout diagnostics suggest.",
        {
            "offline_adapter_best_strategy": best_gripper_strategy,
            "hdf5_action_dim": hdf5_action_dim,
            "policy_state_shape": policy_state_shape,
            "policy_action_shape": policy_action_shape,
        },
        "Plan a separately gated one-sample offline action-decoding check. It may load SmolVLA and run one local inference only after a green risk assessment; it must not create a simulator environment or rollout.",
    )
)

high_count = sum(1 for item in issues if item["severity"] == "high")
passed = not stop_reasons
decision = "no_go_rollout_scaling" if passed else "stop"
ready_for_offline_decoding_plan = bool(passed)

report = {
    "evidence_label": "smolvla_libero_checkpoint_task_alignment_audit",
    "smolvla_libero_checkpoint_task_alignment_audit_passed": passed,
    "decision": decision,
    "ready_for_rollout_scaling": False,
    "ready_for_benchmark_claim": False,
    "ready_for_paper_claim": False,
    "ready_for_offline_demonstration_conditioned_action_decoding_plan": ready_for_offline_decoding_plan,
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
        "libero_root": str(LIBERO),
        "libero_data_root": str(LIBERO_DATA),
        "init_state_summary": str(INIT_SUMMARY),
        "hdf5_audit": str(HDF5_AUDIT),
        "offline_adapter_report": str(OFFLINE_ADAPTER),
        "load_only_report": str(LOAD_ONLY),
    },
    "checkpoint_summary": {
        "type": config.get("type"),
        "repo_id": repo_id,
        "license": config.get("license"),
        "config_load_vlm_weights": config_load_vlm_weights,
        "local_load_only_load_vlm_weights": load_only_load_vlm_weights,
        "vlm_model_name": config.get("vlm_model_name"),
        "preprocessor_tokenizer_name": tokenizer_name,
        "config_provenance_fields": config_provenance_fields,
        "state_shape": policy_state_shape,
        "action_shape": policy_action_shape,
        "image_shapes": policy_image_shapes,
        "preprocessor_image_keys": preprocessor_image_keys,
        "normalization_mapping": config.get("normalization_mapping"),
    },
    "task_summary": {
        "selected_task_name": selected_task_name,
        "selected_bddl_file": selected_bddl.name if selected_bddl else None,
        "selected_bddl_language": selected_language,
        "task_matches_bddl_filename": task_matches_bddl,
        "libero_10_bddl_count": len(bddl_files),
        "hdf5_path": (offline_adapter.get("paths") or {}).get("hdf5_path") or (hdf5_audit.get("paths") or {}).get("hdf5_path"),
    },
    "evidence_summary": {
        "init_summary_decision": init_summary.get("decision"),
        "positive_diagnostic_signal_found": positive_signal,
        "hdf5_init_state_set_in_environment": hdf5_init_state_set,
        "hdf5_action_dim": hdf5_action_dim,
        "best_gripper_strategy_for_first_demo_action": best_gripper_strategy,
        "hdf5_audit_decision": hdf5_audit.get("decision"),
        "offline_adapter_decision": offline_adapter.get("decision"),
    },
    "issues": issues,
    "stop_reasons": stop_reasons,
    "recommended_next_step": (
        "Create a planning-only offline demonstration-conditioned action decoding gate: one HDF5 observation, one expert action target, no simulator environment, no rollout, no training, and no paper claim. Only a later separately gated runner may load SmolVLA for one CPU inference if the risk assessment is green."
        if passed
        else "Resolve missing audit inputs before continuing."
    ),
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# SmolVLA/LIBERO Checkpoint-Task Alignment Audit Report",
    "",
    f"- decision: {report['decision']}",
    f"- audit passed: {report['smolvla_libero_checkpoint_task_alignment_audit_passed']}",
    f"- high severity issues: {report['high_severity_issue_count']}",
    f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
    f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
    f"- offline decoding plan ready: {report['ready_for_offline_demonstration_conditioned_action_decoding_plan']}",
    "",
    "## Key Evidence",
    "",
    f"- selected task: {report['task_summary']['selected_task_name']}",
    f"- BDDL language: {report['task_summary']['selected_bddl_language']}",
    f"- checkpoint repo id: {report['checkpoint_summary']['repo_id']}",
    f"- config load_vlm_weights: {report['checkpoint_summary']['config_load_vlm_weights']}",
    f"- local load-only load_vlm_weights: {report['checkpoint_summary']['local_load_only_load_vlm_weights']}",
    f"- policy action shape: {report['checkpoint_summary']['action_shape']}",
    f"- HDF5 action dim: {report['evidence_summary']['hdf5_action_dim']}",
    f"- positive diagnostic signal found: {report['evidence_summary']['positive_diagnostic_signal_found']}",
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
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
