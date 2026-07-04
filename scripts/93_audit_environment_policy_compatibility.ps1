param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmolVLACkpt = "",
    [string]$LiberoRoot = "",
    [string]$LiberoDataRoot = "",
    [string]$RobosuiteRoot = "",
    [string]$SynthesisReportPath = "reports\learned_policy_diagnostic_synthesis_report.json",
    [string]$RolloutBridgeSourcePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$PolicyLoaderSourcePath = "tca_map\smolvla\single_sample_interface_smoke.py",
    [string]$JsonReportPath = "reports\environment_policy_compatibility_audit_report.json",
    [string]$MarkdownReportPath = "reports\environment_policy_compatibility_audit_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Environment-policy compatibility audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads local configs, source files, and existing reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
if ([string]::IsNullOrWhiteSpace($RobosuiteRoot)) { $RobosuiteRoot = $env:ROBOSUITE_ROOT }
if ([string]::IsNullOrWhiteSpace($RobosuiteRoot)) { $RobosuiteRoot = "C:\assets\repos\robosuite" }

$env:TCA_MAP_COMPAT_SMOLVLA = $SmolVLACkpt
$env:TCA_MAP_COMPAT_LIBERO = $LiberoRoot
$env:TCA_MAP_COMPAT_LIBERO_DATA = $LiberoDataRoot
$env:TCA_MAP_COMPAT_ROBOSUITE = $RobosuiteRoot
$env:TCA_MAP_COMPAT_SYNTHESIS = Resolve-RepoPath -Path $SynthesisReportPath
$env:TCA_MAP_COMPAT_SOURCE = Resolve-RepoPath -Path $RolloutBridgeSourcePath
$env:TCA_MAP_COMPAT_POLICY_LOADER_SOURCE = Resolve-RepoPath -Path $PolicyLoaderSourcePath
$env:TCA_MAP_COMPAT_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_COMPAT_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_COMPAT_GATES = ($setExecutionGates -join ";")

$script = @'
import glob
import json
import os
from pathlib import Path

SMOLVLA = Path(os.environ["TCA_MAP_COMPAT_SMOLVLA"])
LIBERO = Path(os.environ["TCA_MAP_COMPAT_LIBERO"])
LIBERO_DATA = Path(os.environ["TCA_MAP_COMPAT_LIBERO_DATA"])
ROBOSUITE = Path(os.environ["TCA_MAP_COMPAT_ROBOSUITE"])
SYNTHESIS = Path(os.environ["TCA_MAP_COMPAT_SYNTHESIS"])
SOURCE = Path(os.environ["TCA_MAP_COMPAT_SOURCE"])
POLICY_LOADER_SOURCE = Path(os.environ["TCA_MAP_COMPAT_POLICY_LOADER_SOURCE"])
JSON_OUT = Path(os.environ["TCA_MAP_COMPAT_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_COMPAT_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_COMPAT_GATES", "").split(";") if item]


def read_json(path: Path):
    if not path.exists():
        return None, f"Missing JSON file: {path}"
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


stop_reasons = []
if SET_GATES:
    stop_reasons.append("Refusing compatibility audit while execution gates are set: " + ", ".join(SET_GATES))

config, config_error = read_json(SMOLVLA / "config.json")
synthesis, synthesis_error = read_json(SYNTHESIS)
source_text = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""
policy_loader_source_text = POLICY_LOADER_SOURCE.read_text(encoding="utf-8", errors="replace") if POLICY_LOADER_SOURCE.exists() else ""
combined_source_text = source_text + "\n" + policy_loader_source_text
if config_error:
    stop_reasons.append(config_error)
if synthesis_error:
    stop_reasons.append(synthesis_error)
if not SOURCE.exists():
    stop_reasons.append(f"Missing rollout bridge source: {SOURCE}")
if not POLICY_LOADER_SOURCE.exists():
    stop_reasons.append(f"Missing policy loader source: {POLICY_LOADER_SOURCE}")

bddl_files = sorted(Path(p) for p in glob.glob(str(LIBERO / "libero" / "libero" / "bddl_files" / "libero_10" / "*.bddl")))
hdf5_files = sorted(Path(p) for p in glob.glob(str(LIBERO_DATA / "libero_10" / "*.hdf5")))
first_bddl = bddl_files[0] if bddl_files else None
first_language = read_bddl_language(first_bddl) if first_bddl else None

config = config or {}
synthesis = synthesis or {}
input_features = config.get("input_features") or {}
output_features = config.get("output_features") or {}
image_shapes = {key: value.get("shape") for key, value in input_features.items() if isinstance(value, dict) and value.get("type") == "VISUAL"}
state_shape = shape_of(config, "observation.state")
action_shape = shape_of(config, "action")

issues = []
issues.append(
    issue(
        "task_checkpoint_alignment",
        "high",
        "The local SmolVLA config does not record LIBERO task-suite provenance or a confirmed match to the selected LIBERO_10 diagnostic task.",
        {
            "smolvla_repo_id": config.get("repo_id"),
            "config_license": config.get("license"),
            "first_bddl_task": first_bddl.name if first_bddl else None,
            "first_bddl_language": first_language,
            "libero_10_bddl_count": len(bddl_files),
            "libero_10_hdf5_count": len(hdf5_files),
        },
        "Before another learned-policy diagnostic, add a report-only checkpoint/task provenance audit or choose a task explicitly documented for the checkpoint.",
    )
)
issues.append(
    issue(
        "vlm_loading_policy",
        "high",
        "The bounded local learned-policy path keeps load_vlm_weights=false for memory-safe diagnostics, which is valid for smoke but may remove or weaken visual-language grounding needed for task success.",
        {
            "config_vlm_model_name": config.get("vlm_model_name"),
            "policy_loader_sets_load_vlm_weights_false": "load_vlm_weights = False" in combined_source_text,
            "synthesis_decision": synthesis.get("decision"),
        },
        "Do not scale rollouts from load_vlm_weights=false evidence. If a load-with-VLM experiment is considered later, require a separate memory/runtime risk assessment and keep it one-task bounded.",
    )
)
issues.append(
    issue(
        "action_convention",
        "high",
        "The policy emits a 6D action while LIBERO/RoboSuite environments report a 7D action interface, so the bridge relies on a diagnostic gripper adapter whose semantics are not proven correct.",
        {
            "policy_action_shape": action_shape,
            "config_max_action_dim": config.get("max_action_dim"),
            "source_uses_action_adapter": "adapt_policy_action_to_env_action" in source_text,
            "source_reads_env_action_dim": "action_dim" in source_text,
        },
        "Keep rollout scaling blocked until action convention is checked against local LIBERO demonstrations or documented policy deployment code.",
    )
)
issues.append(
    issue(
        "observation_convention",
        "medium",
        "The policy expects a 6D state and three 256x256 image inputs; the diagnostic bridge adapts LIBERO observations into that contract, but the correct camera/state convention remains unproven.",
        {
            "state_shape": state_shape,
            "image_shapes": image_shapes,
            "source_has_camera_aliases": "DEFAULT_IMAGE_ALIASES" in source_text and "_camera_aliases" in source_text,
            "source_has_state_strategies": "STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_LAST3" in source_text,
        },
        "Prefer an offline demonstration observation audit before another rollout variant: compare HDF5 keys, camera shapes, state keys, and action dimensions with the policy config.",
    )
)
issues.append(
    issue(
        "diagnostic_ladder_result",
        "high",
        "The diagnostic ladder is complete but produced no positive reward or success signal across adapter strategy, action scale, prompt format, camera source, or state sufficiency.",
        {
            "diagnostic_ladder_complete": synthesis.get("diagnostic_ladder_complete"),
            "positive_diagnostic_signal_found": synthesis.get("positive_diagnostic_signal_found"),
            "ready_for_rollout_scaling": synthesis.get("ready_for_rollout_scaling"),
            "no_go_reason": synthesis.get("no_go_for_rollout_scaling_reason"),
        },
        "Treat current learned-policy rollout scaling as no-go. Continue only with report-only compatibility audits or one-task fixes tied to a specific hypothesis.",
    )
)

high_count = sum(1 for item in issues if item["severity"] == "high")
passed = not stop_reasons
decision = "no_go_rollout_scaling" if passed else "stop"
recommended_next_step = (
    "Create a bounded offline demonstration interface audit: inspect one LIBERO HDF5 file for action dimensions, action ranges, observation keys, camera shapes, and language/task alignment without model loading or simulator rollout."
    if passed
    else "Fix missing compatibility-audit inputs before continuing."
)

report = {
    "evidence_label": "environment_policy_compatibility_audit",
    "environment_policy_compatibility_audit_passed": passed,
    "decision": decision,
    "ready_for_rollout_scaling": False,
    "ready_for_paper_claim": False,
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
        "robosuite_root": str(ROBOSUITE),
        "synthesis_report": str(SYNTHESIS),
        "rollout_bridge_source": str(SOURCE),
        "policy_loader_source": str(POLICY_LOADER_SOURCE),
    },
    "config_summary": {
        "type": config.get("type"),
        "repo_id": config.get("repo_id"),
        "license": config.get("license"),
        "vlm_model_name": config.get("vlm_model_name"),
        "input_feature_keys": sorted(input_features.keys()),
        "output_feature_keys": sorted(output_features.keys()),
        "state_shape": state_shape,
        "action_shape": action_shape,
        "image_shapes": image_shapes,
        "chunk_size": config.get("chunk_size"),
        "n_action_steps": config.get("n_action_steps"),
        "tokenizer_max_length": config.get("tokenizer_max_length"),
        "max_action_dim": config.get("max_action_dim"),
    },
    "libero_summary": {
        "libero_10_bddl_count": len(bddl_files),
        "libero_10_hdf5_count": len(hdf5_files),
        "first_bddl_task": first_bddl.name if first_bddl else None,
        "first_bddl_language": first_language,
    },
    "issues": issues,
    "stop_reasons": stop_reasons,
    "recommended_next_step": recommended_next_step,
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Environment-Policy Compatibility Audit Report",
    "",
    f"- decision: {report['decision']}",
    f"- audit passed: {report['environment_policy_compatibility_audit_passed']}",
    f"- high severity issues: {report['high_severity_issue_count']}",
    f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
    f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
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
