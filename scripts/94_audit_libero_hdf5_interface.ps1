param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmolVLACkpt = "",
    [string]$LiberoRoot = "",
    [string]$LiberoDataRoot = "",
    [string]$Hdf5Path = "",
    [string]$CompatibilityAuditPath = "reports\environment_policy_compatibility_audit_report.json",
    [string]$JsonReportPath = "reports\libero_hdf5_interface_audit_report.json",
    [string]$MarkdownReportPath = "reports\libero_hdf5_interface_audit_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO HDF5 interface audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads one local HDF5 demonstration, local config, and existing reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$env:TCA_MAP_HDF5_SMOLVLA = $SmolVLACkpt
$env:TCA_MAP_HDF5_LIBERO = $LiberoRoot
$env:TCA_MAP_HDF5_LIBERO_DATA = $LiberoDataRoot
$env:TCA_MAP_HDF5_PATH = $Hdf5Path
$env:TCA_MAP_HDF5_COMPAT = Resolve-RepoPath -Path $CompatibilityAuditPath
$env:TCA_MAP_HDF5_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_HDF5_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_HDF5_GATES = ($setExecutionGates -join ";")

$script = @'
import glob
import json
import os
from pathlib import Path

import numpy as np

SMOLVLA = Path(os.environ["TCA_MAP_HDF5_SMOLVLA"])
LIBERO = Path(os.environ["TCA_MAP_HDF5_LIBERO"])
LIBERO_DATA = Path(os.environ["TCA_MAP_HDF5_LIBERO_DATA"])
HDF5_PATH_RAW = os.environ.get("TCA_MAP_HDF5_PATH", "")
COMPAT = Path(os.environ["TCA_MAP_HDF5_COMPAT"])
JSON_OUT = Path(os.environ["TCA_MAP_HDF5_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_HDF5_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_HDF5_GATES", "").split(";") if item]


def read_json(path: Path):
    if not path.exists():
        return None, f"Missing JSON file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parsing issue.
        return None, f"Could not parse {path}: {exc}"


def shape_of(config: dict, key: str):
    feature = (config.get("input_features") or {}).get(key) or (config.get("output_features") or {}).get(key)
    if isinstance(feature, dict):
        return feature.get("shape")
    return None


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


def stats(array) -> dict:
    data = np.asarray(array)
    if data.size == 0:
        return {"shape": list(data.shape), "dtype": str(data.dtype), "min": None, "max": None, "mean": None, "std": None}
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
    }


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
    stop_reasons.append("Refusing HDF5 interface audit while execution gates are set: " + ", ".join(SET_GATES))

try:
    import h5py
except Exception as exc:  # noqa: BLE001 - exact local dependency issue.
    h5py = None
    stop_reasons.append(f"h5py is required for report-only HDF5 inspection: {exc}")

config, config_error = read_json(SMOLVLA / "config.json")
compat, compat_error = read_json(COMPAT)
if config_error:
    stop_reasons.append(config_error)
if compat_error:
    stop_reasons.append(compat_error)
config = config or {}
compat = compat or {}

hdf5_path = Path(HDF5_PATH_RAW) if HDF5_PATH_RAW else None
if hdf5_path is None or not str(hdf5_path):
    candidates = sorted(Path(p) for p in glob.glob(str(LIBERO_DATA / "libero_10" / "*.hdf5")))
    hdf5_path = candidates[0] if candidates else None
if hdf5_path is None or not hdf5_path.exists():
    stop_reasons.append(f"Missing HDF5 demonstration file under {LIBERO_DATA / 'libero_10'}")

policy_state_shape = shape_of(config, "observation.state")
policy_action_shape = shape_of(config, "action")
policy_image_shapes = {
    key: value.get("shape")
    for key, value in (config.get("input_features") or {}).items()
    if isinstance(value, dict) and value.get("type") == "VISUAL"
}

hdf5_summary = {}
issues = []
if not stop_reasons and h5py is not None and hdf5_path is not None:
    with h5py.File(hdf5_path, "r") as handle:
        demo_names = sorted(handle.get("data", {}).keys())
        first_demo = demo_names[0] if demo_names else None
        if first_demo is None:
            stop_reasons.append("HDF5 file has no data/demo_* groups.")
        else:
            demo = handle["data"][first_demo]
            obs_group = demo.get("obs")
            obs_keys = sorted(obs_group.keys()) if obs_group is not None else []
            actions = demo.get("actions")
            rewards = demo.get("rewards")
            dones = demo.get("dones")
            action_shape = list(actions.shape) if actions is not None else None
            action_dim = action_shape[1] if action_shape and len(action_shape) > 1 else None
            action_sample = actions[: min(64, actions.shape[0])] if actions is not None else np.asarray([])
            gripper_stats = stats(action_sample[:, -1]) if action_sample.ndim == 2 and action_sample.shape[1] else {}
            obs_shapes = {
                key: {
                    "shape": list(obs_group[key].shape),
                    "dtype": str(obs_group[key].dtype),
                }
                for key in obs_keys
                if hasattr(obs_group[key], "shape")
            }
            hdf5_summary = {
                "path": str(hdf5_path),
                "demo_count": len(demo_names),
                "first_demo": first_demo,
                "action_stats_first_64": stats(action_sample),
                "action_dim": action_dim,
                "gripper_last_column_stats_first_64": gripper_stats,
                "reward_stats": stats(rewards[:]) if rewards is not None else None,
                "done_stats": stats(dones[:]) if dones is not None else None,
                "obs_keys": obs_keys,
                "obs_shapes": obs_shapes,
            }
            if policy_action_shape and action_dim != policy_action_shape[-1]:
                issues.append(
                    issue(
                        "action_dimension",
                        "high",
                        "LIBERO demonstration actions are 7D while the SmolVLA policy config action is 6D.",
                        {"hdf5_action_dim": action_dim, "policy_action_shape": policy_action_shape, "gripper_stats": gripper_stats},
                        "Do not scale rollout until the 6D-to-7D action adapter is validated against demonstration action semantics.",
                    )
                )
            ee_states = obs_shapes.get("ee_states", {}).get("shape")
            if policy_state_shape and ee_states and ee_states[-1] == policy_state_shape[-1]:
                state_severity = "low"
                state_finding = "LIBERO obs/ee_states is 6D and matches the SmolVLA policy state dimension."
            else:
                state_severity = "medium"
                state_finding = "Could not confirm a direct 6D ee_states match for the SmolVLA policy state."
            issues.append(
                issue(
                    "state_dimension",
                    state_severity,
                    state_finding,
                    {"policy_state_shape": policy_state_shape, "hdf5_ee_states_shape": ee_states},
                    "Prefer using obs/ee_states for offline adapter checks before additional rollout variants.",
                )
            )
            image_keys = [key for key in obs_keys if key.endswith("_rgb")]
            image_evidence = {"policy_image_shapes": policy_image_shapes, "hdf5_image_shapes": {key: obs_shapes[key]["shape"] for key in image_keys}}
            if len(image_keys) < len(policy_image_shapes):
                issues.append(
                    issue(
                        "camera_count",
                        "medium",
                        "The HDF5 demonstration exposes fewer RGB camera streams than the SmolVLA policy config image inputs.",
                        image_evidence,
                        "Keep camera aliasing explicit and verify whether duplicating agentview for camera3 is a smoke-only approximation.",
                    )
                )
            if any(obs_shapes[key]["shape"][1:3] != [256, 256] for key in image_keys):
                issues.append(
                    issue(
                        "camera_resolution",
                        "low",
                        "HDF5 images are not 256x256, so preprocessing/resizing is required before policy input.",
                        image_evidence,
                        "Treat resizing as expected preprocessing, but record it in offline adapter reports.",
                    )
                )

bddl_files = sorted(Path(p) for p in glob.glob(str(LIBERO / "libero" / "libero" / "bddl_files" / "libero_10" / "*.bddl")))
first_language = read_bddl_language(bddl_files[0]) if bddl_files else None
issues.append(
    issue(
        "task_language",
        "medium",
        "The HDF5 filename and BDDL language can be matched locally, but checkpoint training/task provenance is still not established.",
        {
            "hdf5_file": hdf5_path.name if hdf5_path else None,
            "first_bddl_file": bddl_files[0].name if bddl_files else None,
            "first_bddl_language": first_language,
        },
        "Add a provenance check before assuming this SmolVLA checkpoint should solve the selected LIBERO_10 task.",
    )
)
issues.append(
    issue(
        "compatibility_audit_dependency",
        "high",
        "The previous compatibility audit already blocks rollout scaling.",
        {
            "compat_decision": compat.get("decision"),
            "compat_high_severity_issue_count": compat.get("high_severity_issue_count"),
            "compat_ready_for_rollout_scaling": compat.get("ready_for_rollout_scaling"),
        },
        "Use this HDF5 audit to design offline adapter checks, not to justify rollout scaling.",
    )
)

high_count = sum(1 for item in issues if item["severity"] == "high")
passed = not stop_reasons
decision = "no_go_rollout_scaling" if passed else "stop"
recommended_next_step = (
    "Create a report-only offline adapter reproduction check that builds the SmolVLA-compatible state/image/action adapter inputs from the first HDF5 timestep and compares dimensions/ranges, without model loading or rollout."
    if passed
    else "Fix missing HDF5 audit inputs before continuing."
)

report = {
    "evidence_label": "libero_hdf5_interface_audit",
    "libero_hdf5_interface_audit_passed": passed,
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
        "hdf5_path": str(hdf5_path) if hdf5_path else None,
        "compatibility_audit": str(COMPAT),
    },
    "policy_config": {
        "state_shape": policy_state_shape,
        "action_shape": policy_action_shape,
        "image_shapes": policy_image_shapes,
    },
    "hdf5_summary": hdf5_summary,
    "issues": issues,
    "stop_reasons": stop_reasons,
    "recommended_next_step": recommended_next_step,
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# LIBERO HDF5 Interface Audit Report",
    "",
    f"- decision: {report['decision']}",
    f"- audit passed: {report['libero_hdf5_interface_audit_passed']}",
    f"- high severity issues: {report['high_severity_issue_count']}",
    f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
    f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
    f"- HDF5 file: {report['paths']['hdf5_path']}",
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
        "This audit is report-only offline dataset evidence. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
    ]
)
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
