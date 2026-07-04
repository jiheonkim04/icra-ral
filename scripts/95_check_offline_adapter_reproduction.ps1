param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmolVLACkpt = "",
    [string]$LiberoDataRoot = "",
    [string]$Hdf5Path = "",
    [string]$Hdf5AuditPath = "reports\libero_hdf5_interface_audit_report.json",
    [string]$JsonReportPath = "reports\offline_adapter_reproduction_check_report.json",
    [string]$MarkdownReportPath = "reports\offline_adapter_reproduction_check_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Offline adapter reproduction check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads one local HDF5 demonstration, local config, pure adapter helpers, and existing reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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
if ([string]::IsNullOrWhiteSpace($LiberoDataRoot)) { $LiberoDataRoot = $env:LIBERO_DATA_ROOT }
if ([string]::IsNullOrWhiteSpace($LiberoDataRoot)) { $LiberoDataRoot = "C:\assets\data\libero" }

$env:TCA_MAP_REPRO_SMOLVLA = $SmolVLACkpt
$env:TCA_MAP_REPRO_LIBERO_DATA = $LiberoDataRoot
$env:TCA_MAP_REPRO_HDF5 = $Hdf5Path
$env:TCA_MAP_REPRO_HDF5_AUDIT = Resolve-RepoPath -Path $Hdf5AuditPath
$env:TCA_MAP_REPRO_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_REPRO_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_REPRO_GATES = ($setExecutionGates -join ";")

$script = @'
import glob
import json
import os
from pathlib import Path

import numpy as np

from tca_map.smolvla.interface_adapters import (
    ACTION_STRATEGY_GRIPPER_CLOSE,
    ACTION_STRATEGY_GRIPPER_OPEN,
    ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    adapt_policy_action_to_env_action,
)

SMOLVLA = Path(os.environ["TCA_MAP_REPRO_SMOLVLA"])
LIBERO_DATA = Path(os.environ["TCA_MAP_REPRO_LIBERO_DATA"])
HDF5_PATH_RAW = os.environ.get("TCA_MAP_REPRO_HDF5", "")
HDF5_AUDIT = Path(os.environ["TCA_MAP_REPRO_HDF5_AUDIT"])
JSON_OUT = Path(os.environ["TCA_MAP_REPRO_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_REPRO_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_REPRO_GATES", "").split(";") if item]


def read_json(path: Path):
    if not path.exists():
        return None, f"Missing JSON file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact local parsing issue.
        return None, f"Could not parse {path}: {exc}"


def stats(values) -> dict:
    data = np.asarray(values)
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
    stop_reasons.append("Refusing offline adapter reproduction while execution gates are set: " + ", ".join(SET_GATES))

try:
    import h5py
except Exception as exc:  # noqa: BLE001
    h5py = None
    stop_reasons.append(f"h5py is required for report-only HDF5 inspection: {exc}")

config, config_error = read_json(SMOLVLA / "config.json")
hdf5_audit, hdf5_audit_error = read_json(HDF5_AUDIT)
if config_error:
    stop_reasons.append(config_error)
if hdf5_audit_error:
    stop_reasons.append(hdf5_audit_error)
config = config or {}
hdf5_audit = hdf5_audit or {}

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

reproduction = {}
issues = []
if not stop_reasons and h5py is not None and hdf5_path is not None:
    with h5py.File(hdf5_path, "r") as handle:
        demo_name = sorted(handle["data"].keys())[0]
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"][: min(64, demo["actions"].shape[0])], dtype=np.float32)
        first_action = actions[0]
        policy_sized_action = first_action[: int(policy_action_shape[-1] if policy_action_shape else 6)]
        env_action_dim = int(first_action.shape[0])
        strategies = [
            ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
            ACTION_STRATEGY_GRIPPER_OPEN,
            ACTION_STRATEGY_GRIPPER_CLOSE,
        ]
        action_reproductions = {}
        for strategy in strategies:
            adapted = adapt_policy_action_to_env_action(policy_sized_action, env_action_dim, strategy=strategy).values
            action_reproductions[strategy] = {
                "adapted_action": [float(x) for x in adapted],
                "target_demo_action": [float(x) for x in first_action],
                "l1_to_demo_first_action": float(np.mean(np.abs(np.asarray(adapted, dtype=np.float32) - first_action))),
                "gripper_matches_first_action": bool(np.isclose(adapted[-1], first_action[-1])),
                "metadata": adapt_policy_action_to_env_action(policy_sized_action, env_action_dim, strategy=strategy).metadata,
            }
        best_strategy = min(action_reproductions, key=lambda key: action_reproductions[key]["l1_to_demo_first_action"])

        obs = demo["obs"]
        ee_states = np.asarray(obs["ee_states"][0], dtype=np.float32) if "ee_states" in obs else None
        ee_pos = np.asarray(obs["ee_pos"][0], dtype=np.float32) if "ee_pos" in obs else None
        ee_ori = np.asarray(obs["ee_ori"][0], dtype=np.float32) if "ee_ori" in obs else None
        rebuilt_state = np.concatenate([ee_pos, ee_ori]).astype(np.float32) if ee_pos is not None and ee_ori is not None else None
        state_reproduction = {
            "policy_state_shape": policy_state_shape,
            "hdf5_ee_states": [float(x) for x in ee_states] if ee_states is not None else None,
            "rebuilt_from_ee_pos_ori": [float(x) for x in rebuilt_state] if rebuilt_state is not None else None,
            "rebuilt_matches_ee_states": bool(np.allclose(rebuilt_state, ee_states)) if rebuilt_state is not None and ee_states is not None else False,
        }

        image_keys = [key for key in obs.keys() if key.endswith("_rgb")]
        image_reproduction = {}
        for key in image_keys:
            dataset = obs[key]
            image_reproduction[key] = {
                "hdf5_shape": list(dataset.shape[1:]),
                "dtype": str(dataset.dtype),
                "needs_resize_to_policy": list(dataset.shape[1:3]) != [256, 256],
            }

        reproduction = {
            "hdf5_path": str(hdf5_path),
            "demo_name": demo_name,
            "first_action": [float(x) for x in first_action],
            "first_64_action_stats": stats(actions),
            "first_64_gripper_stats": stats(actions[:, -1]),
            "policy_sized_action_from_demo_first6": [float(x) for x in policy_sized_action],
            "action_reproductions": action_reproductions,
            "best_action_adapter_strategy_for_first_demo_action": best_strategy,
            "state_reproduction": state_reproduction,
            "image_reproduction": image_reproduction,
        }
        if best_strategy != ACTION_STRATEGY_GRIPPER_ZERO_HOLD:
            issues.append(
                issue(
                    "gripper_adapter_strategy",
                    "high",
                    "The first demonstration action is best reproduced by a non-zero-hold gripper strategy.",
                    {
                        "best_strategy": best_strategy,
                        "zero_hold_l1": action_reproductions[ACTION_STRATEGY_GRIPPER_ZERO_HOLD]["l1_to_demo_first_action"],
                        "close_l1": action_reproductions[ACTION_STRATEGY_GRIPPER_CLOSE]["l1_to_demo_first_action"],
                        "open_l1": action_reproductions[ACTION_STRATEGY_GRIPPER_OPEN]["l1_to_demo_first_action"],
                        "first_demo_gripper": float(first_action[-1]),
                    },
                    "Before another learned-policy diagnostic, prefer a demonstration-informed gripper strategy or explicitly test close/open with a task hypothesis.",
                )
            )
        if state_reproduction["rebuilt_matches_ee_states"]:
            issues.append(
                issue(
                    "state_reproduction",
                    "low",
                    "HDF5 ee_pos + ee_ori exactly reproduces the 6D ee_states vector for the first timestep.",
                    {"policy_state_shape": policy_state_shape},
                    "Use HDF5 ee_states as the report-only reference for future state adapter checks.",
                )
            )
        if len(image_keys) < len(policy_image_shapes):
            issues.append(
                issue(
                    "image_reproduction",
                    "medium",
                    "HDF5 provides fewer image streams than the policy config expects, so camera3 remains an alias/duplication choice.",
                    {"hdf5_image_keys": image_keys, "policy_image_keys": sorted(policy_image_shapes)},
                    "Keep camera aliasing labeled as a smoke approximation until a documented deployment camera contract is found.",
                )
            )

issues.append(
    issue(
        "hdf5_audit_dependency",
        "high",
        "The previous HDF5 interface audit already blocks rollout scaling.",
        {
            "hdf5_audit_decision": hdf5_audit.get("decision"),
            "hdf5_audit_high_severity_issue_count": hdf5_audit.get("high_severity_issue_count"),
            "hdf5_audit_ready_for_rollout_scaling": hdf5_audit.get("ready_for_rollout_scaling"),
        },
        "Use this reproduction check to choose a bounded compatibility hypothesis, not to justify rollout scaling.",
    )
)

high_count = sum(1 for item in issues if item["severity"] == "high")
passed = not stop_reasons
decision = "no_go_rollout_scaling" if passed else "stop"
recommended_next_step = (
    "Plan a bounded one-task gripper-close compatibility diagnostic only if it is tied to this HDF5 evidence; otherwise keep learned-policy rollout scaling blocked."
    if passed
    else "Fix missing offline adapter reproduction inputs before continuing."
)

report = {
    "evidence_label": "offline_adapter_reproduction_check",
    "offline_adapter_reproduction_check_passed": passed,
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
        "libero_data_root": str(LIBERO_DATA),
        "hdf5_path": str(hdf5_path) if hdf5_path else None,
        "hdf5_audit": str(HDF5_AUDIT),
    },
    "policy_config": {
        "state_shape": policy_state_shape,
        "action_shape": policy_action_shape,
        "image_shapes": policy_image_shapes,
    },
    "reproduction": reproduction,
    "issues": issues,
    "stop_reasons": stop_reasons,
    "recommended_next_step": recommended_next_step,
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Offline Adapter Reproduction Check Report",
    "",
    f"- decision: {report['decision']}",
    f"- check passed: {report['offline_adapter_reproduction_check_passed']}",
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
        "This check is report-only offline dataset evidence. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
    ]
)
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
