param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\hard_stop_status_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Hard-stop approval status summary"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script summarizes approval gates only. It does not install packages, download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, access tokens, or execute OpenVLA-OFT."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}
$env:TCA_MAP_HARD_STOP_STATUS_REPORT = $reportFullPath

$script = @'
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

REPO_ROOT = Path.cwd()
REQUIRED_RUNTIME = ["torch", "transformers", "lerobot", "safetensors"]
OPTIONAL_RUNTIME = ["accelerate", "huggingface_hub"]
DANGEROUS_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_INSTALLS",
    "ALLOW_CLOUD_HANDOFF",
]

def run_small(args):
    try:
        result = subprocess.run(
            args,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def load_json(path):
    p = REPO_ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def load_local_paths():
    config_path = REPO_ROOT / "configs" / "paths.local.yaml"
    config_assets = {}
    if yaml is not None and config_path.exists():
        try:
            config_assets = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("assets", {})
        except Exception:
            config_assets = {}

    mapping = {
        "openvla_oft_ckpt": "OPENVLA_OFT_CKPT",
        "smolvla_ckpt": "SMOLVLA_CKPT",
        "libero_root": "LIBERO_ROOT",
        "libero_data_root": "LIBERO_DATA_ROOT",
        "robosuite_root": "ROBOSUITE_ROOT",
        "data_root": "DATA_ROOT",
        "checkpoint_root": "CHECKPOINT_ROOT",
        "hf_home": "HF_HOME",
    }
    resolved = {}
    for key, env_name in mapping.items():
        value = os.environ.get(env_name) or config_assets.get(key)
        path = Path(value) if value else None
        resolved[key] = {
            "env": env_name,
            "configured": bool(value),
            "exists": bool(path and path.exists()),
            "path": str(path) if path else None,
        }
    return resolved

def smolvla_file_readiness(paths):
    ckpt = Path(paths.get("smolvla_ckpt", {}).get("path") or "")
    hf_home = Path(paths.get("hf_home", {}).get("path") or "")
    checkpoint_root = Path(paths.get("checkpoint_root", {}).get("path") or "")

    config_found = []
    tokenizer_found = []
    weights_found = []
    if ckpt.exists():
        config_found = [p.name for p in [ckpt / "config.json"] if p.exists()]
        tokenizer_names = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "sentencepiece.bpe.model",
        ]
        tokenizer_found = [name for name in tokenizer_names if (ckpt / name).exists()]
        weight_patterns = ["*.safetensors", "*.bin"]
        for pattern in weight_patterns:
            weights_found.extend(sorted(p.name for p in ckpt.glob(pattern)))

    dep_roots = [
        hf_home / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct",
        hf_home / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct",
        checkpoint_root / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct",
        checkpoint_root / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct",
    ]
    dep_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "config.json",
    ]
    dependency = {
        "name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "found": False,
        "root": None,
        "files_found": [],
        "candidate_roots": [str(path) for path in dep_roots],
    }
    for root in dep_roots:
        found = [name for name in dep_files if (root / name).exists()]
        if found:
            dependency.update({"found": True, "root": str(root), "files_found": found})
            break

    cache_root_exists = paths.get("hf_home", {}).get("exists") or paths.get("checkpoint_root", {}).get("exists")
    checkpoint_files_present = bool(config_found and weights_found and (tokenizer_found or dependency["found"]))
    return {
        "smolvla_path_configured": paths.get("smolvla_ckpt", {}).get("configured", False),
        "smolvla_path_exists": paths.get("smolvla_ckpt", {}).get("exists", False),
        "smolvla_checkpoint_files_present": checkpoint_files_present,
        "ready_for_smolvla_path_check": paths.get("smolvla_ckpt", {}).get("configured", False)
        and paths.get("smolvla_ckpt", {}).get("exists", False),
        "ready_for_smolvla_adapter_smoke": bool(
            paths.get("smolvla_ckpt", {}).get("configured", False)
            and paths.get("smolvla_ckpt", {}).get("exists", False)
            and checkpoint_files_present
            and cache_root_exists
        ),
        "config_found": config_found,
        "tokenizer_found": tokenizer_found,
        "weights_found": weights_found,
        "external_tokenizer_dependency": dependency,
    }

runtime_required = {name: version(name) for name in REQUIRED_RUNTIME}
runtime_optional = {name: version(name) for name in OPTIONAL_RUNTIME}
missing_required_runtime = [name for name, found in runtime_required.items() if found is None]

asset_report = load_json("reports/missing_assets_runtime.json") or {}
smolvla_runtime_deps_report = load_json("reports/smolvla_runtime_deps_report.json") or {}
runtime_install_plan = load_json("reports/smolvla_runtime_install_plan_report.json") or {}
tiny_plan = load_json("reports/tiny_head_only_pilot_plan_report.json") or {}
local_paths = load_local_paths()
smolvla_readiness = smolvla_file_readiness(local_paths)

dangerous_gate_values = {
    name: bool(os.environ.get(name))
    for name in DANGEROUS_GATES
}
dangerous_gates_set = [name for name, is_set in dangerous_gate_values.items() if is_set]

approval_requests = [
    {
        "gate": "runtime_install",
        "required_before": "installing torch/transformers/lerobot/safetensors or changing CUDA/PyTorch",
        "current_blocker": bool(missing_required_runtime),
        "missing_runtime_packages": missing_required_runtime,
        "safe_planner": "scripts/18_plan_smolvla_runtime_install.ps1",
        "approval_env_if_later_approved": "ALLOW_RUNTIME_INSTALL=1 or explicit install task instructions",
    },
    {
        "gate": "smolvla_load_only_heavy_import",
        "required_before": "setting ALLOW_HEAVY_IMPORT=1 and attempting load-only SmolVLA model construction",
        "current_blocker": True,
        "safe_planner": "scripts/15_plan_smolvla_load_only_smoke.ps1",
        "approval_env_if_later_approved": "ALLOW_HEAVY_IMPORT=1",
    },
    {
        "gate": "tiny_head_only_training",
        "required_before": "running any head-only training, even tiny/offline-proxy training",
        "current_blocker": True,
        "safe_planner": "scripts/26_plan_tiny_head_only_pilot.ps1",
        "approval_env_if_later_approved": "ALLOW_TINY_TRAINING=1 plus explicit training task instructions",
    },
]

git_branch = run_small(["git", "branch", "--show-current"])
git_commit = run_small(["git", "log", "-1", "--oneline"])
git_status = run_small(["git", "status", "--short"])

report = {
    "policy": {
        "summary_only": True,
        "installs_performed": False,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
    },
    "git": {
        "branch": git_branch.get("stdout"),
        "commit": git_commit.get("stdout"),
        "status_short": git_status.get("stdout", ""),
    },
    "dangerous_gates": {
        "values": dangerous_gate_values,
        "set": dangerous_gates_set,
        "none_set": not dangerous_gates_set,
    },
    "runtime": {
        "required": runtime_required,
        "optional": runtime_optional,
        "missing_required": missing_required_runtime,
        "ready_for_load_only_runtime": not missing_required_runtime,
        "latest_runtime_deps_report_ready": smolvla_runtime_deps_report.get("runtime_dependencies", {}).get("ready_for_load_only_runtime"),
        "install_plan_ready_to_request": runtime_install_plan.get("ready_to_request_install_approval"),
    },
    "assets": {
        **smolvla_readiness,
        "ready_for_openvla_oft_smoke": local_paths.get("openvla_oft_ckpt", {}).get("exists", False),
        "ready_for_libero_rollout": all(
            local_paths.get(key, {}).get("exists", False)
            for key in ["libero_root", "libero_data_root", "robosuite_root"]
        ),
        "missing_assets": asset_report.get("missing_assets"),
        "paths": {
            key: {k: v for k, v in value.items() if k != "path"}
            for key, value in local_paths.items()
        },
    },
    "tiny_head_only": {
        "ready_to_request_tiny_training_approval": tiny_plan.get("ready_to_request_tiny_training_approval"),
        "safe_to_run_training_now": tiny_plan.get("safe_to_run_training_now"),
        "configs_pass_policy": tiny_plan.get("configs_pass_policy"),
    },
    "approval_requests": approval_requests,
    "hard_stop_reached": True,
    "hard_stop_reason": (
        "Next meaningful steps require explicit approval for runtime installation, heavy import/load-only model construction, or tiny head-only training."
    ),
    "recommended_next_step": (
        "Request explicit approval for exactly one gated task: runtime install, SmolVLA load-only heavy import, or tiny head-only training. "
        "Do not combine gates."
    ),
}

report_path = Path(os.environ["TCA_MAP_HARD_STOP_STATUS_REPORT"])
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
