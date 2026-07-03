param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\lora_adapter_construction_plan_report.json",
    [string]$PathsFile = "configs\paths.local.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LoRA adapter construction planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, or execute OpenVLA-OFT."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to plan LoRA adapter construction while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}
$pathsFullPath = if ([System.IO.Path]::IsPathRooted($PathsFile)) {
    $PathsFile
} else {
    Join-Path $RepoRoot $PathsFile
}

$env:TCA_MAP_LORA_CONSTRUCTION_REPORT = $reportFullPath
$env:TCA_MAP_PATHS_FILE = $pathsFullPath

$script = @'
import json
import os
from pathlib import Path

import yaml

from tca_map.adapters import validate_lora_policy_config

REPO_ROOT = Path.cwd()
REPORT_PATH = Path(os.environ["TCA_MAP_LORA_CONSTRUCTION_REPORT"])
PATHS_FILE = Path(os.environ["TCA_MAP_PATHS_FILE"])

CONFIGS = {
    "lora": REPO_ROOT / "configs" / "lora_adapter_lowcompute.yaml",
    "qlora": REPO_ROOT / "configs" / "qlora_adapter_lowcompute.yaml",
}
TOKENIZER_NAMES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
]
WEIGHT_PATTERNS = ["*.safetensors", "*.bin"]

def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def local_assets():
    assets = {}
    if PATHS_FILE.exists():
        try:
            assets = (load_yaml(PATHS_FILE).get("assets") or {})
        except Exception:
            assets = {}
    mapping = {
        "smolvla_ckpt": "SMOLVLA_CKPT",
        "checkpoint_root": "CHECKPOINT_ROOT",
        "hf_home": "HF_HOME",
    }
    resolved = {}
    for key, env_name in mapping.items():
        value = os.environ.get(env_name) or assets.get(key)
        resolved[key] = {
            "configured": bool(value),
            "exists": bool(value and Path(value).exists()),
            "value_redacted": "set" if value else None,
            "path": str(Path(value)) if value else None,
            "env": env_name,
        }
    return resolved

def files_for(path, names, patterns):
    root = Path(path) if path else Path("")
    if not root.exists():
        return []
    found = [name for name in names if (root / name).exists()]
    for pattern in patterns:
        found.extend(sorted(p.name for p in root.glob(pattern)))
    return sorted(set(found))

configs = {name: load_yaml(path) for name, path in CONFIGS.items()}
validations = {name: validate_lora_policy_config(config) for name, config in configs.items()}
assets = local_assets()
ckpt_path = assets["smolvla_ckpt"]["path"]
config_found = files_for(ckpt_path, ["config.json"], [])
tokenizer_found = files_for(ckpt_path, TOKENIZER_NAMES, [])
weights_found = files_for(ckpt_path, [], WEIGHT_PATTERNS)

lora_cfg = configs["lora"].get("lora", {})
qlora_cfg = configs["qlora"].get("qlora", {})
training_cfg = configs["lora"].get("training", {})
hard_stop_reasons = []
if not validations["lora"]["passed"]:
    hard_stop_reasons.extend(f"lora config: {error}" for error in validations["lora"]["errors"])
if not validations["qlora"]["passed"]:
    hard_stop_reasons.extend(f"qlora config: {error}" for error in validations["qlora"]["errors"])
if training_cfg.get("train_backbone") is not False:
    hard_stop_reasons.append("LoRA config must keep train_backbone=false")
if training_cfg.get("full_finetune") is not False:
    hard_stop_reasons.append("LoRA config must keep full_finetune=false")
if int(training_cfg.get("max_steps", 0)) > 100:
    hard_stop_reasons.append("LoRA tiny-smoke planning config exceeds max_steps<=100")

module_allowlist = sorted(
    set(lora_cfg.get("trainable_modules") or [])
    | set(qlora_cfg.get("trainable_modules") or [])
)

report = {
    "policy": {
        "planning_only": True,
        "required_lora_track": True,
        "required_qlora_feasibility_track": True,
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
    "configs": {
        name: {
            "path": str(CONFIGS[name]),
            "run_mode": configs[name].get("run", {}).get("mode"),
            "validation": validations[name],
        }
        for name in CONFIGS
    },
    "asset_inputs": {
        "smolvla_ckpt": {k: v for k, v in assets["smolvla_ckpt"].items() if k != "path"},
        "checkpoint_root": {k: v for k, v in assets["checkpoint_root"].items() if k != "path"},
        "hf_home": {k: v for k, v in assets["hf_home"].items() if k != "path"},
        "config_found": config_found,
        "tokenizer_found_in_checkpoint": tokenizer_found,
        "weights_found": weights_found,
    },
    "adapter_construction_plan": {
        "module_allowlist": module_allowlist,
        "forbidden_modules": ["full_backbone", "vision_backbone", "language_backbone", "openvla_oft"],
        "freeze_backbone": True,
        "trainable_modules_only": True,
        "batch_size": 1,
        "max_tiny_smoke_steps": 100,
        "max_tiny_smoke_runtime_minutes": 15,
        "max_vram_target_gb": 14,
        "requires_heavy_import_for_actual_construction": True,
        "heavy_import_allowed_now": False,
    },
    "required_experiment_tracks": [
        "ActionMap + LoRA",
        "TCA-Map + LoRA",
        "TCA-Map + LoRA + Distributional TCA-Select",
        "TCA-Map + QLoRA + Distributional TCA-Select if memory/tooling allows",
    ],
    "hard_stop_reasons": hard_stop_reasons,
    "ready_for_lora_adapter_construction_plan": not hard_stop_reasons,
    "safe_to_run_lora_tiny_smoke_now": False,
    "recommended_next_step": (
        "Create a bounded LoRA tiny-smoke scaffold next. Do not train yet; actual LoRA smoke must keep max_steps<=100, runtime<=15 minutes, VRAM<=14GB, no rollout, no simulator, no OpenVLA-OFT, and no full fine-tuning."
        if not hard_stop_reasons
        else "Fix LoRA/QLoRA config hard-stop reasons before any adapter construction scaffold."
    ),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
