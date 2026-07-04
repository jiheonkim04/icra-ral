param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SourceRepo = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    [string]$MetadataJsonPath = "",
    [string]$TargetRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [string]$JsonReportPath = "reports\vlm_enabled_loading_risk_plan_report.json",
    [string]$MarkdownReportPath = "reports\vlm_enabled_loading_risk_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "VLM-enabled loading risk/provenance plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script queries or reads model metadata only. It does not download model weights, install packages, load models, infer, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

$executionGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
    "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
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

$env:TCA_MAP_VLM_RISK_SOURCE_REPO = $SourceRepo
$env:TCA_MAP_VLM_RISK_METADATA = Resolve-RepoPath -Path $MetadataJsonPath
$env:TCA_MAP_VLM_RISK_TARGET_ROOT = $TargetRoot
$env:TCA_MAP_VLM_RISK_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_VLM_RISK_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_VLM_RISK_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
import shutil
from pathlib import Path

SOURCE_REPO = os.environ["TCA_MAP_VLM_RISK_SOURCE_REPO"]
METADATA_PATH = os.environ.get("TCA_MAP_VLM_RISK_METADATA", "")
TARGET_ROOT = Path(os.environ["TCA_MAP_VLM_RISK_TARGET_ROOT"])
JSON_OUT = Path(os.environ["TCA_MAP_VLM_RISK_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_VLM_RISK_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_VLM_RISK_GATES", "").split(";") if item]


def _disk_free_gb(path: Path) -> float:
    anchor = path.anchor or Path.cwd().anchor or str(Path.cwd())
    usage = shutil.disk_usage(anchor)
    return round(usage.free / (1024**3), 3)


def _metadata_from_hf(repo_id: str) -> dict:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    card_data = getattr(info, "card_data", None)
    license_value = None
    if card_data is not None:
        try:
            license_value = card_data.get("license")
        except AttributeError:
            license_value = getattr(card_data, "license", None)
    siblings = []
    for item in getattr(info, "siblings", []) or []:
        siblings.append({"rfilename": item.rfilename, "size": getattr(item, "size", None)})
    return {
        "id": info.id,
        "private": bool(getattr(info, "private", False)),
        "gated": bool(getattr(info, "gated", False) or False),
        "license": license_value,
        "tags": list(getattr(info, "tags", []) or []),
        "siblings": siblings,
        "metadata_source": "huggingface_hub.model_info",
    }


def _read_metadata(path: str, repo_id: str) -> dict:
    if path:
        metadata_path = Path(path)
        return json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    return _metadata_from_hf(repo_id)


def _size_gb(size_bytes: int | float | None) -> float:
    if size_bytes is None:
        return 0.0
    return float(size_bytes) / (1024**3)


stop_reasons: list[str] = []
if SET_GATES:
    stop_reasons.append("Risk planner refuses execution gates: " + ", ".join(SET_GATES))

metadata_error = None
metadata = {}
try:
    metadata = _read_metadata(METADATA_PATH, SOURCE_REPO)
except Exception as exc:  # noqa: BLE001 - exact metadata blocker matters.
    metadata_error = f"{type(exc).__name__}: {exc}"
    stop_reasons.append("Could not read/query official VLM metadata: " + metadata_error)

siblings = metadata.get("siblings") or []
root_safetensors = [
    item
    for item in siblings
    if item.get("rfilename") == "model.safetensors" or (
        str(item.get("rfilename", "")).startswith("model-") and str(item.get("rfilename", "")).endswith(".safetensors")
    )
]
config_tokenizer = [
    item
    for item in siblings
    if Path(str(item.get("rfilename", ""))).name
    in {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
        "added_tokens.json",
    }
]
required_files = root_safetensors + config_tokenizer
expected_size_gb = round(sum(_size_gb(item.get("size")) for item in required_files), 3)
root_weight_size_gb = round(sum(_size_gb(item.get("size")) for item in root_safetensors), 3)
free_gb = _disk_free_gb(TARGET_ROOT)
free_after_gb = round(free_gb - expected_size_gb, 3)

official_source = metadata.get("id") == SOURCE_REPO and SOURCE_REPO.startswith("HuggingFaceTB/")
private = bool(metadata.get("private"))
gated = bool(metadata.get("gated"))
license_value = metadata.get("license")
license_ok = license_value in {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"} or license_value is None
size_known = bool(root_safetensors and all(item.get("size") for item in required_files))
disk_ok = free_after_gb >= 250
size_ok = expected_size_gb <= 8

if not official_source:
    stop_reasons.append("VLM source is not the expected official Hugging FaceTB repo.")
if private or gated:
    stop_reasons.append("VLM source appears private or gated; token/login/license acceptance would be required.")
if not license_ok:
    stop_reasons.append(f"VLM license is not in the low-risk allowlist: {license_value}")
if not size_known:
    stop_reasons.append("Could not estimate required VLM config/tokenizer/weight size.")
if not size_ok:
    stop_reasons.append(f"Estimated required VLM files exceed 8GB: {expected_size_gb}GB")
if not disk_ok:
    stop_reasons.append(f"Free disk after acquisition would be below 250GB: {free_after_gb}GB")

passed = not stop_reasons
report = {
    "evidence_label": "vlm_enabled_loading_risk_plan",
    "vlm_enabled_loading_risk_plan_passed": passed,
    "decision": "proceed" if passed else "stop",
    "ready_for_vlm_weight_acquisition_plan": passed,
    "ready_for_bounded_vlm_enabled_load_smoke_plan": False,
    "ready_for_rollout_scaling": False,
    "ready_for_benchmark_claim": False,
    "ready_for_paper_claim": False,
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
    "claims": {
        "standard_success_claimed": False,
        "benchmark_success_claimed": False,
        "counterfactual_robustness_claimed": False,
        "sota_claimed": False,
        "paper_grade_claim_made": False,
    },
    "source": {
        "repo_id": SOURCE_REPO,
        "metadata_id": metadata.get("id"),
        "official_source": official_source,
        "private": private,
        "gated": gated,
        "license": license_value,
        "metadata_source": metadata.get("metadata_source", "metadata_json_fixture" if METADATA_PATH else "unknown"),
        "token_login_license_payment_required": bool(private or gated),
    },
    "files": {
        "root_safetensors": root_safetensors,
        "config_tokenizer_processor_files": config_tokenizer,
        "required_file_count": len(required_files),
        "all_sibling_count": len(siblings),
    },
    "risk_assessment": {
        "task": "VLM-enabled SmolVLA loading dependency risk plan",
        "source_url": f"https://huggingface.co/{SOURCE_REPO}",
        "target_path": str(TARGET_ROOT),
        "expected_new_disk_gb": expected_size_gb,
        "root_weight_size_gb": root_weight_size_gb,
        "current_free_disk_gb": free_gb,
        "free_disk_after_estimate_gb": free_after_gb,
        "expected_runtime_minutes_for_future_acquisition": 10,
        "expected_runtime_minutes_for_future_load_smoke": 15,
        "expected_ram_gb_for_future_cpu_load": 10,
        "expected_vram_gb": 0,
        "future_acquisition_gate": "ALLOW_DOWNLOADS=1",
        "future_load_gate": "ALLOW_VLM_ENABLED_LOAD_SMOKE=1",
        "simulator_will_run": False,
        "rollout_will_run": False,
        "training_will_run": False,
        "model_load_in_this_planner": False,
        "metadata_query_only": True,
        "stop_condition": "Stop if source becomes gated/private, size cannot be estimated, disk-after estimate drops below 250GB, CUDA/PyTorch changes are needed, OpenVLA-OFT is required, or runtime exceeds budget.",
        "fallback_plan": "Keep load_vlm_weights=false diagnostics and prepare a cloud handoff if VLM-enabled local load is unsafe.",
        "decision": "proceed" if passed else "stop",
        "reason": "Official public metadata is available, license/token risk is low, size is bounded, and disk budget is green." if passed else "; ".join(stop_reasons),
    },
    "stop_reasons": stop_reasons,
    "metadata_error": metadata_error,
    "recommended_next_step": (
        "Create a separately gated VLM weight acquisition plan/runner for the required config/tokenizer/root model.safetensors files only. Do not load the model until acquisition and a bounded VLM-enabled load-smoke plan pass."
        if passed
        else "Resolve the VLM metadata/source/size/license/disk blocker before any VLM-enabled acquisition or load."
    ),
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# VLM-Enabled Loading Risk Plan Report",
    "",
    f"- decision: {report['decision']}",
    f"- plan passed: {report['vlm_enabled_loading_risk_plan_passed']}",
    f"- source: {SOURCE_REPO}",
    f"- official source: {official_source}",
    f"- private/gated: {private}/{gated}",
    f"- license: {license_value}",
    f"- expected new disk GB: {expected_size_gb}",
    f"- root weight size GB: {root_weight_size_gb}",
    f"- free disk after estimate GB: {free_after_gb}",
    f"- ready for VLM weight acquisition plan: {report['ready_for_vlm_weight_acquisition_plan']}",
    f"- ready for VLM-enabled load smoke: {report['ready_for_bounded_vlm_enabled_load_smoke_plan']}",
    "",
    "This planner is metadata-only. It did not download weights, load models, run inference, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.",
    "",
    "## Recommended Next Step",
    "",
    report["recommended_next_step"],
]
MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
