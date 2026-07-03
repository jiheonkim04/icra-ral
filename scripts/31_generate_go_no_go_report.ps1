param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$JsonReportPath = "reports\go_no_go_status_report.json",
    [string]$MarkdownReportPath = "reports\go_no_go_status_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Go/no-go status summary"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads local reports only. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, execute simulators, access tokens, or execute OpenVLA-OFT."

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
    Write-Host ("Refusing to generate go/no-go report while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) {
    $JsonReportPath
} else {
    Join-Path $RepoRoot $JsonReportPath
}
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) {
    $MarkdownReportPath
} else {
    Join-Path $RepoRoot $MarkdownReportPath
}

$env:TCA_MAP_GO_NO_GO_JSON_REPORT = $jsonFullPath
$env:TCA_MAP_GO_NO_GO_MARKDOWN_REPORT = $markdownFullPath

$script = @'
import json
import os
from pathlib import Path

REPO_ROOT = Path.cwd()

def load_json(path):
    p = REPO_ROOT / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}

def passed(report, *keys):
    value = report or {}
    for key in keys:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return value is True

load_only = load_json("reports/smolvla_load_only_smoke_report.json")
single_sample = load_json("reports/smolvla_single_sample_interface_report.json")
feature_cache = load_json("reports/feature_cache_eval_report.json")
tiny_head = load_json("reports/tiny_head_only_smoke_report.json")
hard_stop = load_json("reports/hard_stop_status_report.json")
lora_adapter_plan = load_json("reports/lora_adapter_construction_plan_report.json")
lora_tiny_scaffold = load_json("reports/lora_tiny_smoke_scaffold_report.json")
lora_comparison_plan = load_json("reports/lora_comparison_plan_report.json")
qlora_feasibility = load_json("reports/qlora_feasibility_report.json")

completed = {
    "smolvla_load_only_smoke": passed(load_only, "result", "passed"),
    "single_sample_interface_smoke": passed(single_sample, "result", "passed"),
    "feature_cache_eval_smoke": bool((feature_cache or {}).get("cache_valid")),
    "tiny_head_only_smoke": bool((tiny_head or {}).get("tiny_head_only_smoke_passed")),
}

runtime_reports_available = {
    "smolvla_load_only_smoke_report": load_only is not None,
    "smolvla_single_sample_interface_report": single_sample is not None,
    "feature_cache_eval_report": feature_cache is not None,
    "tiny_head_only_smoke_report": tiny_head is not None,
    "hard_stop_status_report": hard_stop is not None,
    "lora_adapter_construction_plan_report": lora_adapter_plan is not None,
    "lora_tiny_smoke_scaffold_report": lora_tiny_scaffold is not None,
    "lora_comparison_plan_report": lora_comparison_plan is not None,
    "qlora_feasibility_report": qlora_feasibility is not None,
}

tiny_metrics = {}
if isinstance(tiny_head, dict):
    for head in tiny_head.get("heads", []):
        metrics = head.get("metrics", {})
        tiny_metrics[head.get("head", "unknown")] = {
            "offline_standard_proxy": metrics.get("offline_standard_proxy"),
            "action_l1": metrics.get("action_l1"),
            "action_mse": metrics.get("action_mse"),
            "target_top1_accuracy": metrics.get("target_top1_accuracy"),
            "wrong_target_proxy_rate": metrics.get("wrong_target_proxy_rate"),
            "max_gpu_memory_mb": metrics.get("max_gpu_memory_mb"),
        }

all_safe_smokes_passed = all(completed.values())
lora_qlora_planning = {
    "lora_adapter_construction_plan": bool((lora_adapter_plan or {}).get("ready_for_lora_adapter_construction_plan")),
    "lora_tiny_smoke_scaffold": bool((lora_tiny_scaffold or {}).get("lora_tiny_smoke_scaffold_ready")),
    "lora_comparison_plan": bool((lora_comparison_plan or {}).get("lora_comparison_plan_ready")),
    "qlora_feasibility_check_present": qlora_feasibility is not None,
    "qlora_safe_to_run_now": bool(((qlora_feasibility or {}).get("feasibility") or {}).get("safe_to_run_qlora_now")),
    "qlora_locally_feasible_now": bool(((qlora_feasibility or {}).get("feasibility") or {}).get("locally_feasible_now")),
    "qlora_blockers": (qlora_feasibility or {}).get("blockers", []),
}
all_lora_qlora_planning_done = (
    lora_qlora_planning["lora_adapter_construction_plan"]
    and lora_qlora_planning["lora_tiny_smoke_scaffold"]
    and lora_qlora_planning["lora_comparison_plan"]
    and lora_qlora_planning["qlora_feasibility_check_present"]
)
blocked_by = [
    "real LIBERO/LIBERO-CF data and simulator rollout assets are not validated",
    "rollout and simulator execution require explicit approval",
    "real dataset training beyond the tiny smoke budget requires explicit approval",
    "LoRA comparison execution remains blocked until a bounded runner and explicit tiny-training gate are in place",
    "QLoRA execution remains blocked if PEFT/bitsandbytes/tooling are missing or require unapproved CUDA/PyTorch/package changes",
    "OpenVLA-OFT download/import/load/execution remains forbidden locally",
    "offline proxy metrics are not standard success and cannot support paper-grade claims",
]

decision = (
    "no_go_for_next_larger_experimental_stage"
    if all_safe_smokes_passed
    else "no_go_until_safe_smoke_evidence_is_complete"
)
go_for = [
    "routine safe checks",
    "documentation and checker maintenance",
    "planning-only reports",
]
if all_safe_smokes_passed:
    go_for.append("requesting explicit approval for one true next gate, if the user wants to proceed")
if all_lora_qlora_planning_done:
    go_for.append("LoRA/QLoRA planning interpretation and risk review")

report = {
    "policy": {
        "summary_only": True,
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
    "decision": decision,
    "go_for": go_for,
    "no_go_for": [
        "paper-grade empirical claims",
        "real dataset training",
        "LoRA or QLoRA execution without a bounded runner",
        "simulator rollouts",
        "OpenVLA-OFT execution",
        "multi-seed experiments",
    ],
    "completed_safe_smokes": completed,
    "all_safe_smokes_passed": all_safe_smokes_passed,
    "lora_qlora_planning": lora_qlora_planning,
    "all_lora_qlora_planning_done": all_lora_qlora_planning_done,
    "runtime_reports_available": runtime_reports_available,
    "tiny_head_only_metrics": tiny_metrics,
    "blocked_by": blocked_by,
    "hard_stop_status": {
        "hard_stop_reached": (hard_stop or {}).get("hard_stop_reached"),
        "recommended_next_step": (hard_stop or {}).get("recommended_next_step"),
        "ready_for_smolvla_adapter_smoke": ((hard_stop or {}).get("assets") or {}).get("ready_for_smolvla_adapter_smoke"),
        "ready_for_libero_rollout": ((hard_stop or {}).get("assets") or {}).get("ready_for_libero_rollout"),
        "ready_for_openvla_oft_smoke": ((hard_stop or {}).get("assets") or {}).get("ready_for_openvla_oft_smoke"),
    },
    "recommended_next_step": (
        "No-go for larger experimental stages. Continue only with routine checks/docs, LoRA/QLoRA planning interpretation, or a separate bounded-runner proposal; stop before true gates such as real dataset setup, LoRA/QLoRA execution, simulator rollout, larger training, package/CUDA/PyTorch changes, or OpenVLA-OFT."
        if all_safe_smokes_passed
        else "Rerun the missing safe smoke reports before any larger experimental stage."
    ),
}

json_path = Path(os.environ["TCA_MAP_GO_NO_GO_JSON_REPORT"])
md_path = Path(os.environ["TCA_MAP_GO_NO_GO_MARKDOWN_REPORT"])
json_path.parent.mkdir(parents=True, exist_ok=True)
md_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Go/No-Go Status Report",
    "",
    f"Decision: `{decision}`",
    "",
    "## Safe Smoke Evidence",
]
for name, value in completed.items():
    lines.append(f"- `{name}`: `{str(value).lower()}`")
lines.extend(
    [
        "",
        "## Required LoRA/QLoRA Planning",
        *[f"- `{name}`: `{str(value).lower()}`" for name, value in lora_qlora_planning.items() if name != "qlora_blockers"],
        f"- `qlora_blockers`: `{lora_qlora_planning['qlora_blockers']}`",
        "",
        "## Go For",
        *[f"- {item}" for item in go_for],
        "",
        "## No-Go For",
        *[f"- {item}" for item in report["no_go_for"]],
        "",
        "## Blockers",
        *[f"- {item}" for item in blocked_by],
        "",
        "## Recommended Next Step",
        report["recommended_next_step"],
        "",
        "This report is summary-only. It is not paper evidence and does not claim standard success.",
        "",
    ]
)
md_path.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
