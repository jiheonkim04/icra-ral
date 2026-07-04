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
bounded_extension = load_json("reports/bounded_local_pilot_extension_report.json")
hard_stop = load_json("reports/hard_stop_status_report.json")
lora_adapter_plan = load_json("reports/lora_adapter_construction_plan_report.json")
lora_tiny_scaffold = load_json("reports/lora_tiny_smoke_scaffold_report.json")
lora_comparison_plan = load_json("reports/lora_comparison_plan_report.json")
qlora_feasibility = load_json("reports/qlora_feasibility_report.json")
libero_metadata_subset = load_json("reports/libero_metadata_subset_report.json")
libero_offline_interface = load_json("reports/libero_offline_interface_smoke_report.json")
libero_offline_counterfactual_split = load_json("reports/libero_offline_counterfactual_split_report.json")
libero_offline_head_comparison = load_json("reports/libero_offline_actionmap_tca_comparison_report.json")
libero_offline_lora_comparison = load_json("reports/libero_offline_lora_comparison_report.json")

completed = {
    "smolvla_load_only_smoke": passed(load_only, "result", "passed"),
    "single_sample_interface_smoke": passed(single_sample, "result", "passed"),
    "feature_cache_eval_smoke": bool((feature_cache or {}).get("cache_valid")),
    "tiny_head_only_smoke": bool((tiny_head or {}).get("tiny_head_only_smoke_passed")),
    "bounded_local_pilot_extension": bool((bounded_extension or {}).get("bounded_local_pilot_extension_passed")),
}

runtime_reports_available = {
    "smolvla_load_only_smoke_report": load_only is not None,
    "smolvla_single_sample_interface_report": single_sample is not None,
    "feature_cache_eval_report": feature_cache is not None,
    "tiny_head_only_smoke_report": tiny_head is not None,
    "bounded_local_pilot_extension_report": bounded_extension is not None,
    "hard_stop_status_report": hard_stop is not None,
    "lora_adapter_construction_plan_report": lora_adapter_plan is not None,
    "lora_tiny_smoke_scaffold_report": lora_tiny_scaffold is not None,
    "lora_comparison_plan_report": lora_comparison_plan is not None,
    "qlora_feasibility_report": qlora_feasibility is not None,
    "libero_metadata_subset_report": libero_metadata_subset is not None,
    "libero_offline_interface_smoke_report": libero_offline_interface is not None,
    "libero_offline_counterfactual_split_report": libero_offline_counterfactual_split is not None,
    "libero_offline_actionmap_tca_comparison_report": libero_offline_head_comparison is not None,
    "libero_offline_lora_comparison_report": libero_offline_lora_comparison is not None,
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

bounded_extension_summary = {}
if isinstance(bounded_extension, dict):
    bounded_extension_summary = {
        "passed": bool(bounded_extension.get("bounded_local_pilot_extension_passed")),
        "requested_max_steps": (bounded_extension.get("bounds") or {}).get("requested_max_steps"),
        "runner_max_steps_cap": (bounded_extension.get("bounds") or {}).get("runner_max_steps_cap"),
        "local_policy_max_steps": (bounded_extension.get("bounds") or {}).get("local_policy_max_steps"),
        "safe_to_run_real_dataset_pilot": bool(bounded_extension.get("safe_to_run_real_dataset_pilot")),
        "safe_to_run_rollouts": bool(bounded_extension.get("safe_to_run_rollouts")),
        "offline_proxy_only": bool((bounded_extension.get("policy") or {}).get("offline_proxy_only")),
        "not_paper_grade": bool((bounded_extension.get("policy") or {}).get("not_paper_grade")),
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
libero_data_gates = {
    "metadata_subset_report_present": libero_metadata_subset is not None,
    "metadata_subset_ready": bool((libero_metadata_subset or {}).get("ready_for_metadata_only_subset")),
    "offline_interface_report_present": libero_offline_interface is not None,
    "offline_interface_decision": (libero_offline_interface or {}).get("decision"),
    "ready_for_offline_interface_smoke": bool((libero_offline_interface or {}).get("ready_for_offline_interface_smoke")),
    "counterfactual_split_report_present": libero_offline_counterfactual_split is not None,
    "ready_for_tiny_offline_counterfactual_split": bool((libero_offline_counterfactual_split or {}).get("ready_for_tiny_offline_counterfactual_split")),
    "ready_for_tiny_offline_actionmap_tca_comparison": bool((libero_offline_counterfactual_split or {}).get("ready_for_tiny_offline_actionmap_tca_comparison")),
    "counterfactual_pair_count": (libero_offline_counterfactual_split or {}).get("counterfactual_pair_count"),
    "offline_actionmap_tca_report_present": libero_offline_head_comparison is not None,
    "offline_actionmap_tca_comparison_passed": bool((libero_offline_head_comparison or {}).get("libero_offline_head_comparison_passed")),
    "ready_for_required_tiny_lora_comparison": bool((libero_offline_head_comparison or {}).get("ready_for_required_tiny_lora_comparison")),
    "offline_lora_report_present": libero_offline_lora_comparison is not None,
    "offline_lora_comparison_passed": bool((libero_offline_lora_comparison or {}).get("libero_offline_lora_comparison_passed")),
    "ready_for_bounded_local_pilot_report": bool((libero_offline_lora_comparison or {}).get("ready_for_bounded_local_pilot_report")),
    "ready_for_rollout": bool((libero_offline_interface or {}).get("ready_for_rollout")),
    "reason": (libero_offline_interface or {}).get("reason"),
}
ready_for_bounded_local_pilot = all_safe_smokes_passed
blocked_for_larger_paper_grade_stage = True
blocked_by = [
    "real LIBERO/LIBERO-CF data and simulator rollout assets require risk assessment before use",
    "rollout and simulator execution require risk assessment; proceed only if inside strict budget",
    "real dataset training or benchmark evaluation must stay labeled local/offline unless rollout evidence exists",
    "QLoRA execution requires risk assessment if PEFT/bitsandbytes/tooling are missing or require CUDA/PyTorch/package changes",
    "OpenVLA-OFT download/import/load/execution remains forbidden locally",
    "offline proxy metrics are not standard success and cannot support paper-grade claims",
]
if not libero_data_gates["ready_for_offline_interface_smoke"]:
    blocked_by.append("no tiny local LIBERO-style demo file is ready for offline interface smoke")
if not libero_data_gates["ready_for_tiny_offline_counterfactual_split"]:
    blocked_by.append("no tiny local LIBERO HDF5-backed counterfactual split is ready")
if not libero_data_gates["offline_actionmap_tca_comparison_passed"]:
    blocked_by.append("no tiny local LIBERO offline ActionMap vs TCA-Map comparison has passed")
if not libero_data_gates["offline_lora_comparison_passed"]:
    blocked_by.append("no tiny local LIBERO offline required LoRA comparison has passed")

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
    go_for.append("risk-assessed bounded local SmolVLA pilot tasks inside budget")
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
        "risk_assessed_autonomy_policy": True,
    },
    "decision": decision,
    "go_for": go_for,
    "no_go_for": [
        "paper-grade empirical claims",
        "real benchmark claims from offline proxy evidence",
        "simulator rollouts without passing risk assessment",
        "OpenVLA-OFT execution",
        "multi-seed experiments",
    ],
    "risk_assessment_required_for": [
        "downloads",
        "GPU tasks",
        "bounded training",
        "real dataset setup",
        "simulator readiness",
        "bounded rollouts",
        "large package/tooling changes",
    ],
    "external_irreversible_stop_gates": [
        "token/secret/API key access",
        "paid service",
        "license click-through",
        "external upload/submission/publishing",
        "deleting user files outside approved cache/repo cleanup",
        "system-wide CUDA/PyTorch/driver changes",
        "admin/system-level installers",
        "paper-level empirical claims",
    ],
    "completed_safe_smokes": completed,
    "all_safe_smokes_passed": all_safe_smokes_passed,
    "ready_for_bounded_local_pilot": ready_for_bounded_local_pilot,
    "blocked_for_larger_paper_grade_stage": blocked_for_larger_paper_grade_stage,
    "lora_qlora_planning": lora_qlora_planning,
    "all_lora_qlora_planning_done": all_lora_qlora_planning_done,
    "runtime_reports_available": runtime_reports_available,
    "tiny_head_only_metrics": tiny_metrics,
    "bounded_local_pilot_extension": bounded_extension_summary,
    "libero_data_gates": libero_data_gates,
    "blocked_by": blocked_by,
    "hard_stop_status": {
        "hard_stop_reached": (hard_stop or {}).get("hard_stop_reached"),
        "recommended_next_step": (hard_stop or {}).get("recommended_next_step"),
        "ready_for_smolvla_adapter_smoke": ((hard_stop or {}).get("assets") or {}).get("ready_for_smolvla_adapter_smoke"),
        "ready_for_libero_rollout": ((hard_stop or {}).get("assets") or {}).get("ready_for_libero_rollout"),
        "ready_for_openvla_oft_smoke": ((hard_stop or {}).get("assets") or {}).get("ready_for_openvla_oft_smoke"),
    },
    "recommended_next_step": (
        "Ready for risk-assessed bounded local SmolVLA pilot work. Proceed automatically if the next task is inside budget: official/documented source, <=80GB download with >=100GB disk remaining by default, official LIBERO data exception <=180GB with >=250GB remaining, <=14GB VRAM, <=30 minutes runtime, batch size 1, SmolVLA-only frozen/LoRA/QLoRA training <=300 steps after stable smoke, no OpenVLA-OFT, no token/license/payment gate, and no paper claim."
        if all_safe_smokes_passed
        else "Rerun the missing safe smoke reports before any bounded local pilot or larger experimental stage."
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
        f"Ready for risk-assessed bounded local pilot: `{str(ready_for_bounded_local_pilot).lower()}`",
    f"Blocked for larger paper-grade stage: `{str(blocked_for_larger_paper_grade_stage).lower()}`",
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
        "## LIBERO Data Gates",
        *[f"- `{name}`: `{value}`" for name, value in libero_data_gates.items()],
        "",
        "## Go For",
        *[f"- {item}" for item in go_for],
        "",
        "## No-Go For",
        *[f"- {item}" for item in report["no_go_for"]],
        "",
        "## Risk Assessment Required For",
        *[f"- {item}" for item in report["risk_assessment_required_for"]],
        "",
        "## External Stop Gates",
        *[f"- {item}" for item in report["external_irreversible_stop_gates"]],
        "",
        "## Current Blockers",
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
