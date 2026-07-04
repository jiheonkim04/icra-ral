param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ZeroActionComparisonPath = "reports\zero_action_policy_diagnostic_comparison_report.json",
    [string]$AdapterStrategyReportPath = "reports\adapter_strategy_action_scale_diagnostic_report.json",
    [string]$ActionScaleReportPath = "reports\action_scale_diagnostic_report.json",
    [string]$PromptFormatReportPath = "reports\prompt_format_diagnostic_report.json",
    [string]$CameraSourceReportPath = "reports\camera_source_diagnostic_report.json",
    [string]$StateSufficiencyReportPath = "reports\state_sufficiency_diagnostic_report.json",
    [string]$JsonReportPath = "reports\learned_policy_diagnostic_synthesis_report.json",
    [string]$MarkdownReportPath = "reports\learned_policy_diagnostic_synthesis_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Learned-policy diagnostic synthesis"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads existing diagnostic reports only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$env:TCA_MAP_SYNTH_ZERO = Resolve-RepoPath -Path $ZeroActionComparisonPath
$env:TCA_MAP_SYNTH_ADAPTER = Resolve-RepoPath -Path $AdapterStrategyReportPath
$env:TCA_MAP_SYNTH_SCALE = Resolve-RepoPath -Path $ActionScaleReportPath
$env:TCA_MAP_SYNTH_PROMPT = Resolve-RepoPath -Path $PromptFormatReportPath
$env:TCA_MAP_SYNTH_CAMERA = Resolve-RepoPath -Path $CameraSourceReportPath
$env:TCA_MAP_SYNTH_STATE = Resolve-RepoPath -Path $StateSufficiencyReportPath
$env:TCA_MAP_SYNTH_JSON = Resolve-RepoPath -Path $JsonReportPath
$env:TCA_MAP_SYNTH_MD = Resolve-RepoPath -Path $MarkdownReportPath
$env:TCA_MAP_SYNTH_GATES = ($setExecutionGates -join ";")

$script = @'
import json
import os
from pathlib import Path

REPORTS = {
    "zero_action_comparison": Path(os.environ["TCA_MAP_SYNTH_ZERO"]),
    "adapter_strategy": Path(os.environ["TCA_MAP_SYNTH_ADAPTER"]),
    "action_scale": Path(os.environ["TCA_MAP_SYNTH_SCALE"]),
    "prompt_format": Path(os.environ["TCA_MAP_SYNTH_PROMPT"]),
    "camera_source": Path(os.environ["TCA_MAP_SYNTH_CAMERA"]),
    "state_sufficiency": Path(os.environ["TCA_MAP_SYNTH_STATE"]),
}
JSON_OUT = Path(os.environ["TCA_MAP_SYNTH_JSON"])
MD_OUT = Path(os.environ["TCA_MAP_SYNTH_MD"])
SET_GATES = [item for item in os.environ.get("TCA_MAP_SYNTH_GATES", "").split(";") if item]


def read_json(path: Path):
    if not path.exists():
        return None, f"Missing diagnostic report: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # noqa: BLE001 - exact report parsing issue.
        return None, f"Could not parse {path}: {exc}"


def nested(payload, *keys, default=None):
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def passed_flag(axis: str, payload: dict) -> bool:
    names = {
        "zero_action_comparison": "zero_action_policy_diagnostic_comparison_passed",
        "adapter_strategy": "adapter_strategy_diagnostic_passed",
        "action_scale": "action_scale_diagnostic_passed",
        "prompt_format": "prompt_format_diagnostic_passed",
        "camera_source": "camera_source_diagnostic_passed",
        "state_sufficiency": "state_sufficiency_diagnostic_passed",
    }
    return bool(payload.get(names[axis]))


def best_summary(axis: str, payload: dict) -> dict:
    result = payload.get("result") or {}
    if axis == "zero_action_comparison":
        comparison = payload.get("comparison") or {}
        learned = comparison.get("learned_policy") or comparison.get("smolvla_action") or {}
        zero = comparison.get("zero_action") or {}
        return {
            "axis": axis,
            "passed": passed_flag(axis, payload),
            "variants_completed": None,
            "best_variant": None,
            "best_diagnostic_success_rate": float(learned.get("diagnostic_success_rate", 0) or 0),
            "best_reward_sum": float(learned.get("reward_sum", 0) or 0),
            "zero_action_reward_sum": float(zero.get("reward_sum", 0) or 0),
            "ready_for_rollout_scaling": bool(payload.get("ready_for_rollout_scaling", False)),
        }
    best_keys = [
        "best_strategy",
        "best_action_scale",
        "best_prompt_strategy",
        "best_camera_alias_strategy",
        "best_state_adapter_strategy",
    ]
    best_variant = next((result.get(key) for key in best_keys if key in result), None)
    return {
        "axis": axis,
        "passed": passed_flag(axis, payload),
        "variants_completed": result.get("variants_completed"),
        "best_variant": best_variant,
        "best_diagnostic_success_rate": float(result.get("best_diagnostic_success_rate", 0) or 0),
        "best_reward_sum": float(result.get("best_reward_sum", 0) or 0),
        "ready_for_rollout_scaling": bool(payload.get("ready_for_rollout_scaling", False)),
    }


loaded = {}
stop_reasons = []
if SET_GATES:
    stop_reasons.append("Refusing synthesis while execution gates are set: " + ", ".join(SET_GATES))
for axis, path in REPORTS.items():
    payload, error = read_json(path)
    if error:
        stop_reasons.append(error)
    else:
        loaded[axis] = payload

axis_summaries = [best_summary(axis, loaded[axis]) for axis in REPORTS if axis in loaded]
all_reports_present = len(axis_summaries) == len(REPORTS)
all_axis_passed = all(summary["passed"] for summary in axis_summaries) if axis_summaries else False
positive_signal_axes = [
    summary["axis"]
    for summary in axis_summaries
    if summary["best_reward_sum"] > 0 or summary["best_diagnostic_success_rate"] > 0
]
any_axis_ready_to_scale = any(summary["ready_for_rollout_scaling"] for summary in axis_summaries)
diagnostic_ladder_complete = all_reports_present and all_axis_passed
positive_signal_found = bool(positive_signal_axes)
ready_for_rollout_scaling = bool(diagnostic_ladder_complete and positive_signal_found and any_axis_ready_to_scale)

if all_reports_present and not all_axis_passed:
    failed_axes = [summary["axis"] for summary in axis_summaries if not summary["passed"]]
    stop_reasons.append("One or more diagnostic reports did not pass wrapper checks: " + ", ".join(failed_axes))

no_go_reasons = []
if diagnostic_ladder_complete and not positive_signal_found:
    no_go_reasons.append("All bounded learned-policy diagnostic axes completed, but none produced nonzero reward or diagnostic success.")
if not any_axis_ready_to_scale:
    no_go_reasons.append("Every diagnostic report keeps ready_for_rollout_scaling=false.")
if stop_reasons:
    no_go_reasons.append("Required diagnostic inputs are incomplete or unsafe for synthesis.")

passed = not stop_reasons and diagnostic_ladder_complete
decision = "no_go_rollout_scaling" if passed and not ready_for_rollout_scaling else ("proceed_to_bounded_scaling" if ready_for_rollout_scaling else "stop")
recommended_next_step = (
    "Do not scale learned-policy rollouts. Create a bounded environment-policy compatibility audit focused on task/checkpoint alignment, action convention, and observation convention before another one-task diagnostic."
    if decision == "no_go_rollout_scaling"
    else "Fix missing or failed diagnostic reports before synthesis."
    if decision == "stop"
    else "Run only a separately risk-assessed bounded rollout-scaling planner; do not make paper-grade claims."
)

report = {
    "evidence_label": "learned_policy_diagnostic_synthesis",
    "learned_policy_diagnostic_synthesis_passed": passed,
    "decision": decision,
    "diagnostic_ladder_complete": diagnostic_ladder_complete,
    "positive_diagnostic_signal_found": positive_signal_found,
    "positive_signal_axes": positive_signal_axes,
    "ready_for_rollout_scaling": ready_for_rollout_scaling,
    "ready_for_paper_claim": False,
    "no_go_for_rollout_scaling_reason": "; ".join(no_go_reasons) if no_go_reasons else None,
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
    "source_reports": {axis: str(path) for axis, path in REPORTS.items()},
    "diagnostic_axes": axis_summaries,
    "stop_reasons": stop_reasons,
    "recommended_next_step": recommended_next_step,
}

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Learned-Policy Diagnostic Synthesis Report",
    "",
    f"- decision: {report['decision']}",
    f"- synthesis passed: {report['learned_policy_diagnostic_synthesis_passed']}",
    f"- diagnostic ladder complete: {report['diagnostic_ladder_complete']}",
    f"- positive diagnostic signal found: {report['positive_diagnostic_signal_found']}",
    f"- rollout scaling ready: {report['ready_for_rollout_scaling']}",
    f"- paper-grade claim ready: {report['ready_for_paper_claim']}",
    "",
    "## Axis Summary",
    "",
]
for summary in axis_summaries:
    lines.append(
        f"- {summary['axis']}: passed={summary['passed']}, best={summary['best_variant']}, "
        f"success={summary['best_diagnostic_success_rate']}, reward={summary['best_reward_sum']}, "
        f"ready_for_rollout_scaling={summary['ready_for_rollout_scaling']}"
    )
lines.extend(
    [
        "",
        "## No-Go Reason",
        "",
        report["no_go_for_rollout_scaling_reason"] or "No no-go reason recorded.",
        "",
        "## Recommended Next Step",
        "",
        report["recommended_next_step"],
        "",
        "This synthesis is diagnostic/local-pilot evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.",
    ]
)
MD_OUT.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
'@

$script | & $Python -
exit $LASTEXITCODE
