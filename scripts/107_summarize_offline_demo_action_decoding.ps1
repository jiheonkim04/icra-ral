param(
    [string]$SourceReportPath = "reports\offline_demo_action_decoding_report.json",
    [string]$JsonReportPath = "reports\offline_demo_action_decoding_summary_report.json",
    [string]$MarkdownReportPath = "reports\offline_demo_action_decoding_summary_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Offline demonstration action decoding summary"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is summary-only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Read-JsonFileIfPresent {
    param([string]$Path)
    $fullPath = Resolve-RepoPath -Path $Path
    if (-not (Test-Path -LiteralPath $fullPath)) {
        return [ordered]@{ present = $false; path = $fullPath; data = $null; error = $null }
    }
    try {
        $text = [System.IO.File]::ReadAllText($fullPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
        return [ordered]@{ present = $true; path = $fullPath; data = ($text | ConvertFrom-Json); error = $null }
    } catch {
        return [ordered]@{ present = $true; path = $fullPath; data = $null; error = $_.Exception.Message }
    }
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
    $markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Offline Demonstration Action Decoding Summary Report",
        "",
        "- decision: $($Report.decision)",
        "- summary passed: $($Report.offline_demo_action_decoding_summary_passed)",
        "- source diagnostic passed: $($Report.metrics.source_diagnostic_passed)",
        "- action L1 to expert: $($Report.metrics.action_l1_to_expert)",
        "- action MSE to expert: $($Report.metrics.action_mse_to_expert)",
        "- policy first-6 L1 to expert first-6: $($Report.metrics.policy6_l1_to_expert_first6)",
        "- offline alignment signal: $($Report.metrics.offline_alignment_signal)",
        "- rollout scaling ready: $($Report.ready_for_rollout_scaling)",
        "- paper-grade claim ready: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This summary is diagnostic only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 14
}

$executionGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
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
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT"
)
$setExecutionGates = @($executionGates | Where-Object { -not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_)) })

$source = Read-JsonFileIfPresent -Path $SourceReportPath
$stopReasons = New-Object System.Collections.Generic.List[string]
if ($setExecutionGates.Count -gt 0) {
    $stopReasons.Add("summary-only offline decoding report refuses execution gates: $($setExecutionGates -join ', ')")
}
if (-not $source.present -or $source.error -or $null -eq $source.data) {
    $stopReasons.Add("Missing or unreadable offline decoding report: $($source.path)")
}

$metrics = [ordered]@{
    source_diagnostic_passed = $false
    action_l1_to_expert = $null
    action_mse_to_expert = $null
    policy6_l1_to_expert_first6 = $null
    action_finite = $false
    load_vlm_weights = $null
    offline_alignment_signal = "unknown"
}
if ($stopReasons.Count -eq 0) {
    $sourceMetrics = $source.data.metrics
    $metrics.source_diagnostic_passed = [bool]$source.data.offline_demo_action_decoding_passed
    $metrics.action_l1_to_expert = [double]$sourceMetrics.action_l1_to_expert
    $metrics.action_mse_to_expert = [double]$sourceMetrics.action_mse_to_expert
    $metrics.policy6_l1_to_expert_first6 = [double]$sourceMetrics.policy6_l1_to_expert_first6
    $metrics.action_finite = [bool]$sourceMetrics.action_finite
    $metrics.load_vlm_weights = [bool]$sourceMetrics.load_vlm_weights
    $metrics.offline_alignment_signal = if (-not $metrics.source_diagnostic_passed -or -not $metrics.action_finite) {
        "invalid"
    } elseif ($metrics.action_l1_to_expert -le 0.1) {
        "strong"
    } elseif ($metrics.action_l1_to_expert -le 0.25) {
        "moderate"
    } else {
        "weak"
    }
}

$summaryPassed = [bool]($stopReasons.Count -eq 0)
$decision = if (-not $summaryPassed) {
    "stop"
} elseif ($metrics.offline_alignment_signal -in @("strong", "moderate")) {
    "proceed_with_caution"
} else {
    "no_go_rollout_scaling"
}

$report = [ordered]@{
    evidence_label = "offline_demo_action_decoding_summary"
    offline_demo_action_decoding_summary_passed = $summaryPassed
    decision = $decision
    ready_for_rollout_scaling = $false
    ready_for_benchmark_claim = $false
    ready_for_paper_claim = $false
    policy = [ordered]@{
        summary_only = $true
        downloads_performed = $false
        installs_performed = $false
        heavy_model_imports_performed = $false
        model_load_performed = $false
        model_inference_performed = $false
        simulator_environment_created = $false
        rollouts_performed = $false
        benchmark_rollouts_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    claims = [ordered]@{
        standard_success_claimed = $false
        benchmark_success_claimed = $false
        counterfactual_robustness_claimed = $false
        sota_claimed = $false
        paper_grade_claim_made = $false
    }
    source_report = $source.path
    metrics = $metrics
    dangerous_execution_gates_set = @($setExecutionGates)
    stop_reasons = @($stopReasons)
    reason = if ($summaryPassed) {
        if ($metrics.offline_alignment_signal -eq "weak") {
            "One-sample offline action decoding completed, but action error to expert is large; rollout scaling remains blocked."
        } else {
            "One-sample offline action decoding completed; still diagnostic-only and not rollout evidence."
        }
    } else {
        $stopReasons -join "; "
    }
    recommended_next_step = if ($summaryPassed -and $metrics.offline_alignment_signal -eq "weak") {
        "Do not scale rollout. Inspect VLM loading policy, checkpoint provenance, and action normalization before another learned-policy rollout."
    } elseif ($summaryPassed) {
        "Plan a bounded repeated offline decoding check over a tiny HDF5 subset before any rollout decision."
    } else {
        "Resolve missing offline decoding summary inputs."
    }
}

Write-Reports -Report $report
exit 0
