param(
    [string]$ResetStepReportPath = "reports\bounded_simulator_reset_step_smoke_report.json",
    [int]$TaskCount = 1,
    [int]$MaxEpisodes = 1,
    [int]$MaxStepsPerEpisode = 5,
    [int]$ExpectedRuntimeMinutes = 10,
    [double]$ExpectedVramGb = 0,
    [string]$JsonReportPath = "reports\tiny_diagnostic_rollout_plan_report.json",
    [string]$MarkdownReportPath = "reports\tiny_diagnostic_rollout_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Tiny diagnostic rollout risk planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not execute rollout, create simulator environments, run policy inference, train, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

function Read-JsonFileIfPresent {
    param([string]$Path)
    $fullPath = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
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
    $jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
    $markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Tiny Diagnostic Rollout Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- risk envelope inside budget: $($Report.risk_envelope_inside_budget)",
        "- reset/step smoke passed: $($Report.reset_step_smoke.bounded_simulator_reset_step_smoke_passed)",
        "- bounded tiny diagnostic rollout execution authorized: $($Report.ready_for_tiny_diagnostic_rollout_execution)",
        "- rollouts performed: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning-only. It is not rollout evidence, standard success, benchmark evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

$dangerousGateNames = @(
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_SIMULATOR_RENDER_SMOKE"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })
$resetRead = Read-JsonFileIfPresent -Path $ResetStepReportPath

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ($dangerousGatesSet.Count -gt 0) {
    $stopReasons.Add("planning-only rollout gate refuses execution environment gates: $($dangerousGatesSet -join ', ')")
}
if (-not $resetRead.present -or $resetRead.error -or $null -eq $resetRead.data) {
    $stopReasons.Add("bounded reset/step smoke report is missing or unreadable")
}

$resetPassed = $false
if ($resetRead.present -and -not $resetRead.error -and $null -ne $resetRead.data) {
    $resetPassed = [bool]$resetRead.data.bounded_simulator_reset_step_smoke_passed
}
if (-not $resetPassed) {
    $stopReasons.Add("bounded reset/step smoke has not passed")
}
if ($TaskCount -lt 1 -or $TaskCount -gt 5) {
    $stopReasons.Add("tiny diagnostic rollout plan is capped at 5 tasks")
}
if ($MaxEpisodes -lt 1 -or $MaxEpisodes -gt 1) {
    $stopReasons.Add("tiny diagnostic rollout plan is capped at one episode")
}
if ($MaxStepsPerEpisode -lt 1 -or $MaxStepsPerEpisode -gt 5) {
    $stopReasons.Add("tiny diagnostic rollout plan is capped at 5 steps per episode")
}
if ($ExpectedRuntimeMinutes -gt 30) {
    $stopReasons.Add("tiny diagnostic rollout planning envelope exceeds 30 minutes")
}
if ($ExpectedVramGb -gt 14) {
    $stopReasons.Add("tiny diagnostic rollout planning envelope exceeds 14 GB VRAM")
}

$riskEnvelopeInsideBudget = [bool]($stopReasons.Count -eq 0)
$decision = if ($riskEnvelopeInsideBudget) { "proceed" } else { "stop" }
$reason = if ($riskEnvelopeInsideBudget) {
    "Planning envelope is bounded; bounded tiny diagnostic rollout execution is authorized only through a separate task-local ALLOW_TINY_ROLLOUT=1 execution script."
} else {
    $stopReasons -join "; "
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        downloads_performed = $false
        installs_performed = $false
        simulator_env_created = $false
        simulator_imports_performed = $false
        render_smoke_performed = $false
        reset_step_smoke_performed = $false
        rollouts_performed = $false
        policy_inference_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        heavy_model_imports_performed = $false
        model_load_performed = $false
        model_inference_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    risk_assessment = [ordered]@{
        task = "tiny diagnostic rollout planning"
        source = "local LIBERO/RoboSuite source/data paths and passed WSL import/render/reset-step smoke reports"
        expected_size_gb = 0
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 4
        expected_vram_gb = $ExpectedVramGb
        task_count = $TaskCount
        max_episodes = $MaxEpisodes
        max_steps_per_episode = $MaxStepsPerEpisode
        token_license_payment_needed = $false
        cuda_driver_system_graphics_changes = $false
        rollout_would_run_now = $false
        bounded_tiny_diagnostic_rollout_allowed_after_green_risk = $true
        decision = $decision
        reason = $reason
    }
    reset_step_smoke = [ordered]@{
        report_present = [bool]$resetRead.present
        report_path = $resetRead.path
        report_error = $resetRead.error
        bounded_simulator_reset_step_smoke_passed = $resetPassed
    }
    dangerous_execution_gates_set = @($dangerousGatesSet)
    risk_envelope_inside_budget = $riskEnvelopeInsideBudget
    ready_for_tiny_diagnostic_rollout_plan = $riskEnvelopeInsideBudget
    ready_for_tiny_diagnostic_rollout_execution = $riskEnvelopeInsideBudget
    ready_for_rollout = $false
    execution_authorized_by_this_planner = $riskEnvelopeInsideBudget
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = if ($riskEnvelopeInsideBudget) {
        "Run scripts\63_bounded_tiny_diagnostic_rollout.ps1 with task-local ALLOW_TINY_ROLLOUT=1. Do not run benchmark rollouts, multi-seed rollouts, OpenVLA-OFT, training, or paper claims."
    } else {
        "Resolve listed blockers before any rollout planning or execution."
    }
}

Write-Reports -Report $report
exit 0
