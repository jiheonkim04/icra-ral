param(
    [string]$ImportSmokeReportPath = "reports\bounded_simulator_import_smoke_report.json",
    [string]$RenderSmokeReportPath = "reports\bounded_simulator_render_smoke_report.json",
    [string]$ResetStepReportPath = "reports\bounded_simulator_reset_step_smoke_report.json",
    [string]$TinyDiagnosticReportPath = "reports\bounded_tiny_diagnostic_rollout_report.json",
    [int]$TaskCount = 1,
    [int]$MaxStepsPerTask = 3,
    [int]$ExpectedRuntimeMinutes = 15,
    [double]$ExpectedVramGb = 0,
    [string]$JsonReportPath = "reports\libero_robosuite_diagnostic_rollout_plan_report.json",
    [string]$MarkdownReportPath = "reports\libero_robosuite_diagnostic_rollout_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO/RoboSuite diagnostic rollout risk planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not create environments, rollout, train, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

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
        "# LIBERO/RoboSuite Diagnostic Rollout Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- risk envelope inside budget: $($Report.risk_envelope_inside_budget)",
        "- ready for bounded diagnostic execution: $($Report.ready_for_libero_robosuite_diagnostic_rollout_execution)",
        "- task count: $($Report.risk_assessment.task_count)",
        "- max steps per task: $($Report.risk_assessment.max_steps_per_task)",
        "- benchmark rollout ready: false",
        "- paper claim ready: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning-only. It is not rollout evidence, standard success, benchmark evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

$dangerousGateNames = @(
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_TINY_ROLLOUT"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$importRead = Read-JsonFileIfPresent -Path $ImportSmokeReportPath
$renderRead = Read-JsonFileIfPresent -Path $RenderSmokeReportPath
$resetRead = Read-JsonFileIfPresent -Path $ResetStepReportPath
$tinyRead = Read-JsonFileIfPresent -Path $TinyDiagnosticReportPath

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ($dangerousGatesSet.Count -gt 0) {
    $stopReasons.Add("planning-only diagnostic rollout gate refuses execution environment gates: $($dangerousGatesSet -join ', ')")
}
if ($TaskCount -lt 1 -or $TaskCount -gt 5) {
    $stopReasons.Add("diagnostic rollout task count is capped at 5")
}
if ($MaxStepsPerTask -lt 1 -or $MaxStepsPerTask -gt 5) {
    $stopReasons.Add("diagnostic rollout is capped at 5 steps per task")
}
if ($ExpectedRuntimeMinutes -gt 30) {
    $stopReasons.Add("diagnostic rollout planning envelope exceeds 30 minutes")
}
if ($ExpectedVramGb -gt 14) {
    $stopReasons.Add("diagnostic rollout planning envelope exceeds 14 GB VRAM")
}

$importPassed = $false
if ($importRead.present -and -not $importRead.error -and $null -ne $importRead.data) {
    $importPassed = [bool]$importRead.data.bounded_simulator_import_smoke_passed
} else {
    $stopReasons.Add("bounded simulator import smoke report is missing or unreadable")
}
if (-not $importPassed) { $stopReasons.Add("bounded simulator import smoke has not passed") }

$renderPassed = $false
if ($renderRead.present -and -not $renderRead.error -and $null -ne $renderRead.data) {
    $renderPassed = [bool]$renderRead.data.bounded_simulator_render_smoke_passed
} else {
    $stopReasons.Add("bounded simulator render smoke report is missing or unreadable")
}
if (-not $renderPassed) { $stopReasons.Add("bounded simulator render smoke has not passed") }

$resetPassed = $false
if ($resetRead.present -and -not $resetRead.error -and $null -ne $resetRead.data) {
    $resetPassed = [bool]$resetRead.data.bounded_simulator_reset_step_smoke_passed
} else {
    $stopReasons.Add("bounded simulator reset/step smoke report is missing or unreadable")
}
if (-not $resetPassed) { $stopReasons.Add("bounded simulator reset/step smoke has not passed") }

$tinyDiagnosticPassed = $false
if ($tinyRead.present -and -not $tinyRead.error -and $null -ne $tinyRead.data) {
    $tinyDiagnosticPassed = [bool]$tinyRead.data.bounded_tiny_diagnostic_rollout_passed
} else {
    $warnings.Add("bounded toy MuJoCo diagnostic rollout report is missing; LIBERO/RoboSuite diagnostic can still proceed if import/render/reset-step passed")
}

$liberoRoot = "C:\assets\repos\LIBERO"
$robosuiteRoot = "C:\assets\repos\robosuite"
$liberoDataRoot = "C:\assets\data\libero"
foreach ($path in @($liberoRoot, $robosuiteRoot, $liberoDataRoot)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $stopReasons.Add("required local path does not exist: $path")
    }
}

$riskEnvelopeInsideBudget = [bool]($stopReasons.Count -eq 0)
$decision = if ($riskEnvelopeInsideBudget) { "proceed" } else { "stop" }
$reason = if ($riskEnvelopeInsideBudget) {
    "Import, render, reset/step, local paths, and rollout budget are green for one bounded LIBERO/RoboSuite diagnostic runner."
} else {
    $stopReasons -join "; "
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        downloads_performed = $false
        installs_performed = $false
        simulator_environment_created = $false
        diagnostic_rollouts_performed = $false
        benchmark_rollouts_performed = $false
        policy_inference_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        heavy_model_imports_performed = $false
        model_load_performed = $false
        model_inference_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        benchmark_claims_made = $false
        sota_claims_made = $false
        paper_grade_claims_made = $false
    }
    risk_assessment = [ordered]@{
        task = "bounded LIBERO/RoboSuite diagnostic rollout planning"
        command = "scripts\65_bounded_libero_robosuite_diagnostic_rollout.ps1"
        source = "local official LIBERO and RoboSuite source checkouts plus local official LIBERO data"
        expected_size_gb = 0
        target_paths = @($liberoRoot, $robosuiteRoot, $liberoDataRoot)
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 6
        expected_vram_gb = $ExpectedVramGb
        task_count = $TaskCount
        max_steps_per_task = $MaxStepsPerTask
        token_license_payment_needed = $false
        cuda_driver_system_graphics_changes = $false
        simulator_will_run_in_execution_script = $riskEnvelopeInsideBudget
        benchmark_rollout_would_run = $false
        paper_claim_would_be_made = $false
        decision = $decision
        reason = $reason
    }
    prerequisites = [ordered]@{
        import_smoke = [ordered]@{ report_present = [bool]$importRead.present; report_path = $importRead.path; report_error = $importRead.error; passed = $importPassed }
        render_smoke = [ordered]@{ report_present = [bool]$renderRead.present; report_path = $renderRead.path; report_error = $renderRead.error; passed = $renderPassed }
        reset_step_smoke = [ordered]@{ report_present = [bool]$resetRead.present; report_path = $resetRead.path; report_error = $resetRead.error; passed = $resetPassed }
        toy_diagnostic_rollout = [ordered]@{ report_present = [bool]$tinyRead.present; report_path = $tinyRead.path; report_error = $tinyRead.error; passed = $tinyDiagnosticPassed }
    }
    dangerous_execution_gates_set = @($dangerousGatesSet)
    risk_envelope_inside_budget = $riskEnvelopeInsideBudget
    ready_for_libero_robosuite_diagnostic_rollout_plan = $riskEnvelopeInsideBudget
    ready_for_libero_robosuite_diagnostic_rollout_execution = $riskEnvelopeInsideBudget
    ready_for_benchmark_rollout = $false
    ready_for_paper_claim = $false
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = if ($riskEnvelopeInsideBudget) {
        "Run scripts\65_bounded_libero_robosuite_diagnostic_rollout.ps1 with task-local ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT=1. Do not run benchmark rollouts, multi-seed rollouts, OpenVLA-OFT, training, or paper claims."
    } else {
        "Resolve listed blockers before any LIBERO/RoboSuite diagnostic rollout."
    }
}

Write-Reports -Report $report
exit 0
