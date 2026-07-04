param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [ValidateSet("auto", "windows", "wsl", "linux")]
    [string]$RuntimePlatform = "auto",
    [string]$ImportSmokeReportPath = "reports\bounded_simulator_import_smoke_report.json",
    [string]$RenderSmokeReportPath = "reports\bounded_simulator_render_smoke_report.json",
    [string]$JsonReportPath = "reports\simulator_render_reset_plan_report.json",
    [string]$MarkdownReportPath = "reports\simulator_render_reset_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Simulator render/reset-step risk planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not render, reset or step simulator environments, rollout, train, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

function Read-JsonFileIfPresent {
    param([string]$Path)
    $fullPath = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $fullPath)) {
        return [ordered]@{
            present = $false
            path = $fullPath
            data = $null
            error = $null
        }
    }
    try {
        $text = [System.IO.File]::ReadAllText($fullPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
        return [ordered]@{
            present = $true
            path = $fullPath
            data = ($text | ConvertFrom-Json)
            error = $null
        }
    } catch {
        return [ordered]@{
            present = $true
            path = $fullPath
            data = $null
            error = $_.Exception.Message
        }
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
        "# Simulator Render/Reset-Step Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- ready for bounded render smoke plan: $($Report.ready_for_bounded_render_smoke_plan)",
        "- ready for bounded reset/step smoke plan: $($Report.ready_for_bounded_reset_step_smoke_plan)",
        "- import smoke passed: $($Report.import_smoke.bounded_simulator_import_smoke_passed)",
        "- render smoke already passed: $($Report.render_smoke.bounded_simulator_render_smoke_passed)",
        "- render smoke performed: false",
        "- reset/step smoke performed: false",
        "- rollouts performed: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning-only. It is not render evidence, not rollout evidence, and not paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

$dangerousGateNames = @(
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING"
)
$dangerousGatesSet = @(
    $dangerousGateNames |
        Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" }
)

$runDir = Join-Path $RepoRoot "runs\simulator_render_reset_plan"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$plannerJson = Join-Path $runDir "simulator_readiness_plan_report.json"
$plannerMd = Join-Path $runDir "simulator_readiness_plan_report.md"

& powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\43_plan_simulator_readiness.ps1") -PathsFile $PathsFile -RuntimePlatform $RuntimePlatform -JsonReportPath $plannerJson -MarkdownReportPath $plannerMd | Out-Null
$plannerExitCode = $LASTEXITCODE
$plannerRead = Read-JsonFileIfPresent -Path $plannerJson
$importRead = Read-JsonFileIfPresent -Path $ImportSmokeReportPath
$renderRead = Read-JsonFileIfPresent -Path $RenderSmokeReportPath

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
if ($plannerExitCode -ne 0 -or -not $plannerRead.present -or $plannerRead.error) {
    $stopReasons.Add("simulator readiness planner failed or did not write a readable report")
}
if ($dangerousGatesSet.Count -gt 0) {
    $stopReasons.Add("planning-only render/reset-step gate refuses execution environment gates: $($dangerousGatesSet -join ', ')")
}

$planner = $plannerRead.data
$effectivePlatform = $null
$readyForImportSmoke = $false
if ($null -ne $planner) {
    $effectivePlatform = $planner.host.effective_runtime_platform
    $readyForImportSmoke = [bool]$planner.ready_for_simulator_import_smoke
}

$importPassed = $false
if ($importRead.present -and -not $importRead.error -and $null -ne $importRead.data) {
    $importPassed = [bool]$importRead.data.bounded_simulator_import_smoke_passed
} else {
    $stopReasons.Add("bounded simulator import smoke report is missing or unreadable")
}
if (-not $importPassed) {
    $stopReasons.Add("bounded simulator import-only smoke has not passed")
}
if (-not $readyForImportSmoke) {
    $stopReasons.Add("simulator readiness planner does not allow simulator import-smoke readiness")
}
if ($effectivePlatform -eq "windows") {
    $stopReasons.Add("native Windows remains planning-only for simulator render/reset-step work; use WSL2/Linux")
}

$renderPassed = $false
if ($renderRead.present -and -not $renderRead.error -and $null -ne $renderRead.data) {
    $renderPassed = [bool]$renderRead.data.bounded_simulator_render_smoke_passed
} else {
    $warnings.Add("bounded render-smoke report is not present; reset/step planning remains blocked")
}

$readyForRenderPlan = [bool]($stopReasons.Count -eq 0)
$readyForResetStepPlan = [bool]($readyForRenderPlan -and $renderPassed)
$decision = if ($readyForRenderPlan) { "proceed" } else { "stop" }
$recommendedNextStep = if ($readyForResetStepPlan) {
    "Create a separate bounded reset/step smoke branch. It must be task-local gated and still must not rollout, train, use GPU, execute OpenVLA-OFT, or make benchmark claims."
} elseif ($readyForRenderPlan) {
    "Create a separate bounded render-smoke branch. It must be task-local gated and must not reset/step environments, rollout, train, use GPU, execute OpenVLA-OFT, or make benchmark claims."
} else {
    "Resolve the listed blockers before any simulator render, reset/step, rollout, benchmark, or paper-claim task."
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        installs_performed = $false
        downloads_performed = $false
        simulator_imports_performed = $false
        render_smoke_performed = $false
        reset_step_smoke_performed = $false
        rollouts_performed = $false
        simulator_environment_steps_performed = $false
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
        task = "bounded simulator render/reset-step risk planning"
        source = "local LIBERO/RoboSuite checkouts, local LIBERO data, WSL/Linux runtime, and bounded import-only smoke report"
        expected_size_gb = 0
        target_runtime_platform = $effectivePlatform
        expected_runtime_minutes = 10
        expected_ram_gb = 4
        expected_vram_gb = 0
        token_license_payment_needed = $false
        cuda_driver_system_graphics_changes = $false
        simulator_render_would_run = $false
        simulator_reset_step_would_run = $false
        rollout_would_run = $false
        decision = $decision
        reason = if ($stopReasons.Count -eq 0) { "bounded import-only readiness is present; next render/reset-step work must be a separate gated task" } else { $stopReasons -join "; " }
    }
    planner = $planner
    import_smoke = [ordered]@{
        report_present = [bool]$importRead.present
        report_path = $importRead.path
        report_error = $importRead.error
        bounded_simulator_import_smoke_passed = $importPassed
    }
    render_smoke = [ordered]@{
        report_present = [bool]$renderRead.present
        report_path = $renderRead.path
        report_error = $renderRead.error
        bounded_simulator_render_smoke_passed = $renderPassed
    }
    dangerous_execution_gates_set = @($dangerousGatesSet)
    ready_for_bounded_render_smoke_plan = $readyForRenderPlan
    ready_for_bounded_reset_step_smoke_plan = $readyForResetStepPlan
    ready_for_rollout = $false
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = $recommendedNextStep
}

Write-Reports -Report $report
exit 0
