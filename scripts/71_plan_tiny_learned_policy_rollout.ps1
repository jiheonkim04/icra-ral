param(
    [string]$PolicyReadinessReportPath = "reports\libero_policy_rollout_readiness_plan_report.json",
    [string]$SingleActionReportPath = "reports\wsl_smolvla_single_action_smoke_report.json",
    [int]$TaskCount = 1,
    [int]$MaxStepsPerTask = 3,
    [int]$ExpectedRuntimeMinutes = 30,
    [double]$ExpectedVramGb = 0,
    [string]$JsonReportPath = "reports\tiny_learned_policy_rollout_plan_report.json",
    [string]$MarkdownReportPath = "reports\tiny_learned_policy_rollout_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Tiny learned-policy LIBERO rollout planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not load models, run inference, create simulator environments, rollout, train, use GPU, download, execute OpenVLA-OFT, access tokens, or make paper claims."

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

function Get-FreeDiskGb {
    try {
        $drive = Get-PSDrive -Name "C" -ErrorAction Stop
        return [math]::Round(($drive.Free / 1GB), 3)
    } catch {
        return $null
    }
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
    $markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Tiny Learned-Policy LIBERO Rollout Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- ready for execution: $($Report.ready_for_tiny_learned_policy_rollout_execution)",
        "- task count: $($Report.risk_assessment.task_count)",
        "- max steps per task: $($Report.risk_assessment.max_steps_per_task)",
        "- expected runtime minutes: $($Report.risk_assessment.expected_runtime_minutes)",
        "- paper claim ready: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning-only. It performs no model load, inference, simulator rollout, training, GPU job, download, OpenVLA-OFT execution, token access, or paper claim."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 12
}

$dangerousGateNames = @(
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$policyRead = Read-JsonFileIfPresent -Path $PolicyReadinessReportPath
$singleActionRead = Read-JsonFileIfPresent -Path $SingleActionReportPath
$policyReady = $false
if ($policyRead.present -and -not $policyRead.error -and $null -ne $policyRead.data) {
    $policyReady = [bool]$policyRead.data.ready_for_tiny_learned_policy_rollout_execution
}
$singleActionPassed = $false
if ($singleActionRead.present -and -not $singleActionRead.error -and $null -ne $singleActionRead.data) {
    $singleActionPassed = [bool]$singleActionRead.data.wsl_smolvla_single_action_smoke_passed
}

$stopReasons = New-Object System.Collections.Generic.List[string]
if ($dangerousGatesSet.Count -gt 0) { $stopReasons.Add("planning-only tiny learned-policy rollout refuses execution gates: $($dangerousGatesSet -join ', ')") }
if (-not $policyReady) { $stopReasons.Add("learned-policy rollout readiness planner is not execution-ready") }
if (-not $singleActionPassed) { $stopReasons.Add("WSL SmolVLA single-action smoke has not passed") }
if ($TaskCount -lt 1 -or $TaskCount -gt 5) { $stopReasons.Add("tiny learned-policy rollout is capped at 5 tasks") }
if ($MaxStepsPerTask -lt 1 -or $MaxStepsPerTask -gt 10) { $stopReasons.Add("tiny learned-policy rollout is capped at 10 steps per task") }
if ($ExpectedRuntimeMinutes -gt 30) { $stopReasons.Add("expected runtime exceeds 30 minutes") }
if ($ExpectedVramGb -gt 14) { $stopReasons.Add("expected VRAM exceeds 14 GB") }

$decision = if ($stopReasons.Count -eq 0) { "proceed" } else { "stop" }
$reason = if ($stopReasons.Count -eq 0) {
    "WSL policy runtime, local SmolVLA action interface, and LIBERO simulator readiness are green for one bounded learned-policy diagnostic rollout."
} else {
    $stopReasons -join "; "
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
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
    risk_assessment = [ordered]@{
        task = "bounded tiny learned-policy LIBERO rollout planning"
        command = "scripts\72_bounded_tiny_learned_policy_rollout.ps1"
        source = "local SmolVLA checkpoint, local official LIBERO/RoboSuite source, local official LIBERO data"
        expected_size_gb = 0
        disk_free_before_gb = Get-FreeDiskGb
        target_output_paths = @("reports\tiny_learned_policy_rollout_report.json", "reports\tiny_learned_policy_rollout_report.md")
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 12
        expected_vram_gb = $ExpectedVramGb
        task_count = $TaskCount
        max_steps_per_task = $MaxStepsPerTask
        token_login_license_payment_needed = $false
        simulator_will_run = $true
        learned_policy_inference_will_run = $true
        training_will_run = $false
        openvla_oft_will_run = $false
        paper_claim_will_be_made = $false
        decision = $decision
        reason = $reason
    }
    prerequisites = [ordered]@{
        policy_readiness = [ordered]@{ report_present = [bool]$policyRead.present; report_path = $policyRead.path; report_error = $policyRead.error; ready = $policyReady }
        wsl_single_action_smoke = [ordered]@{ report_present = [bool]$singleActionRead.present; report_path = $singleActionRead.path; report_error = $singleActionRead.error; passed = $singleActionPassed }
    }
    dangerous_execution_gates_set = @($dangerousGatesSet)
    ready_for_tiny_learned_policy_rollout_execution = [bool]($decision -eq "proceed")
    ready_for_benchmark_rollout = $false
    ready_for_paper_claim = $false
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = if ($decision -eq "proceed") {
        "Run scripts\72_bounded_tiny_learned_policy_rollout.ps1 with task-local ALLOW_TINY_LEARNED_POLICY_ROLLOUT=1. Label output as diagnostic only."
    } else {
        "Resolve listed blockers before any learned-policy LIBERO rollout."
    }
}

Write-Reports -Report $report
exit 0
