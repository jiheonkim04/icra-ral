param(
    [string]$Hdf5ReplayReportPath = "reports\bounded_hdf5_initial_state_replay_report.json",
    [string]$PolicyReadinessReportPath = "reports\libero_policy_rollout_readiness_plan_report.json",
    [string]$SingleActionReportPath = "reports\wsl_smolvla_single_action_smoke_report.json",
    [string]$ReducedScopeReportPath = "reports\bounded_reduced_scope_learned_policy_rollout_report.json",
    [int]$TaskCount = 1,
    [int]$MaxStepsPerTask = 3,
    [int]$ExpectedRuntimeMinutes = 30,
    [double]$ExpectedVramGb = 0,
    [string]$JsonReportPath = "reports\init_state_learned_policy_recheck_plan_report.json",
    [string]$MarkdownReportPath = "reports\init_state_learned_policy_recheck_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Init-state learned-policy LIBERO recheck planner"
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
    $Report | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Init-State Learned-Policy Recheck Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- ready for bounded recheck runner: $($Report.ready_for_bounded_init_state_learned_policy_recheck_runner)",
        "- task count: $($Report.risk_assessment.task_count)",
        "- max steps per task: $($Report.risk_assessment.max_steps_per_task)",
        "- HDF5 replay passed: $($Report.prerequisites.hdf5_replay.passed)",
        "- HDF5 init-state set ok: $($Report.prerequisites.hdf5_replay.set_init_state_ok)",
        "- learned-policy inference in future runner: $($Report.risk_assessment.learned_policy_inference_will_run)",
        "- benchmark rollout ready: false",
        "- paper claim ready: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning-only. It performs no model load, inference, simulator rollout, training, GPU job, download, OpenVLA-OFT execution, token access, or paper claim."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 14
}

$executionGates = @(
    "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_DOWNLOADS"
)
$setExecutionGates = @($executionGates | Where-Object { -not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_)) })

$hdf5Read = Read-JsonFileIfPresent -Path $Hdf5ReplayReportPath
$policyRead = Read-JsonFileIfPresent -Path $PolicyReadinessReportPath
$singleActionRead = Read-JsonFileIfPresent -Path $SingleActionReportPath
$reducedScopeRead = Read-JsonFileIfPresent -Path $ReducedScopeReportPath

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ($setExecutionGates.Count -gt 0) {
    $stopReasons.Add("planning-only init-state learned-policy recheck refuses execution gates: $($setExecutionGates -join ', ')")
}

$hdf5Passed = $false
$hdf5Ready = $false
$setInitStateOk = $false
$hdf5TaskSuite = $null
$hdf5TaskId = $null
$hdf5Steps = $null
$hdf5Path = $null
if ($hdf5Read.present -and -not $hdf5Read.error -and $null -ne $hdf5Read.data) {
    $hdf5Passed = [bool]$hdf5Read.data.bounded_hdf5_initial_state_replay_passed
    $hdf5Ready = [bool]$hdf5Read.data.ready_for_learned_policy_rollout_recheck
    if ($null -ne $hdf5Read.data.replay_result) {
        $setInitStateOk = [bool]$hdf5Read.data.replay_result.set_init_state_ok
        $hdf5Steps = $hdf5Read.data.replay_result.steps_performed
        $hdf5Path = $hdf5Read.data.replay_result.hdf5_path
    }
    if ($null -ne $hdf5Read.data.policy) {
        $hdf5TaskSuite = $hdf5Read.data.policy.task_suite
        $hdf5TaskId = $hdf5Read.data.policy.task_id
    }
} else {
    $stopReasons.Add("bounded HDF5 initial-state replay report is missing or unreadable")
}
if (-not $hdf5Passed) { $stopReasons.Add("bounded HDF5 initial-state replay has not passed") }
if (-not $hdf5Ready) { $stopReasons.Add("HDF5 replay report did not mark learned-policy rollout recheck ready") }
if (-not $setInitStateOk) { $stopReasons.Add("HDF5 replay did not successfully set the demonstration init_state") }

$policyReady = $false
if ($policyRead.present -and -not $policyRead.error -and $null -ne $policyRead.data) {
    $policyReady = [bool]$policyRead.data.ready_for_tiny_learned_policy_rollout_execution
} else {
    $stopReasons.Add("LIBERO learned-policy rollout readiness report is missing or unreadable")
}
if (-not $policyReady) { $stopReasons.Add("LIBERO learned-policy rollout readiness is not execution-ready") }

$singleActionPassed = $false
if ($singleActionRead.present -and -not $singleActionRead.error -and $null -ne $singleActionRead.data) {
    $singleActionPassed = [bool]$singleActionRead.data.wsl_smolvla_single_action_smoke_passed
} else {
    $stopReasons.Add("WSL SmolVLA single-action smoke report is missing or unreadable")
}
if (-not $singleActionPassed) { $stopReasons.Add("WSL SmolVLA single-action smoke has not passed") }

$previousReducedScopePassed = $false
$previousRewardSum = $null
$previousSuccessRate = $null
if ($reducedScopeRead.present -and -not $reducedScopeRead.error -and $null -ne $reducedScopeRead.data) {
    $previousReducedScopePassed = [bool]$reducedScopeRead.data.bounded_reduced_scope_learned_policy_rollout_passed
    try {
        $previousRewardSum = $reducedScopeRead.data.rollout_result.result.reward_sum
        $previousSuccessRate = $reducedScopeRead.data.rollout_result.result.diagnostic_success_rate
    } catch {
        $warnings.Add("reduced-scope rollout report was present but did not expose reward/success summary")
    }
} else {
    $warnings.Add("previous reduced-scope learned-policy rollout report is unavailable; planning can still proceed from HDF5 replay plus readiness evidence")
}

if ($TaskCount -ne 1) { $stopReasons.Add("init-state learned-policy recheck is capped to exactly one task") }
if ($MaxStepsPerTask -lt 1 -or $MaxStepsPerTask -gt 5) { $stopReasons.Add("init-state learned-policy recheck is capped to 1..5 steps per task") }
if ($ExpectedRuntimeMinutes -gt 30) { $stopReasons.Add("expected runtime exceeds 30 minutes") }
if ($ExpectedVramGb -gt 14) { $stopReasons.Add("expected VRAM exceeds 14 GB") }

$decision = if ($stopReasons.Count -eq 0) { "proceed" } else { "stop" }
$reason = if ($decision -eq "proceed") {
    "HDF5 initial-state replay, WSL SmolVLA single-action smoke, and learned-policy rollout readiness are green for a separately gated one-task init-state recheck."
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
        learned_policy_inference_performed = $false
        simulator_environment_created = $false
        rollouts_performed = $false
        benchmark_rollouts_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        benchmark_claims_made = $false
        sota_claims_made = $false
        paper_grade_claims_made = $false
    }
    prerequisites = [ordered]@{
        hdf5_replay = [ordered]@{
            report_present = [bool]$hdf5Read.present
            report_path = $hdf5Read.path
            report_error = $hdf5Read.error
            passed = $hdf5Passed
            ready_for_learned_policy_rollout_recheck = $hdf5Ready
            set_init_state_ok = $setInitStateOk
            steps_performed = $hdf5Steps
            task_suite = $hdf5TaskSuite
            task_id = $hdf5TaskId
            hdf5_path = $hdf5Path
        }
        policy_rollout_readiness = [ordered]@{
            report_present = [bool]$policyRead.present
            report_path = $policyRead.path
            report_error = $policyRead.error
            ready = $policyReady
        }
        wsl_single_action_smoke = [ordered]@{
            report_present = [bool]$singleActionRead.present
            report_path = $singleActionRead.path
            report_error = $singleActionRead.error
            passed = $singleActionPassed
        }
        previous_reduced_scope_rollout = [ordered]@{
            report_present = [bool]$reducedScopeRead.present
            report_path = $reducedScopeRead.path
            report_error = $reducedScopeRead.error
            passed = $previousReducedScopePassed
            reward_sum = $previousRewardSum
            diagnostic_success_rate = $previousSuccessRate
        }
    }
    risk_assessment = [ordered]@{
        task = "bounded init-state learned-policy LIBERO recheck planning"
        future_command = "scripts\102_bounded_init_state_learned_policy_recheck.ps1"
        source = "local HDF5 demonstration initial state plus local SmolVLA checkpoint and local official LIBERO/RoboSuite source/data"
        expected_size_gb = 0
        current_free_disk_gb = Get-FreeDiskGb
        target_output_paths = @("reports\init_state_learned_policy_recheck_report.json", "reports\init_state_learned_policy_recheck_report.md")
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 14
        expected_vram_gb = $ExpectedVramGb
        task_count = $TaskCount
        max_steps_per_task = $MaxStepsPerTask
        token_login_license_payment_needed = $false
        simulator_will_run_in_future_runner = $true
        rollout_will_run_in_future_runner = $true
        learned_policy_inference_will_run = $true
        training_will_run = $false
        gpu_job_will_run = $false
        openvla_oft_will_run = $false
        paper_claim_will_be_made = $false
        decision = $decision
        reason = $reason
    }
    runner_plan = [ordered]@{
        task_local_gate = "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1"
        max_tasks = 1
        max_hdf5_demos = 1
        max_steps_per_task = $MaxStepsPerTask
        set_hdf5_init_state_required = $true
        policy_inference_allowed = $true
        training_allowed = $false
        gpu_allowed_by_default = $false
        openvla_oft_allowed = $false
        benchmark_rollout_allowed = $false
        multi_seed_allowed = $false
        evidence_label = "bounded_init_state_learned_policy_recheck_diagnostic"
        acceptance_checks = @(
            "no downloads",
            "no installs",
            "no training",
            "no GPU jobs by default",
            "no OpenVLA-OFT",
            "one HDF5 demo initial state",
            "one task only",
            "at most five policy-controlled steps",
            "diagnostic label only",
            "no standard-success or paper-grade claim"
        )
    }
    dangerous_execution_gates_set = @($setExecutionGates)
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    ready_for_bounded_init_state_learned_policy_recheck_runner = [bool]($decision -eq "proceed")
    ready_for_rollout_scaling = $false
    ready_for_benchmark_rollout = $false
    ready_for_paper_claim = $false
    decision = $decision
    reason = $reason
    recommended_next_step = if ($decision -eq "proceed") {
        "Implement a separately gated one-task init-state learned-policy recheck runner. Keep max steps <=5, CPU/WSL default, no training, no GPU, no OpenVLA-OFT, no multi-seed, and diagnostic-only evidence labels."
    } else {
        "Resolve listed blockers before any init-state learned-policy recheck runner."
    }
}

Write-Reports -Report $report
exit 0
