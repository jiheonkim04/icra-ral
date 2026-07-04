param(
    [string]$MetricSummaryReportPath = "reports\tiny_learned_policy_metric_summary_report.json",
    [string]$JsonReportPath = "reports\bounded_learned_policy_rollout_matrix_plan_report.json",
    [string]$MarkdownReportPath = "reports\bounded_learned_policy_rollout_matrix_plan_report.md",
    [int]$MaxMatrixTasks = 3,
    [int]$ReducedScopeTasks = 1,
    [int]$MaxStepsPerTask = 10,
    [int]$ExpectedRuntimeMinutes = 30
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded learned-policy rollout matrix planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

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

$metricFullPath = if ([System.IO.Path]::IsPathRooted($MetricSummaryReportPath)) {
    $MetricSummaryReportPath
} else {
    Join-Path $RepoRoot $MetricSummaryReportPath
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

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{
            present = $false
            data = $null
            error = "missing"
        }
    }
    try {
        $data = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
        return [pscustomobject]@{
            present = $true
            data = $data
            error = $null
        }
    } catch {
        return [pscustomobject]@{
            present = $true
            data = $null
            error = $_.Exception.Message
        }
    }
}

$metricRead = Read-JsonFile -Path $metricFullPath
$stopReasons = @()
$warnings = @()

if ($setExecutionGates.Count -gt 0) {
    $stopReasons += "Execution gates are set during planning: $($setExecutionGates -join ', ')"
}
if (-not $metricRead.present) {
    $stopReasons += "Missing metric summary report: $metricFullPath"
}
if ($metricRead.error) {
    $stopReasons += "Could not read metric summary report: $($metricRead.error)"
}

$metricData = $metricRead.data
$metricSummaryPassed = $false
$sourceRolloutPassed = $false
$diagnosticSuccessRate = $null
$diagnosticSuccessCount = $null
$rewardSum = $null
$sourcePolicyCalls = $null
$sourceTotalSteps = $null

if ($metricData) {
    $metricSummaryPassed = [bool]$metricData.tiny_learned_policy_metric_summary_passed
    $sourceRolloutPassed = [bool]$metricData.metric_summary.source_rollout_passed
    $diagnosticSuccessRate = $metricData.metric_summary.diagnostic_success_rate
    $diagnosticSuccessCount = $metricData.metric_summary.diagnostic_success_count
    $rewardSum = $metricData.metric_summary.reward_sum_total
    $sourcePolicyCalls = $metricData.metric_summary.policy_calls
    $sourceTotalSteps = $metricData.metric_summary.total_steps
}

if ($metricRead.present -and -not $metricSummaryPassed) {
    $stopReasons += "Tiny learned-policy metric summary did not pass."
}
if ($metricRead.present -and -not $sourceRolloutPassed) {
    $stopReasons += "Source tiny learned-policy rollout did not pass."
}

if ($MaxMatrixTasks -gt 5) {
    $warnings += "MaxMatrixTasks was above the first-rung rollout budget; clamping recommendation to 5."
    $MaxMatrixTasks = 5
}
if ($ExpectedRuntimeMinutes -gt 30) {
    $warnings += "ExpectedRuntimeMinutes was above the current budget; clamping recommendation to 30."
    $ExpectedRuntimeMinutes = 30
}
if ($MaxStepsPerTask -gt 20) {
    $warnings += "MaxStepsPerTask was high for the first matrix planner; clamping recommendation to 20."
    $MaxStepsPerTask = 20
}

$decision = "stop"
$reason = "Planner prerequisites are not satisfied."
$readyForReducedScopeRunner = $false
$readyForSmallMatrixRunner = $false
$recommendedTaskCount = 0
$recommendedStepsPerTask = 0
$recommendedRung = "none"

if ($stopReasons.Count -eq 0) {
    $successRateNumber = 0.0
    if ($null -ne $diagnosticSuccessRate) {
        $successRateNumber = [double]$diagnosticSuccessRate
    }

    if ($successRateNumber -le 0.0) {
        $decision = "reduce_scope"
        $reason = "Topology passed, but diagnostic success rate is 0.0. Run one longer single-task diagnostic before a multi-task matrix."
        $readyForReducedScopeRunner = $true
        $readyForSmallMatrixRunner = $false
        $recommendedTaskCount = [Math]::Max(1, $ReducedScopeTasks)
        $recommendedStepsPerTask = $MaxStepsPerTask
        $recommendedRung = "one_task_longer_diagnostic"
    } else {
        $decision = "proceed"
        $reason = "Topology passed and diagnostic success rate is positive; a bounded small learned-policy matrix is inside the current risk budget."
        $readyForReducedScopeRunner = $true
        $readyForSmallMatrixRunner = $true
        $recommendedTaskCount = [Math]::Min([Math]::Max(1, $MaxMatrixTasks), 5)
        $recommendedStepsPerTask = $MaxStepsPerTask
        $recommendedRung = "bounded_small_matrix"
    }
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
    evidence_policy = [ordered]@{
        evidence_label = "bounded_learned_policy_rollout_plan"
        standard_success_claimed = $false
        benchmark_success_claimed = $false
        paper_grade_claim_made = $false
        multi_seed_claim_made = $false
    }
    inputs = [ordered]@{
        metric_summary_report_path = $metricFullPath
        metric_summary_report_present = [bool]$metricRead.present
        metric_summary_report_error = $metricRead.error
        tiny_learned_policy_metric_summary_passed = $metricSummaryPassed
        source_rollout_passed = $sourceRolloutPassed
        source_total_steps = $sourceTotalSteps
        source_policy_calls = $sourcePolicyCalls
        diagnostic_success_count = $diagnosticSuccessCount
        diagnostic_success_rate = $diagnosticSuccessRate
        reward_sum_total = $rewardSum
    }
    risk_assessment = [ordered]@{
        task = "bounded learned-policy rollout matrix planning"
        command = "future runner gated by ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1"
        source = "local SmolVLA checkpoint, local official LIBERO/RoboSuite simulator/data, existing metric summary"
        expected_size_gb = 0
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 12
        expected_vram_gb = 0
        max_matrix_tasks = $MaxMatrixTasks
        reduced_scope_tasks = $ReducedScopeTasks
        recommended_task_count = $recommendedTaskCount
        max_steps_per_task = $MaxStepsPerTask
        recommended_steps_per_task = $recommendedStepsPerTask
        token_login_license_payment_needed = $false
        simulator_will_run_in_future_runner = $true
        learned_policy_inference_will_run_in_future_runner = $true
        training_will_run = $false
        openvla_oft_will_run = $false
        paper_claim_will_be_made = $false
    }
    dangerous_execution_gates_set = $setExecutionGates
    warnings = $warnings
    stop_reasons = $stopReasons
    decision = $decision
    reason = $reason
    recommended_rung = $recommendedRung
    ready_for_reduced_scope_learned_policy_runner = $readyForReducedScopeRunner
    ready_for_bounded_small_learned_policy_matrix_runner = $readyForSmallMatrixRunner
    recommended_next_step = if ($decision -eq "reduce_scope") {
        "Create a separately gated one-task longer diagnostic runner using ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1; keep evidence diagnostic/local-pilot only."
    } elseif ($decision -eq "proceed") {
        "Create a separately gated bounded small learned-policy matrix runner; keep task count, steps, runtime, and evidence labels capped."
    } else {
        "Regenerate the tiny learned-policy rollout and metric summary before planning a larger learned-policy rollout."
    }
}

$json = $report | ConvertTo-Json -Depth 8
$jsonParent = Split-Path -Parent $jsonFullPath
$mdParent = Split-Path -Parent $markdownFullPath
New-Item -ItemType Directory -Force -Path $jsonParent | Out-Null
New-Item -ItemType Directory -Force -Path $mdParent | Out-Null
Set-Content -LiteralPath $jsonFullPath -Value $json -Encoding UTF8

$md = @(
    "# Bounded Learned-Policy Rollout Matrix Plan Report",
    "",
    "- decision: $decision",
    "- reason: $reason",
    "- recommended rung: $recommendedRung",
    "- ready for reduced-scope runner: $readyForReducedScopeRunner",
    "- ready for bounded small matrix runner: $readyForSmallMatrixRunner",
    "- source rollout passed: $sourceRolloutPassed",
    "- diagnostic success rate: $diagnosticSuccessRate",
    "- reward sum: $rewardSum",
    "- recommended task count: $recommendedTaskCount",
    "- recommended steps per task: $recommendedStepsPerTask",
    "- standard success claimed: false",
    "- benchmark success claimed: false",
    "- paper-grade claim made: false",
    "",
    $report.recommended_next_step,
    ""
)
Set-Content -LiteralPath $markdownFullPath -Value $md -Encoding UTF8

Write-Host $json
