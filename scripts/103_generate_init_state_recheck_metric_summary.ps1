param(
    [string]$InitStateReportPath = "reports\init_state_learned_policy_recheck_report.json",
    [string]$TinyResetReportPath = "reports\tiny_learned_policy_rollout_report.json",
    [string]$ReducedResetReportPath = "reports\bounded_reduced_scope_learned_policy_rollout_report.json",
    [string]$JsonReportPath = "reports\init_state_recheck_metric_summary_report.json",
    [string]$MarkdownReportPath = "reports\init_state_recheck_metric_summary_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Init-state learned-policy recheck metric summary"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is summary-only. It does not load models, run inference, create simulator environments, rollout, train, use GPU, download, execute OpenVLA-OFT, access tokens, or make paper claims."

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

function Get-TaskMetric {
    param(
        [object]$Report,
        [string]$Name,
        [string]$PassedField,
        [bool]$UsesHdf5InitState
    )
    $task = $null
    if ($null -ne $Report.rollout_result -and $null -ne $Report.rollout_result.tasks -and $Report.rollout_result.tasks.Count -gt 0) {
        $task = $Report.rollout_result.tasks[0]
    }
    $result = if ($null -ne $Report.rollout_result) { $Report.rollout_result.result } else { $null }
    $policy = if ($null -ne $Report.policy) { $Report.policy } else { $null }
    $passed = $false
    if ($null -ne $Report.PSObject.Properties[$PassedField]) {
        $passed = [bool]$Report.$PassedField
    } elseif ($null -ne $result -and $null -ne $result.PSObject.Properties["passed"]) {
        $passed = [bool]$result.passed
    }
    $reward = if ($null -ne $task -and $null -ne $task.PSObject.Properties["reward_sum"]) { [double]$task.reward_sum } else { $null }
    $success = if ($null -ne $task -and $null -ne $task.PSObject.Properties["success_check"]) { [bool]$task.success_check } else { $false }
    $steps = if ($null -ne $result -and $null -ne $result.PSObject.Properties["total_steps_performed"]) { [int]$result.total_steps_performed } elseif ($null -ne $task -and $null -ne $task.PSObject.Properties["steps_performed"]) { [int]$task.steps_performed } else { 0 }
    $policyCalls = if ($null -ne $task -and $null -ne $task.PSObject.Properties["policy_calls"]) { [int]$task.policy_calls } else { 0 }
    $adapterStrategy = $null
    if ($null -ne $policy -and $null -ne $policy.PSObject.Properties["action_adapter_strategy"]) {
        $adapterStrategy = [string]$policy.action_adapter_strategy
    } elseif ($null -ne $task -and $null -ne $task.PSObject.Properties["last_action_adapter_metadata"] -and $null -ne $task.last_action_adapter_metadata.PSObject.Properties["strategy"]) {
        $adapterStrategy = [string]$task.last_action_adapter_metadata.strategy
    } elseif ($null -ne $task -and $null -ne $task.PSObject.Properties["last_adapter_metadata"] -and $null -ne $task.last_adapter_metadata.action_adapter.PSObject.Properties["strategy"]) {
        $adapterStrategy = [string]$task.last_adapter_metadata.action_adapter.strategy
    }
    return [ordered]@{
        name = $Name
        source_passed = $passed
        uses_hdf5_init_state = $UsesHdf5InitState
        hdf5_init_state_set = if ($UsesHdf5InitState -and $null -ne $Report.policy -and $null -ne $Report.policy.PSObject.Properties["hdf5_init_state_set_in_environment"]) { [bool]$Report.policy.hdf5_init_state_set_in_environment } else { $false }
        task_name = if ($null -ne $task -and $null -ne $task.PSObject.Properties["task_name"]) { [string]$task.task_name } else { $null }
        steps = $steps
        policy_calls = $policyCalls
        diagnostic_success = $success
        reward_sum = $reward
        done_seen = if ($null -ne $task -and $null -ne $task.PSObject.Properties["done_seen"]) { [bool]$task.done_seen } else { $false }
        action_adapter_strategy = $adapterStrategy
        last_env_action_preview = if ($null -ne $task -and $null -ne $task.PSObject.Properties["last_env_action_preview"]) { @($task.last_env_action_preview) } else { @() }
        error = if ($null -ne $task -and $null -ne $task.PSObject.Properties["error"]) { $task.error } else { $null }
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
        "# Init-State Recheck Metric Summary Report",
        "",
        "- decision: $($Report.decision)",
        "- summary passed: $($Report.init_state_recheck_metric_summary_passed)",
        "- positive diagnostic signal found: $($Report.metric_summary.positive_diagnostic_signal_found)",
        "- init-state reward sum: $($Report.metric_summary.init_state_reward_sum)",
        "- init-state diagnostic success: $($Report.metric_summary.init_state_diagnostic_success)",
        "- rollout scaling ready: $($Report.ready_for_rollout_scaling)",
        "- paper claim ready: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is summary-only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 14
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
    "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK",
    "ALLOW_OPENVLA_OFT"
)
$setExecutionGates = @($executionGates | Where-Object { -not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_)) })

$initRead = Read-JsonFileIfPresent -Path $InitStateReportPath
$tinyRead = Read-JsonFileIfPresent -Path $TinyResetReportPath
$reducedRead = Read-JsonFileIfPresent -Path $ReducedResetReportPath

$stopReasons = New-Object System.Collections.Generic.List[string]
if ($setExecutionGates.Count -gt 0) {
    $stopReasons.Add("summary-only init-state recheck metric report refuses execution gates: $($setExecutionGates -join ', ')")
}
if (-not $initRead.present -or $initRead.error -or $null -eq $initRead.data) {
    $stopReasons.Add("Missing or unreadable init-state recheck report: $($initRead.path)")
}
if (-not $tinyRead.present -or $tinyRead.error -or $null -eq $tinyRead.data) {
    $stopReasons.Add("Missing or unreadable tiny reset-only learned-policy report: $($tinyRead.path)")
}
if (-not $reducedRead.present -or $reducedRead.error -or $null -eq $reducedRead.data) {
    $stopReasons.Add("Missing or unreadable reduced-scope reset-only learned-policy report: $($reducedRead.path)")
}

$scenarios = @()
if ($stopReasons.Count -eq 0) {
    $scenarios += Get-TaskMetric -Report $tinyRead.data -Name "reset_only_3_step" -PassedField "tiny_learned_policy_rollout_passed" -UsesHdf5InitState $false
    $scenarios += Get-TaskMetric -Report $reducedRead.data -Name "reset_only_10_step" -PassedField "bounded_reduced_scope_learned_policy_rollout_passed" -UsesHdf5InitState $false
    $scenarios += Get-TaskMetric -Report $initRead.data -Name "hdf5_init_state_3_step" -PassedField "bounded_init_state_learned_policy_recheck_passed" -UsesHdf5InitState $true
}

$positiveSignal = $false
foreach ($scenario in $scenarios) {
    if ([bool]$scenario.diagnostic_success -or ($null -ne $scenario.reward_sum -and [double]$scenario.reward_sum -gt 0.0)) {
        $positiveSignal = $true
    }
}
$initScenario = $scenarios | Where-Object { $_.name -eq "hdf5_init_state_3_step" } | Select-Object -First 1
$resetTiny = $scenarios | Where-Object { $_.name -eq "reset_only_3_step" } | Select-Object -First 1
$initReward = if ($null -ne $initScenario) { $initScenario.reward_sum } else { $null }
$resetReward = if ($null -ne $resetTiny) { $resetTiny.reward_sum } else { $null }
$rewardDelta = if ($null -ne $initReward -and $null -ne $resetReward) { [double]$initReward - [double]$resetReward } else { $null }

$summaryPassed = [bool]($stopReasons.Count -eq 0)
$decision = if (-not $summaryPassed) {
    "stop"
} elseif ($positiveSignal) {
    "proceed_with_caution"
} else {
    "no_go_rollout_scaling"
}

$report = [ordered]@{
    evidence_label = "init_state_learned_policy_metric_summary"
    init_state_recheck_metric_summary_passed = $summaryPassed
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
    source_reports = [ordered]@{
        init_state_recheck = $initRead.path
        tiny_reset_only = $tinyRead.path
        reduced_reset_only = $reducedRead.path
    }
    metric_summary = [ordered]@{
        scenarios = @($scenarios)
        positive_diagnostic_signal_found = $positiveSignal
        init_state_reward_sum = $initReward
        init_state_diagnostic_success = if ($null -ne $initScenario) { [bool]$initScenario.diagnostic_success } else { $false }
        reset_only_3_step_reward_sum = $resetReward
        init_state_vs_reset_3_step_reward_delta = $rewardDelta
        hdf5_init_state_set_in_environment = if ($null -ne $initScenario) { [bool]$initScenario.hdf5_init_state_set } else { $false }
    }
    dangerous_execution_gates_set = @($setExecutionGates)
    stop_reasons = @($stopReasons)
    ready_for_rollout_scaling = [bool]($summaryPassed -and $positiveSignal)
    ready_for_benchmark_claim = $false
    ready_for_paper_claim = $false
    decision = $decision
    reason = if ($summaryPassed) {
        if ($positiveSignal) { "At least one diagnostic scenario had nonzero reward or success; still not paper-grade." } else { "Init-state recheck executed but reward and success remain zero; rollout scaling remains blocked." }
    } else {
        $stopReasons -join "; "
    }
    recommended_next_step = if ($summaryPassed -and -not $positiveSignal) {
        "Stop rollout scaling. Inspect checkpoint/task alignment, VLM loading policy, and offline demonstration-conditioned action decoding before more learned-policy rollouts."
    } elseif ($summaryPassed) {
        "Plan a narrowly bounded follow-up diagnostic with explicit evidence labels; still no paper claims."
    } else {
        "Resolve missing summary input reports before any further rollout decision."
    }
}

Write-Reports -Report $report
exit 0
