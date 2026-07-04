param(
    [string]$MetricSummaryReportPath = "reports\reduced_scope_rollout_metric_summary_report.json",
    [string]$SmolVlaCkptPath = "C:\assets\checkpoints\smolvla",
    [string]$JsonReportPath = "reports\action_interface_diagnostic_plan_report.json",
    [string]$MarkdownReportPath = "reports\action_interface_diagnostic_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Action-interface diagnostic planner"
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

$metricFullPath = if ([System.IO.Path]::IsPathRooted($MetricSummaryReportPath)) { $MetricSummaryReportPath } else { Join-Path $RepoRoot $MetricSummaryReportPath }
$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
$configPath = Join-Path $SmolVlaCkptPath "config.json"

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{ present = $false; data = $null; error = "missing" }
    }
    try {
        $data = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
        return [pscustomobject]@{ present = $true; data = $data; error = $null }
    } catch {
        return [pscustomobject]@{ present = $true; data = $null; error = $_.Exception.Message }
    }
}

$metricRead = Read-JsonFile -Path $metricFullPath
$configRead = Read-JsonFile -Path $configPath
$stopReasons = @()
$warnings = @()

if ($setExecutionGates.Count -gt 0) {
    $stopReasons += "Execution gates are set during planning: $($setExecutionGates -join ', ')"
}
if (-not $metricRead.present) {
    $stopReasons += "Missing reduced-scope metric summary report: $metricFullPath"
}
if ($metricRead.error) {
    $stopReasons += "Could not read reduced-scope metric summary report: $($metricRead.error)"
}
if ($configRead.error -and $configRead.present) {
    $warnings += "Could not parse SmolVLA config metadata: $($configRead.error)"
}
if (-not $configRead.present) {
    $warnings += "SmolVLA config metadata not found at $configPath; continuing with rollout metrics only."
}

$metric = $metricRead.data
$summary = if ($metric) { $metric.metric_summary } else { $null }
$summaryPassed = if ($metric) { [bool]$metric.reduced_scope_rollout_metric_summary_passed } else { $false }
$successRate = if ($summary) { $summary.diagnostic_success_rate } else { $null }
$rewardSum = if ($summary) { $summary.reward_sum_total } else { $null }
$policyShape = if ($summary -and $summary.policy_action_shapes -and $summary.policy_action_shapes.Count -gt 0) { @($summary.policy_action_shapes[0]) } else { @() }
$policyActionDim = if ($policyShape.Count -ge 2) { [int]$policyShape[1] } else { $null }
$envDims = if ($summary -and $summary.env_action_dims) { @($summary.env_action_dims) } else { @() }
$envActionDim = if ($envDims.Count -gt 0) { [int]$envDims[0] } else { $null }
$gripper = if ($summary) { $summary.last_env_action_gripper_component } else { $null }
$actionMaxAbs = if ($summary) { $summary.last_env_action_max_abs } else { $null }
$actionL2 = if ($summary) { $summary.last_env_action_l2 } else { $null }

if ($metricRead.present -and -not $summaryPassed) {
    $stopReasons += "Reduced-scope metric summary did not pass."
}

$dimMismatch = ($null -ne $policyActionDim -and $null -ne $envActionDim -and $policyActionDim -ne $envActionDim)
$gripperPaddedZero = ($dimMismatch -and $null -ne $gripper -and [double]$gripper -eq 0.0)
$noTaskSuccess = ($null -ne $successRate -and [double]$successRate -eq 0.0)
$nontrivialActionMagnitude = ($null -ne $actionMaxAbs -and [double]$actionMaxAbs -gt 0.05)

$diagnostics = @(
    [ordered]@{
        name = "action_dimension_and_gripper_mapping"
        priority = if ($dimMismatch -or $gripperPaddedZero) { "high" } else { "medium" }
        reason = "Policy action dimension and environment action dimension differ, and gripper may be padded."
        evidence = [ordered]@{ policy_action_dim = $policyActionDim; env_action_dim = $envActionDim; gripper_component = $gripper }
        next_check = "Audit whether SmolVLA outputs gripper separately or expects a 6D delta pose only."
    },
    [ordered]@{
        name = "action_normalization_and_scale"
        priority = if ($nontrivialActionMagnitude -and $noTaskSuccess) { "high" } else { "medium" }
        reason = "Actions are nontrivial but the task remains unsolved."
        evidence = [ordered]@{ action_max_abs = $actionMaxAbs; action_l2 = $actionL2; reward_sum = $rewardSum }
        next_check = "Inspect unnormalizer/postprocessor metadata and compare action ranges to LIBERO action semantics."
    },
    [ordered]@{
        name = "observation_state_mapping"
        priority = "high"
        reason = "State vector was assembled from available LIBERO observation keys and may not match SmolVLA training state semantics."
        evidence = [ordered]@{ expected_policy_state_dim = 6 }
        next_check = "Audit state key order, units, and whether proprioception should include gripper/joint state differently."
    },
    [ordered]@{
        name = "camera_mapping"
        priority = "medium"
        reason = "Camera feature names are mapped heuristically from LIBERO observations."
        evidence = [ordered]@{ camera_size = 64 }
        next_check = "Confirm camera order, image orientation, channel order, and whether wrist/agent views match policy expectations."
    },
    [ordered]@{
        name = "language_prompt_mapping"
        priority = "medium"
        reason = "Language is derived from BDDL filename words rather than a canonical dataset instruction field."
        evidence = [ordered]@{ prompt_source = "BDDL filename stem" }
        next_check = "Compare prompt text against LIBERO task language and SmolVLA expected prompt template."
    },
    [ordered]@{
        name = "zero_action_vs_smolvla_action_comparison"
        priority = "medium"
        reason = "A small controlled comparison can separate simulator/task difficulty from policy-action interface issues."
        evidence = [ordered]@{ current_success_rate = $successRate; current_reward_sum = $rewardSum }
        next_check = "Run a bounded zero-action versus SmolVLA-action diagnostic comparison if this plan remains green."
    }
)

$decision = "stop"
$reason = "Action-interface diagnostic prerequisites are not satisfied."
$readyForActionInterfaceAudit = $false
$readyForZeroVsPolicyComparison = $false
if ($stopReasons.Count -eq 0) {
    $decision = "proceed"
    $reason = "Reduced-scope rollout summary is available; action-interface diagnostics are the next safe step before scaling."
    $readyForActionInterfaceAudit = $true
    $readyForZeroVsPolicyComparison = $true
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
        evidence_label = "action_interface_diagnostic_plan"
        standard_success_claimed = $false
        benchmark_success_claimed = $false
        paper_grade_claim_made = $false
    }
    inputs = [ordered]@{
        reduced_scope_metric_summary_path = $metricFullPath
        reduced_scope_metric_summary_present = [bool]$metricRead.present
        reduced_scope_metric_summary_passed = $summaryPassed
        smolvla_config_path = $configPath
        smolvla_config_present = [bool]$configRead.present
    }
    observed_signals = [ordered]@{
        diagnostic_success_rate = $successRate
        reward_sum_total = $rewardSum
        policy_action_dim = $policyActionDim
        env_action_dim = $envActionDim
        action_dim_mismatch = $dimMismatch
        gripper_component = $gripper
        gripper_padded_zero = $gripperPaddedZero
        action_max_abs = $actionMaxAbs
        action_l2 = $actionL2
        nontrivial_action_magnitude = $nontrivialActionMagnitude
    }
    diagnostics = $diagnostics
    dangerous_execution_gates_set = $setExecutionGates
    warnings = $warnings
    stop_reasons = $stopReasons
    decision = $decision
    reason = $reason
    ready_for_action_interface_audit = $readyForActionInterfaceAudit
    ready_for_zero_action_vs_policy_action_diagnostic = $readyForZeroVsPolicyComparison
    recommended_next_step = if ($decision -eq "proceed") {
        "Create a bounded action-interface audit or zero-action-vs-policy diagnostic. Keep evidence diagnostic/local-pilot only."
    } else {
        "Regenerate the reduced-scope rollout metric summary before action-interface diagnostics."
    }
}

$json = $report | ConvertTo-Json -Depth 8
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
Set-Content -LiteralPath $jsonFullPath -Value $json -Encoding UTF8

$md = @(
    "# Action-Interface Diagnostic Plan Report",
    "",
    "- decision: $decision",
    "- reason: $reason",
    "- diagnostic success rate: $successRate",
    "- reward sum: $rewardSum",
    "- policy action dim: $policyActionDim",
    "- env action dim: $envActionDim",
    "- action dim mismatch: $dimMismatch",
    "- gripper component: $gripper",
    "- gripper padded zero: $gripperPaddedZero",
    "- action max abs: $actionMaxAbs",
    "- action L2: $actionL2",
    "- ready for action-interface audit: $readyForActionInterfaceAudit",
    "- ready for zero-action vs policy diagnostic: $readyForZeroVsPolicyComparison",
    "- standard success claimed: false",
    "- paper-grade claim made: false",
    "",
    $report.recommended_next_step,
    ""
)
Set-Content -LiteralPath $markdownFullPath -Value $md -Encoding UTF8

Write-Host $json
