param(
    [string]$CameraSourceReportPath = "reports\camera_source_diagnostic_report.json",
    [string]$RolloutBridgeSourcePath = "tca_map\smolvla\libero_learned_policy_rollout.py",
    [string]$JsonReportPath = "reports\state_sufficiency_diagnostic_plan_report.json",
    [string]$MarkdownReportPath = "reports\state_sufficiency_diagnostic_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "State-sufficiency diagnostic planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This planner reads existing reports and source files only. It does not download, install, load models, infer, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Read-JsonFile {
    param([string]$Path)
    $fullPath = Resolve-RepoPath -Path $Path
    $text = [System.IO.File]::ReadAllText($fullPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    return $text | ConvertFrom-Json
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

$cameraReportPathResolved = Resolve-RepoPath -Path $CameraSourceReportPath
$sourcePathResolved = Resolve-RepoPath -Path $RolloutBridgeSourcePath
$jsonOut = Resolve-RepoPath -Path $JsonReportPath
$mdOut = Resolve-RepoPath -Path $MarkdownReportPath

$stopReasons = @()
$cameraReport = $null
if (-not (Test-Path -LiteralPath $cameraReportPathResolved)) {
    $stopReasons += "Missing input report: $cameraReportPathResolved"
} else {
    try {
        $cameraReport = Read-JsonFile -Path $cameraReportPathResolved
    } catch {
        $stopReasons += "Could not read ${cameraReportPathResolved}: $($_.Exception.Message)"
    }
}
if (-not (Test-Path -LiteralPath $sourcePathResolved)) {
    $stopReasons += "Missing rollout bridge source: $sourcePathResolved"
}
if ($setExecutionGates.Count -gt 0) {
    $stopReasons += "Refusing state-sufficiency planning while execution gates are set: $($setExecutionGates -join ', ')"
}

$sourceText = ""
if (Test-Path -LiteralPath $sourcePathResolved) {
    $sourceText = [System.IO.File]::ReadAllText($sourcePathResolved, [System.Text.Encoding]::UTF8)
}

$cameraPassed = $false
if ($null -ne $cameraReport) {
    $cameraPassed = [bool]$cameraReport.camera_source_diagnostic_passed
}
$sourceHasCli = $sourceText.Contains("--state-adapter-strategy") -and $sourceText.Contains("args.state_adapter_strategy")
$sourceHasStrategies = $sourceText.Contains("STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_LAST3") -and $sourceText.Contains("STATE_ADAPTER_STRATEGY_EEF_POS_ZERO_ROT")

if (-not $cameraPassed) {
    $stopReasons += "Camera-source diagnostic has not passed."
}
if (-not $sourceHasCli) {
    $stopReasons += "Rollout bridge does not expose the state-adapter-strategy CLI and batch hook."
}
if (-not $sourceHasStrategies) {
    $stopReasons += "Rollout bridge does not expose state-sufficiency strategy helpers."
}

$ready = $stopReasons.Count -eq 0
$report = [ordered]@{
    state_sufficiency_diagnostic_plan_passed = $ready
    decision = if ($ready) { "proceed" } else { "stop" }
    reason = if ($ready) { "Camera-source diagnostic passed with zero reward; bounded state-sufficiency diagnostic runner is ready." } else { "State-sufficiency diagnostic prerequisites are not satisfied." }
    source_reports = [ordered]@{
        camera_source_diagnostic = $cameraReportPathResolved
        rollout_bridge_source = $sourcePathResolved
    }
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
    claims = [ordered]@{
        standard_success_claimed = $false
        benchmark_success_claimed = $false
        counterfactual_robustness_claimed = $false
        sota_claimed = $false
        paper_grade_claim_made = $false
    }
    inputs = [ordered]@{
        camera_source_diagnostic_passed = $cameraPassed
        variants_completed = if ($null -ne $cameraReport) { $cameraReport.result.variants_completed } else { $null }
        best_camera_alias_strategy = if ($null -ne $cameraReport) { $cameraReport.result.best_camera_alias_strategy } else { $null }
        best_diagnostic_success_rate = if ($null -ne $cameraReport) { $cameraReport.result.best_diagnostic_success_rate } else { $null }
        best_reward_sum = if ($null -ne $cameraReport) { $cameraReport.result.best_reward_sum } else { $null }
        source_has_state_strategy_cli = $sourceHasCli
        source_has_state_strategy_helpers = $sourceHasStrategies
    }
    risk_assessment = [ordered]@{
        task = "bounded state-sufficiency diagnostic"
        source = "local SmolVLA checkpoint and local LIBERO/RoboSuite WSL simulator topology"
        expected_size_gb = 0
        expected_runtime_minutes = 15
        expected_ram_gb = 8
        expected_vram_gb = 0
        task_count = 1
        max_steps_per_variant = 10
        token_login_license_payment_needed = $false
        simulator_will_run_in_future_runner = $true
        learned_policy_inference_will_run_in_future_runner = $true
        training_will_run = $false
        openvla_oft_will_run = $false
        paper_claim_will_be_made = $false
    }
    diagnostic_plan = [ordered]@{
        evidence_label = "state_sufficiency_diagnostic_plan"
        max_tasks = 1
        max_steps_per_variant = 10
        max_variants_first_runner = 3
        expected_runtime_minutes = 15
        expected_vram_gb = 0
        action_adapter_strategy = "policy_6d_delta_pose_plus_gripper_zero_hold"
        action_scale = 1.0
        prompt_strategy = "bddl_language"
        camera_alias_strategy = "current_aliases"
        state_adapter_strategy_variants = @("eef_pos_quat_first3", "eef_pos_quat_last3", "eef_pos_zero_rot")
        acceptance_checks = @(
            "no downloads",
            "no installs",
            "no training",
            "no GPU jobs",
            "no OpenVLA-OFT",
            "one task only",
            "at most 10 steps per variant",
            "state_adapter_strategy and state adapter metadata recorded for every variant",
            "results labeled diagnostic only"
        )
    }
    stop_reasons = $stopReasons
    ready_for_state_sufficiency_diagnostic_runner = $ready
    ready_for_rollout_scaling = $false
    recommended_next_step = if ($ready) { "Run a separately gated one-task state-sufficiency diagnostic runner; do not scale rollout or make claims." } else { "Fix missing state-sufficiency diagnostic inputs before running the bounded diagnostic." }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonOut) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $mdOut) | Out-Null
$json = $report | ConvertTo-Json -Depth 12
$json | Set-Content -LiteralPath $jsonOut -Encoding UTF8
$md = @(
    "# State-Sufficiency Diagnostic Plan Report",
    "",
    "- decision: $($report.decision)",
    "- planner passed: $($report.state_sufficiency_diagnostic_plan_passed)",
    "- ready for runner: $($report.ready_for_state_sufficiency_diagnostic_runner)",
    "- ready for rollout scaling: $($report.ready_for_rollout_scaling)",
    "- source camera-source passed: $cameraPassed",
    "- planned state variants: $($report.diagnostic_plan.state_adapter_strategy_variants -join ', ')",
    "",
    $report.recommended_next_step,
    "",
    "This is planning evidence only. It is not benchmark success, standard success, SOTA evidence, or paper-grade evidence."
)
$md -join "`n" | Set-Content -LiteralPath $mdOut -Encoding UTF8
Write-Output $json
