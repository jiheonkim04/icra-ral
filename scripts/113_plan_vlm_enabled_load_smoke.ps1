param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$AcquisitionReportPath = "reports\vlm_required_files_acquisition_report.json",
    [string]$RiskReportPath = "reports\vlm_enabled_loading_risk_plan_report.json",
    [string]$JsonReportPath = "reports\vlm_enabled_load_smoke_plan_report.json",
    [string]$MarkdownReportPath = "reports\vlm_enabled_load_smoke_plan_report.md",
    [double]$ExpectedRamGb = 16.0,
    [double]$MinTotalRamGb = 20.0,
    [int]$MaxRuntimeMinutes = 15
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Resolve-RepoPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function ConvertFrom-JsonOutput {
    param([string]$Text)
    $start = $Text.IndexOf("{")
    if ($start -lt 0) {
        throw "No JSON object found in checker output."
    }
    return $Text.Substring($start) | ConvertFrom-Json
}

function Read-JsonFile {
    param([string]$Path)
    $resolved = Resolve-RepoPath -Path $Path
    if (-not (Test-Path -LiteralPath $resolved)) {
        return $null
    }
    return Get-Content -Raw -LiteralPath $resolved -Encoding UTF8 | ConvertFrom-Json
}

function Get-RamInfo {
    try {
        $system = Get-CimInstance Win32_ComputerSystem
        $os = Get-CimInstance Win32_OperatingSystem
        return [ordered]@{
            total_ram_gb = [math]::Round(($system.TotalPhysicalMemory / 1GB), 3)
            free_ram_gb = [math]::Round(($os.FreePhysicalMemory * 1KB / 1GB), 3)
        }
    } catch {
        return [ordered]@{
            total_ram_gb = $null
            free_ram_gb = $null
        }
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$executionGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_VLM_ENABLED_LOAD_SMOKE",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
    "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
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

$jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
$markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null

$adapterChecker = Join-Path $RepoRoot "scripts\13_check_smolvla_adapter_smoke.ps1"
$runtimeChecker = Join-Path $RepoRoot "scripts\17_check_smolvla_runtime_deps.ps1"
$adapterOutput = & powershell -ExecutionPolicy Bypass -File $adapterChecker -PathsFile $PathsFile -Python $Python 2>&1 | Out-String
$runtimeOutput = & powershell -ExecutionPolicy Bypass -File $runtimeChecker -Python $Python 2>&1 | Out-String
$adapterReport = ConvertFrom-JsonOutput -Text $adapterOutput
$runtimeReport = ConvertFrom-JsonOutput -Text $runtimeOutput
$acquisitionReport = Read-JsonFile -Path $AcquisitionReportPath
$riskReport = Read-JsonFile -Path $RiskReportPath
$ram = Get-RamInfo

$stopReasons = @()
if ($setExecutionGates.Count -gt 0) {
    $stopReasons += "Planning-only script refuses execution gates: $($setExecutionGates -join ', ')"
}
if ($null -eq $acquisitionReport) {
    $stopReasons += "Missing VLM required-file acquisition report: $AcquisitionReportPath"
} elseif (-not [bool]$acquisitionReport.vlm_required_files_acquisition_passed) {
    $stopReasons += "VLM required-file acquisition report did not pass."
}
if ($null -eq $riskReport) {
    $stopReasons += "Missing VLM-enabled loading risk report: $RiskReportPath"
} elseif (-not [bool]$riskReport.ready_for_vlm_weight_acquisition_plan) {
    $stopReasons += "VLM-enabled loading risk report did not authorize the dependency file acquisition path."
}
if (-not [bool]$adapterReport.ready_for_smolvla_adapter_smoke) {
    $stopReasons += "SmolVLA adapter readiness checker is not ready."
}
if (-not [bool]$runtimeReport.runtime_dependencies.ready_for_load_only_runtime) {
    $stopReasons += "SmolVLA runtime dependency checker is not ready."
}
if ($null -ne $ram.total_ram_gb -and [double]$ram.total_ram_gb -lt $MinTotalRamGb) {
    $stopReasons += "Total system RAM is below the VLM-enabled load-smoke planning minimum: $($ram.total_ram_gb)GB < $MinTotalRamGb GB"
}

$acqRisk = if ($null -ne $acquisitionReport) { $acquisitionReport.risk_assessment } else { $null }
$targetPath = if ($null -ne $acqRisk) { $acqRisk.target_path } else { "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct" }
$sourceRepo = if ($null -ne $acqRisk) { $acqRisk.source_repo } else { "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" }
$sourceUrl = if ($null -ne $acqRisk) { $acqRisk.source_url } else { "https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct" }
$expectedDiskGb = if ($null -ne $acqRisk) { $acqRisk.expected_new_disk_gb } else { $null }
$decision = if ($stopReasons.Count -eq 0) { "proceed" } else { "stop" }
$passed = $decision -eq "proceed"

$report = [ordered]@{
    evidence_label = "vlm_enabled_load_smoke_plan"
    vlm_enabled_load_smoke_plan_passed = $passed
    decision = $decision
    ready_for_bounded_vlm_enabled_load_smoke_runner = $passed
    ready_for_rollout_scaling = $false
    ready_for_benchmark_claim = $false
    ready_for_paper_claim = $false
    policy = [ordered]@{
        plan_only = $true
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
        execution_gates_set = $setExecutionGates
    }
    claims = [ordered]@{
        standard_success_claimed = $false
        benchmark_success_claimed = $false
        counterfactual_robustness_claimed = $false
        sota_claimed = $false
        paper_grade_claim_made = $false
    }
    prerequisites = [ordered]@{
        adapter_readiness = $adapterReport.ready_for_smolvla_adapter_smoke
        runtime_ready = $runtimeReport.runtime_dependencies.ready_for_load_only_runtime
        vlm_required_files_acquired = $(if ($null -ne $acquisitionReport) { $acquisitionReport.vlm_required_files_acquisition_passed } else { $false })
        vlm_acquisition_decision = $(if ($null -ne $acquisitionReport) { $acquisitionReport.decision } else { $null })
        vlm_risk_decision = $(if ($null -ne $riskReport) { $riskReport.decision } else { $null })
    }
    risk_assessment = [ordered]@{
        task = "Plan bounded VLM-enabled SmolVLA load-only smoke"
        command = "scripts\114_bounded_vlm_enabled_load_smoke.ps1"
        source_repo = $sourceRepo
        source_url = $sourceUrl
        source_official_documented = $true
        local_vlm_dependency_path = $targetPath
        expected_disk_gb = $expectedDiskGb
        expected_runtime_minutes = $MaxRuntimeMinutes
        expected_ram_gb = $ExpectedRamGb
        expected_vram_gb = 0
        total_ram_gb = $ram.total_ram_gb
        free_ram_gb = $ram.free_ram_gb
        simulator_will_run = $false
        rollout_will_run = $false
        training_will_run = $false
        model_load_in_this_planner = $false
        future_model_load_policy = "load_vlm_weights=true, CPU first, no inference, no rollout"
        required_future_gates = @("ALLOW_HEAVY_IMPORT=1", "ALLOW_VLM_ENABLED_LOAD_SMOKE=1")
        stop_condition = "Stop if RAM/runtime estimate is red, VLM file acquisition is incomplete, runtime dependencies are missing, token/license/payment is required, OpenVLA-OFT is needed, or the future load exceeds budget."
        fallback_plan = "Keep load_vlm_weights=false diagnostics and inspect action normalization/checkpoint provenance if VLM-enabled load is unsafe."
        decision = $decision
        reason = $(if ($passed) { "Prerequisites are present, system RAM meets the planning threshold, VLM files are local, and the future task can be bounded to CPU load-only with no inference." } else { $stopReasons -join "; " })
    }
    future_runner_scope = [ordered]@{
        script_to_create = "scripts\114_bounded_vlm_enabled_load_smoke.ps1"
        device = "cpu"
        load_vlm_weights = $true
        max_runtime_minutes = $MaxRuntimeMinutes
        expected_no_inference = $true
        expected_no_training = $true
        expected_no_rollout = $true
        expected_no_gpu_job = $true
        expected_no_openvla_oft = $true
        expected_no_tokens = $true
        evidence_level = "engineering load smoke only"
    }
    stop_reasons = $stopReasons
    recommended_next_step = $(if ($passed) { "Create a separately gated bounded VLM-enabled load-smoke runner. Do not run inference, rollout, training, GPU jobs, or OpenVLA-OFT." } else { "Resolve the listed blockers before creating or running a VLM-enabled load-smoke runner." })
}

$report | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
$md = @(
    "# VLM-Enabled Load Smoke Plan Report",
    "",
    "- decision: $decision",
    "- plan passed: $passed",
    "- ready for bounded VLM-enabled load-smoke runner: $($report.ready_for_bounded_vlm_enabled_load_smoke_runner)",
    "- source: $sourceRepo",
    "- local VLM dependency path: $targetPath",
    "- expected RAM GB: $ExpectedRamGb",
    "- total/free RAM GB: $($ram.total_ram_gb) / $($ram.free_ram_gb)",
    "- expected VRAM GB: 0",
    "- required future gates: ALLOW_HEAVY_IMPORT=1, ALLOW_VLM_ENABLED_LOAD_SMOKE=1",
    "",
    "This planner did not download, install, load models, run inference, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.",
    "",
    "## Recommended Next Step",
    "",
    $report.recommended_next_step
)
$md -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

Write-Host "VLM-enabled load smoke plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not load models, run inference, train, rollout, download assets, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."
$report | ConvertTo-Json -Depth 16

if ($setExecutionGates.Count -gt 0) {
    exit 2
}
exit 0
