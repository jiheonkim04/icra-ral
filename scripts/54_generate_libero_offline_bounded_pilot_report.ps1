param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$InterfaceReportPath = "reports\libero_offline_interface_smoke_report.json",
    [string]$SplitReportPath = "reports\libero_offline_counterfactual_split_report.json",
    [string]$HeadReportPath = "reports\libero_offline_actionmap_tca_comparison_report.json",
    [string]$LoraReportPath = "reports\libero_offline_lora_comparison_report.json",
    [string]$JsonReportPath = "reports\libero_offline_bounded_pilot_report.json",
    [string]$MarkdownReportPath = "reports\libero_offline_bounded_pilot_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO offline bounded pilot report"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads existing runtime reports only. It does not download assets, run GPU jobs, train, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$executionGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF"
)

$setExecutionGates = @()
foreach ($gate in $executionGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setExecutionGates += $gate
    }
}

if ($setExecutionGates.Count -gt 0) {
    Write-Host ("Refusing LIBERO offline bounded pilot report while execution gates are set: " + ($setExecutionGates -join ", "))
    exit 20
}

& $Python -m tca_map.datasets.libero_offline_pilot_report `
    --interface-report $InterfaceReportPath `
    --split-report $SplitReportPath `
    --head-report $HeadReportPath `
    --lora-report $LoraReportPath `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath

exit $LASTEXITCODE
