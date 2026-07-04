param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_report.json",
    [int]$MaxPairs = 4,
    [int]$MaxActionSteps = 16,
    [int]$GridSize = 8,
    [string]$JsonReportPath = "reports\libero_offline_actionmap_tca_comparison_report.json",
    [string]$MarkdownReportPath = "reports\libero_offline_actionmap_tca_comparison_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO offline ActionMap vs TCA-Map comparison"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads a tiny local HDF5 action subset only. It does not download datasets, run GPU jobs, train, rollout, import simulators or heavy VLA models, access tokens, or execute OpenVLA-OFT."

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_ROLLOUTS",
    "ALLOW_OPENVLA"
)
$setGates = @()
foreach ($gate in $dangerousGates) {
    if ([Environment]::GetEnvironmentVariable($gate) -eq "1") {
        $setGates += $gate
    }
}
if ($setGates.Count -gt 0) {
    Write-Host ("Refusing offline comparison while execution gates are set: " + ($setGates -join ", "))
    exit 20
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}

& $Python -m tca_map.datasets.libero_offline_head_comparison `
    --manifest $ManifestPath `
    --max-pairs $MaxPairs `
    --max-action-steps $MaxActionSteps `
    --grid-size $GridSize `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
