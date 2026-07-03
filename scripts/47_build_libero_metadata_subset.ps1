param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$Config = "configs\libero_metadata_subset.yaml",
    [string]$JsonReportPath = "reports\libero_metadata_subset_report.json",
    [string]$MarkdownReportPath = "reports\libero_metadata_subset_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO metadata-only subset builder"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads local BDDL/task metadata only. It does not download datasets, run GPU jobs, train, rollout, import simulators or heavy VLA models, access tokens, or execute OpenVLA-OFT."

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
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
    Write-Host ("Refusing metadata-only builder while execution gates are set: " + ($setGates -join ", "))
    exit 20
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}

& $Python -m tca_map.datasets.libero_metadata_subset `
    --paths-file $PathsFile `
    --config $Config `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
