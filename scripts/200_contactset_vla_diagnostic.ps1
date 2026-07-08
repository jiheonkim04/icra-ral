param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$LiberoDataRoot = "",
    [int]$MaxDemos = 6,
    [int]$MaxActionSteps = 140,
    [int]$FeatureWidth = 48,
    [double]$Ridge = 0.001,
    [string]$JsonReportPath = "reports\contactset_vla_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports\contactset_vla_diagnostic_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "ContactSet-VLA offline action-head diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script trains tiny CPU NumPy ridge action heads over local LIBERO HDF5 chunks. It does not download, run GPU jobs, run rollouts, import VLA models or simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT"
)
$setGates = @()
foreach ($gate in $dangerousGates) {
    if ([Environment]::GetEnvironmentVariable($gate) -eq "1") {
        $setGates += $gate
    }
}
if ($setGates.Count -gt 0) {
    Write-Host ("Refusing ContactSet-VLA diagnostic while dangerous gates are set: " + ($setGates -join ", "))
    exit 20
}

if ([Environment]::GetEnvironmentVariable("ALLOW_TINY_TRAINING") -ne "1") {
    Write-Host "ALLOW_TINY_TRAINING=1 is required for this bounded tiny CPU action-head diagnostic."
    exit 21
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}

if ($MaxDemos -lt 1 -or $MaxDemos -gt 24) {
    Write-Host "MaxDemos must be between 1 and 24."
    exit 2
}
if ($MaxActionSteps -lt 8 -or $MaxActionSteps -gt 320) {
    Write-Host "MaxActionSteps must be between 8 and 320."
    exit 3
}
if ($FeatureWidth -lt 16 -or $FeatureWidth -gt 256) {
    Write-Host "FeatureWidth must be between 16 and 256."
    exit 4
}

$argsList = @(
    "-m", "tca_map.contactset_vla.diagnostic",
    "--paths-file", $PathsFile,
    "--max-demos", "$MaxDemos",
    "--max-action-steps", "$MaxActionSteps",
    "--feature-width", "$FeatureWidth",
    "--ridge", "$Ridge",
    "--report-json", $JsonReportPath,
    "--report-md", $MarkdownReportPath
)

if (-not [string]::IsNullOrWhiteSpace($LiberoDataRoot)) {
    $argsList += @("--libero-data-root", $LiberoDataRoot)
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

