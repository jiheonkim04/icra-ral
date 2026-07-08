param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$LiberoDataRoot = "",
    [int]$MaxDemos = 8,
    [int]$MaxActionSteps = 180,
    [int]$FeatureWidth = 48,
    [int]$MaxSteps = 120,
    [double]$LearningRate = 0.18,
    [int]$TransBins = 7,
    [int]$RotBins = 7,
    [string]$JsonReportPath = "reports\actionmap_anchor_state1_result.json",
    [string]$MarkdownReportPath = "reports\actionmap_anchor_state1_result.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "ActionMap mini-anchor feasibility diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script trains tiny CPU NumPy action heads on local LIBERO HDF5 action labels. It does not attempt full official ActionMap reproduction, implement an extension, download, use GPU, run rollouts, import VLA models or simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

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
    Write-Host ("Refusing ActionMap anchor diagnostic while dangerous gates are set: " + ($setGates -join ", "))
    exit 20
}

if ([Environment]::GetEnvironmentVariable("ALLOW_TINY_TRAINING") -ne "1") {
    Write-Host "ALLOW_TINY_TRAINING=1 is required for this bounded tiny CPU reproduction diagnostic."
    exit 21
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}
if ($MaxDemos -lt 1 -or $MaxDemos -gt 32) {
    Write-Host "MaxDemos must be between 1 and 32."
    exit 2
}
if ($MaxActionSteps -lt 12 -or $MaxActionSteps -gt 320) {
    Write-Host "MaxActionSteps must be between 12 and 320."
    exit 3
}
if ($MaxSteps -lt 1 -or $MaxSteps -gt 300) {
    Write-Host "MaxSteps must be between 1 and 300."
    exit 4
}

$argsList = @(
    "-m", "tca_map.actionmap_anchor.diagnostic",
    "--paths-file", $PathsFile,
    "--max-demos", "$MaxDemos",
    "--max-action-steps", "$MaxActionSteps",
    "--feature-width", "$FeatureWidth",
    "--max-steps", "$MaxSteps",
    "--learning-rate", "$LearningRate",
    "--trans-bins", "$TransBins",
    "--rot-bins", "$RotBins",
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
