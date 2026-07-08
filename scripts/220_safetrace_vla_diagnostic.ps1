param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$LiberoDataRoot = "",
    [string]$LiberoRoot = "",
    [int]$MaxDemos = 8,
    [int]$MaxActionSteps = 180,
    [int]$Chunk = 8,
    [string]$JsonReportPath = "reports\safetrace_vla_state1_result.json",
    [string]$MarkdownReportPath = "reports\safetrace_vla_state1_result.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "SafeTrace-VLA STATE 1 temporal-safety diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script computes local HDF5 temporal safety proxy metrics and preference-pair headroom. It does not download, use GPU, train, run rollouts, import VLA models or simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

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
    Write-Host ("Refusing SafeTrace-VLA diagnostic while dangerous gates are set: " + ($setGates -join ", "))
    exit 20
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}
if ($MaxDemos -lt 1 -or $MaxDemos -gt 32) {
    Write-Host "MaxDemos must be between 1 and 32."
    exit 2
}
if ($MaxActionSteps -lt 12 -or $MaxActionSteps -gt 360) {
    Write-Host "MaxActionSteps must be between 12 and 360."
    exit 3
}
if ($Chunk -lt 4 -or $Chunk -gt 32) {
    Write-Host "Chunk must be between 4 and 32."
    exit 4
}

$argsList = @(
    "-m", "tca_map.safetrace_vla.diagnostic",
    "--paths-file", $PathsFile,
    "--max-demos", "$MaxDemos",
    "--max-action-steps", "$MaxActionSteps",
    "--chunk", "$Chunk",
    "--report-json", $JsonReportPath,
    "--report-md", $MarkdownReportPath
)

if (-not [string]::IsNullOrWhiteSpace($LiberoDataRoot)) {
    $argsList += @("--libero-data-root", $LiberoDataRoot)
}
if (-not [string]::IsNullOrWhiteSpace($LiberoRoot)) {
    $argsList += @("--libero-root", $LiberoRoot)
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

