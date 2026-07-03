param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\feature_cache_plan_report.json",
    [string]$OutputDir = "runs\feature_cache\dummy_contract",
    [switch]$WriteDummyCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Feature cache interface planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script does not download assets, run GPU jobs, import heavy VLA models, load models, infer, train, rollout, or execute OpenVLA-OFT."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to run feature cache planner while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

$argsList = @(
    "-m",
    "tca_map.features.cache",
    "--report-path",
    $ReportPath,
    "--output-dir",
    $OutputDir
)

if ($WriteDummyCache) {
    $argsList += "--write-dummy-cache"
}

& $Python @argsList
exit $LASTEXITCODE
