param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CacheDir = "runs\feature_cache\dummy_contract",
    [string]$ReportPath = "reports\tiny_head_only_smoke_report.json",
    [int]$MaxSteps = 16,
    [int]$MaxRuntimeSeconds = 900,
    [switch]$PrepareDummyCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Tiny head-only smoke runner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This bounded script trains only tiny NumPy heads on cached/dummy features. It does not download assets, run GPU jobs, import heavy VLA models, load SmolVLA/OpenVLA, run VLA inference, rollout, execute simulators, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 100) {
    Write-Host "Refusing: MaxSteps must be between 1 and 100 for tiny head-only smoke."
    exit 11
}

if ($MaxRuntimeSeconds -lt 1 -or $MaxRuntimeSeconds -gt 900) {
    Write-Host "Refusing: MaxRuntimeSeconds must be between 1 and 900."
    exit 12
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to run tiny head-only smoke while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

if ($env:ALLOW_TINY_TRAINING -ne "1") {
    Write-Host "Refusing: ALLOW_TINY_TRAINING=1 is required for bounded tiny head-only smoke."
    exit 21
}

$argsList = @(
    "-m",
    "tca_map.features.tiny_head_only_smoke",
    "--cache-dir",
    $CacheDir,
    "--report-path",
    $ReportPath,
    "--max-steps",
    [string]$MaxSteps,
    "--max-runtime-seconds",
    [string]$MaxRuntimeSeconds
)

if ($PrepareDummyCache) {
    $argsList += "--prepare-dummy-cache"
}

& $Python @argsList
exit $LASTEXITCODE
