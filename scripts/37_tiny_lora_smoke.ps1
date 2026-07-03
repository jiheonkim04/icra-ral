param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CacheDir = "runs\feature_cache\dummy_contract",
    [string]$ReportPath = "reports\tiny_lora_smoke_report.json",
    [int]$MaxSteps = 16,
    [int]$MaxRuntimeSeconds = 900,
    [int]$MaxSamples = 4,
    [int]$Rank = 4,
    [switch]$PrepareDummyCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Tiny LoRA smoke runner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This bounded script trains only tiny NumPy LoRA adapter weights on cached/dummy features. It does not download assets, run GPU jobs, import heavy VLA models, load SmolVLA/OpenVLA, run model inference, rollout, execute simulators, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 100) {
    Write-Host "Refusing: MaxSteps must be between 1 and 100 for tiny LoRA smoke."
    exit 11
}

if ($MaxRuntimeSeconds -lt 1 -or $MaxRuntimeSeconds -gt 900) {
    Write-Host "Refusing: MaxRuntimeSeconds must be between 1 and 900."
    exit 12
}

if ($MaxSamples -lt 1 -or $MaxSamples -gt 200) {
    Write-Host "Refusing: MaxSamples must be between 1 and 200 for tiny LoRA smoke."
    exit 13
}

if ($Rank -lt 1 -or $Rank -gt 16) {
    Write-Host "Refusing: Rank must be between 1 and 16 for tiny LoRA smoke."
    exit 14
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to run tiny LoRA smoke while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

if ($env:ALLOW_TINY_TRAINING -ne "1") {
    Write-Host "Refusing: ALLOW_TINY_TRAINING=1 is required for bounded tiny LoRA smoke."
    exit 21
}

$argsList = @(
    "-m",
    "tca_map.adapters.tiny_lora_smoke",
    "--cache-dir",
    $CacheDir,
    "--report-path",
    $ReportPath,
    "--max-steps",
    [string]$MaxSteps,
    "--max-runtime-seconds",
    [string]$MaxRuntimeSeconds,
    "--max-samples",
    [string]$MaxSamples,
    "--rank",
    [string]$Rank
)

if ($PrepareDummyCache) {
    $argsList += "--prepare-dummy-cache"
}

& $Python @argsList
exit $LASTEXITCODE
