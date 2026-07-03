param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$TargetPath = "C:\assets\data\libero",
    [string]$CachePath = "C:\assets\hf_home",
    [string]$SourceConfig = "configs\libero_robosuite_sources.yaml",
    [string]$JsonReportPath = "reports\libero_data_acquisition_report.json",
    [string]$MarkdownReportPath = "reports\libero_data_acquisition_report.md",
    [switch]$RemoteSizeCheck,
    [switch]$Acquire
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Official LIBERO data acquisition gate"
Write-Host "Repo root: $RepoRoot"
Write-Host "Source: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets"
Write-Host "Target: $TargetPath"
Write-Host "Cache: $CachePath"
Write-Host "This script does not run GPU jobs, train, rollout, import simulators or heavy VLA models, access tokens, execute OpenVLA-OFT, upload externally, or make paper claims."

$dangerousGates = @(
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_ROLLOUTS",
    "ALLOW_OPENVLA",
    "ALLOW_CLOUD_HANDOFF"
)
$setGates = @()
foreach ($gate in $dangerousGates) {
    if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($gate))) {
        $setGates += $gate
    }
}
if ($setGates.Count -gt 0) {
    Write-Host ("Refusing LIBERO data acquisition while unrelated execution gates are set: " + ($setGates -join ", "))
    exit 20
}

if ($Acquire -and [Environment]::GetEnvironmentVariable("ALLOW_DOWNLOADS") -ne "1") {
    Write-Host "Refusing acquisition: set task-local ALLOW_DOWNLOADS=1 only for this official LIBERO data acquisition task."
    exit 21
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 10
}

$argsList = @(
    "-m", "tca_map.datasets.libero_data_acquisition",
    "--target", $TargetPath,
    "--cache", $CachePath,
    "--config", $SourceConfig,
    "--report-json", $JsonReportPath,
    "--report-md", $MarkdownReportPath
)
if ($RemoteSizeCheck) { $argsList += "--remote-size-check" }
if ($Acquire) { $argsList += "--acquire" }

& $Python @argsList
exit $LASTEXITCODE
