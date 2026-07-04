param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_report.json",
    [int]$MaxPairs = 4,
    [int]$MaxActionSteps = 16,
    [int]$GridSize = 8,
    [int]$MaxSteps = 64,
    [double]$LearningRate = 0.05,
    [int]$MaxRuntimeSeconds = 900,
    [string]$JsonReportPath = "reports\libero_offline_actionmap_tca_comparison_report.json",
    [string]$MarkdownReportPath = "reports\libero_offline_actionmap_tca_comparison_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO offline ActionMap vs TCA-Map tiny training/eval"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script trains tiny CPU NumPy head-only models on local HDF5 action snippets. It does not download datasets, run GPU jobs, run rollouts, import simulators or heavy VLA models, access tokens, or execute OpenVLA-OFT."

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
    Write-Host ("Refusing offline training/eval while dangerous gates are set: " + ($setGates -join ", "))
    exit 20
}

if ([Environment]::GetEnvironmentVariable("ALLOW_TINY_TRAINING") -ne "1") {
    Write-Host "ALLOW_TINY_TRAINING=1 is required for this bounded tiny offline training/eval milestone."
    exit 21
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
    --max-steps $MaxSteps `
    --learning-rate $LearningRate `
    --max-runtime-seconds $MaxRuntimeSeconds `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
