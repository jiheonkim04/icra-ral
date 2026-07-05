param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_report.json",
    [string]$JsonReportPath = "reports\libero_target_prior_fixed_head_comparison_report.json",
    [string]$MarkdownReportPath = "reports\libero_target_prior_fixed_head_comparison_report.md",
    [int]$MaxPairs = 4,
    [int]$MaxActionSteps = 16,
    [int]$MaxSamples = 8,
    [int]$GridSize = 8,
    [int]$MaxSteps = 64,
    [double]$LearningRate = 0.05,
    [int]$MaxRuntimeSeconds = 900
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO target-prior-fixed ActionMap vs TCA-Map comparison"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script trains tiny CPU NumPy diagnostic heads on the fixed 8-sample offline split. It does not download, use GPU, run rollouts, run LoRA, import heavy VLA models, execute simulators, access tokens, or execute OpenVLA-OFT."

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
    Write-Host ("Refusing target-prior-fixed comparison while dangerous gates are set: " + ($setGates -join ", "))
    exit 20
}

if ([Environment]::GetEnvironmentVariable("ALLOW_TINY_TRAINING") -ne "1") {
    Write-Host "ALLOW_TINY_TRAINING=1 is required for this bounded tiny diagnostic training milestone."
    exit 21
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}

& $Python -m tca_map.datasets.libero_target_prior_fixed_head_comparison `
    --manifest $ManifestPath `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath `
    --max-pairs $MaxPairs `
    --max-action-steps $MaxActionSteps `
    --max-samples $MaxSamples `
    --grid-size $GridSize `
    --max-steps $MaxSteps `
    --learning-rate $LearningRate `
    --max-runtime-seconds $MaxRuntimeSeconds

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
