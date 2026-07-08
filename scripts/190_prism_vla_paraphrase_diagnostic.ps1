param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$LiberoRoot = "",
    [string]$LiberoDataRoot = "",
    [string]$LiberoParaMetadataCsv = "C:\assets\data\libero_para\libero_para_metadata.csv",
    [int]$MaxTasks = 5,
    [int]$MaxParaphrasesPerTask = 18,
    [int]$MaxActionSteps = 8,
    [int]$MaxSteps = 160,
    [double]$LearningRate = 0.12,
    [int]$FeatureWidth = 96,
    [string]$JsonReportPath = "reports\prism_vla_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports\prism_vla_diagnostic_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "PRISM-VLA paraphrase robustness diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script trains tiny CPU NumPy surrogate policies on local LIBERO action chunks and paraphrase metadata. It does not download datasets, run GPU jobs, run rollouts, import simulators or heavy VLA models, access tokens, or execute OpenVLA-OFT."

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
    Write-Host ("Refusing PRISM-VLA diagnostic while dangerous gates are set: " + ($setGates -join ", "))
    exit 20
}

if ([Environment]::GetEnvironmentVariable("ALLOW_TINY_TRAINING") -ne "1") {
    Write-Host "ALLOW_TINY_TRAINING=1 is required for this bounded tiny CPU training/eval milestone."
    exit 21
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}

$argsList = @(
    "-m", "tca_map.prism_vla.paraphrase_diagnostic",
    "--paths-file", $PathsFile,
    "--libero-para-metadata-csv", $LiberoParaMetadataCsv,
    "--max-tasks", "$MaxTasks",
    "--max-paraphrases-per-task", "$MaxParaphrasesPerTask",
    "--max-action-steps", "$MaxActionSteps",
    "--max-steps", "$MaxSteps",
    "--learning-rate", "$LearningRate",
    "--feature-width", "$FeatureWidth",
    "--report-json", $JsonReportPath,
    "--report-md", $MarkdownReportPath
)

if (-not [string]::IsNullOrWhiteSpace($LiberoRoot)) {
    $argsList += @("--libero-root", $LiberoRoot)
}
if (-not [string]::IsNullOrWhiteSpace($LiberoDataRoot)) {
    $argsList += @("--libero-data-root", $LiberoDataRoot)
}

& $Python @argsList

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
