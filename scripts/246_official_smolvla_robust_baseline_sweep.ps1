param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PredictionArtifact = "reports\fcar_prediction_artifact.json",
    [string]$FcarResultJson = "reports\fcar_tiny_gate_result.json",
    [int]$FoldCount = 5,
    [string]$JsonReportPath = "reports\official_smolvla_robust_baseline_sweep_result.json",
    [string]$MarkdownResultPath = "reports\official_smolvla_robust_baseline_sweep_result.md",
    [string]$PlanPath = "reports\official_smolvla_robust_baseline_sweep_plan.md",
    [string]$PostmortemPath = "reports\fcar_tiny_gate_postmortem.md",
    [string]$DecisionPath = "reports\official_smolvla_post_fcar_decision.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Official SmolVLA-LIBERO robust baseline sweep"
Write-Host "Repo root: $RepoRoot"
Write-Host "This postmortem reads the official FCAR prediction artifact and evaluates baselines across episode-disjoint folds."
Write-Host "It does not tune FCAR, train a new method, download assets, run rollouts, full benchmark, OpenVLA-OFT, or the old custom LIBERO_7D route."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.official_libero_robust_baseline_sweep `
    --prediction-artifact $PredictionArtifact `
    --fcar-result-json $FcarResultJson `
    --fold-count $FoldCount `
    --report-json $JsonReportPath `
    --result-md $MarkdownResultPath `
    --plan-md $PlanPath `
    --postmortem-md $PostmortemPath `
    --decision-md $DecisionPath

exit $LASTEXITCODE
