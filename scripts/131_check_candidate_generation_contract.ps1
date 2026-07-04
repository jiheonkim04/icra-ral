param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReadinessReportPath = "reports\candidate_generation_readiness_plan_report.json",
    [string]$JsonReportPath = "reports\candidate_generation_contract_check_report.json",
    [string]$MarkdownReportPath = "reports\candidate_generation_contract_check_report.md",
    [int]$CandidateCount = 4,
    [int]$HeatmapGrid = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Synthetic candidate-generation contract check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script uses synthetic tensors only. It does not train, download, import heavy VLA models, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.candidate_generation_contract_check `
    --readiness-report $ReadinessReportPath `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath `
    --candidate-count $CandidateCount `
    --heatmap-grid $HeatmapGrid
exit $LASTEXITCODE
