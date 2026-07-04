param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SmokeReportPath = "reports\real_candidate_generation_smoke_report.json",
    [string]$ReportPath = "reports\real_candidate_generation_smoke_summary_report.json",
    [string]$MarkdownReportPath = "reports\real_candidate_generation_smoke_summary_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Real candidate-generation smoke summary"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is report-only. It reads an existing smoke report and does not import models, infer, train, rollout, use GPU jobs, download assets, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.real_candidate_generation_smoke_summary `
    --smoke-report $SmokeReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
