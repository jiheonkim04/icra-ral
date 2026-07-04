param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$LiberoActionStatReportPath = "reports\libero_action_stat_subset_audit_report.json",
    [string]$VlmSummaryReportPath = "reports\vlm_enabled_offline_decoding_summary_report.json",
    [string]$ReportPath = "reports\normalized_action_space_probe_plan_report.json",
    [string]$MarkdownReportPath = "reports\normalized_action_space_probe_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Normalized action-space probe / checkpoint provenance plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It reads the LIBERO action-stat subset audit and chooses the next safe normalized-action/provenance step."
Write-Host "It does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.normalized_action_space_probe_plan `
    --libero-action-stat-report $LiberoActionStatReportPath `
    --vlm-summary-report $VlmSummaryReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
