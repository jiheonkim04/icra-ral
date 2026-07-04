param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$AuditReportPath = "reports\action_normalization_provenance_audit_report.json",
    [string]$ReportPath = "reports\action_stat_provenance_correction_plan_report.json",
    [string]$MarkdownReportPath = "reports\action_stat_provenance_correction_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Action-stat provenance correction plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It reads the action normalization provenance audit and selects the next safe correction/audit step."
Write-Host "It does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, alter policy behavior, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.action_stat_provenance_correction_plan `
    --audit-report $AuditReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
