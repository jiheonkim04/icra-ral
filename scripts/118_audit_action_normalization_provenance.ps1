param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SummaryReportPath = "reports\vlm_enabled_offline_decoding_summary_report.json",
    [string]$VlmEnabledReportPath = "reports\vlm_enabled_repeated_offline_decoding_report.json",
    [string]$ReportPath = "reports\action_normalization_provenance_audit_report.json",
    [string]$MarkdownReportPath = "reports\action_normalization_provenance_audit_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Action normalization provenance audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is report-only. It reads local config, processor JSON/safetensors, and existing offline diagnostic reports."
Write-Host "It does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.action_normalization_provenance_audit `
    --summary-report $SummaryReportPath `
    --vlm-enabled-report $VlmEnabledReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
