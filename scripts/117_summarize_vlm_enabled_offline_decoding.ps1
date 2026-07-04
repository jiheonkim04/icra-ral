param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$NoVlmReportPath = "reports\repeated_offline_demo_action_decoding_report.json",
    [string]$VlmEnabledReportPath = "reports\vlm_enabled_repeated_offline_decoding_report.json",
    [string]$ReportPath = "reports\vlm_enabled_offline_decoding_summary_report.json",
    [string]$MarkdownReportPath = "reports\vlm_enabled_offline_decoding_summary_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "VLM-enabled offline decoding summary"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is report-only. It reads existing offline diagnostic reports and local config JSON only."
Write-Host "It does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.vlm_enabled_offline_decoding_summary `
    --no-vlm-report $NoVlmReportPath `
    --vlm-enabled-report $VlmEnabledReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
