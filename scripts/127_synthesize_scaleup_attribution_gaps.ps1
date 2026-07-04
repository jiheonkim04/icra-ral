param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$EvidenceReportPath = "reports\offline_tca_lora_evidence_gap_report_runtime.json",
    [string]$TcaSelectStressReportPath = "reports\tca_select_ambiguity_stress_report.json",
    [string]$ReportPath = "reports\scaleup_attribution_gap_synthesis_report.json",
    [string]$MarkdownReportPath = "reports\scaleup_attribution_gap_synthesis_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Scale-up attribution gap synthesis"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is report-only. It does not train, download, import heavy VLA models, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.scaleup_attribution_gap_synthesis `
    --evidence-report $EvidenceReportPath `
    --tca-select-stress-report $TcaSelectStressReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
