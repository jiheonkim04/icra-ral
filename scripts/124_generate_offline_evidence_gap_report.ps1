param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PivotReportPath = "reports\offline_tca_map_lora_pivot_plan_report.json",
    [string]$HeadReportPath = "reports\libero_offline_actionmap_tca_comparison_report.json",
    [string]$LoraReportPath = "reports\libero_offline_lora_comparison_report.json",
    [string]$BoundedLoraScaleupReportPath = "reports\bounded_lora_offline_scaleup_report.json",
    [string]$TcaSelectStressReportPath = "reports\tca_select_ambiguity_stress_report.json",
    [string]$ProvenanceReportPath = "reports\checkpoint_task_provenance_resolution_report.json",
    [string]$BoundedPilotReportPath = "reports\libero_offline_bounded_pilot_report.json",
    [string]$ReportPath = "reports\offline_tca_lora_evidence_gap_report_runtime.json",
    [string]$MarkdownReportPath = "reports\offline_tca_lora_evidence_gap_report_runtime.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Offline TCA-Map / LoRA evidence gap report"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is report-only. It consolidates existing offline proxy reports and provenance blockers."
Write-Host "It does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.offline_evidence_gap_report `
    --pivot-report $PivotReportPath `
    --head-report $HeadReportPath `
    --lora-report $LoraReportPath `
    --bounded-lora-scaleup-report $BoundedLoraScaleupReportPath `
    --tca-select-stress-report $TcaSelectStressReportPath `
    --provenance-report $ProvenanceReportPath `
    --bounded-pilot-report $BoundedPilotReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
