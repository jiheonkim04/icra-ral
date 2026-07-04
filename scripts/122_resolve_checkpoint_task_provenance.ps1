param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointRoot = "C:\assets\checkpoints\smolvla",
    [string]$NormalizedPlanReportPath = "reports\normalized_action_space_probe_plan_report.json",
    [string]$LiberoActionStatReportPath = "reports\libero_action_stat_subset_audit_report.json",
    [string]$ReportPath = "reports\checkpoint_task_provenance_resolution_report.json",
    [string]$MarkdownReportPath = "reports\checkpoint_task_provenance_resolution_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Checkpoint / task provenance resolution audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "Checkpoint root: $CheckpointRoot"
Write-Host "This script is report-only. It reads local checkpoint metadata, the normalized-action plan, and the LIBERO action-stat audit."
Write-Host "It does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.checkpoint_task_provenance_resolution `
    --checkpoint-root $CheckpointRoot `
    --normalized-plan-report $NormalizedPlanReportPath `
    --libero-action-stat-report $LiberoActionStatReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
