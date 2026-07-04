param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PlanReportPath = "reports\vlm_enabled_repeated_offline_decoding_plan_report.json",
    [string]$PreviousReportPath = "reports\repeated_offline_demo_action_decoding_report.json",
    [string]$ReportPath = "reports\vlm_enabled_repeated_offline_decoding_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:CUDA_VISIBLE_DEVICES = ""

Write-Host "Bounded VLM-enabled repeated offline action decoding"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner requires ALLOW_HEAVY_IMPORT=1 and ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING=1."
Write-Host "It decodes at most three local HDF5 timesteps on CPU. It does not create simulator environments, rollout, train, download, install, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.vlm_enabled_repeated_offline_decoding `
    --plan-report $PlanReportPath `
    --previous-report $PreviousReportPath `
    --report-path $ReportPath `
    --device cpu
exit $LASTEXITCODE
