param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PlanReportPath = "reports\vlm_enabled_load_smoke_plan_report.json",
    [string]$ReportPath = "reports\vlm_enabled_load_smoke_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:CUDA_VISIBLE_DEVICES = ""

Write-Host "Bounded VLM-enabled SmolVLA load-only smoke"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner requires ALLOW_HEAVY_IMPORT=1 and ALLOW_VLM_ENABLED_LOAD_SMOKE=1."
Write-Host "It is CPU-first and load-only. It does not infer, train, rollout, download, install, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.vlm_enabled_load_smoke `
    --plan-report $PlanReportPath `
    --report-path $ReportPath `
    --device cpu
exit $LASTEXITCODE
