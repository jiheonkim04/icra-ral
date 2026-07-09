param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PlanReportPath = "reports\vlm_enabled_repeated_offline_decoding_plan_report.json",
    [string]$ReportPath = "reports\patchguard_vla_state1_result.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:CUDA_VISIBLE_DEVICES = ""

Write-Host "Bounded PatchGuard-VLA STATE 1 diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner requires ALLOW_HEAVY_IMPORT=1 and ALLOW_PATCHGUARD_VLA_STATE1=1."
Write-Host "It loads local SmolVLA on CPU and decodes clean/patched local LIBERO HDF5 observations. It does not train, rollout, download, install, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.patchguard_vla.diagnostic `
    --plan-report $PlanReportPath `
    --report-path $ReportPath `
    --device cpu
exit $LASTEXITCODE

