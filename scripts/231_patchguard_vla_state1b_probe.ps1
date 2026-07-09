param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PlanReportPath = "reports\vlm_enabled_repeated_offline_decoding_plan_report.json",
    [string]$ReportPath = "reports\patchguard_vla_state1b_result.json",
    [int]$MaxSteps = 10,
    [switch]$DependencyInstallHappened
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "PatchGuard-VLA STATE 1B environment and tiny LoRA feasibility gate"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_HEAVY_IMPORT=1, ALLOW_PATCHGUARD_VLA_STATE1B=1, and ALLOW_PATCHGUARD_TINY_LORA_TRAINING=1."
Write-Host "Runs CUDA dependency checks, real SmolVLA LoRA injection, and at most $MaxSteps bounded optimization steps per variant. It does not download models/datasets, rollout, run OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

$installFlag = @()
if ($DependencyInstallHappened) {
    $installFlag = @("--dependency-install-happened")
}

& $Python -m tca_map.patchguard_vla.state1b `
    --plan-report $PlanReportPath `
    --report-path $ReportPath `
    --max-steps $MaxSteps `
    @installFlag
exit $LASTEXITCODE
