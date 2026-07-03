param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\smolvla_single_sample_interface_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "SmolVLA single-sample interface smoke"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script uses one synthetic observation only. It does not train, run rollouts, use a simulator, download assets, access tokens, evaluate datasets, or execute OpenVLA-OFT."
Write-Host "It requires ALLOW_HEAVY_IMPORT=1 and ALLOW_SINGLE_SAMPLE_INFERENCE=1, which may be set only after a green risk assessment for this bounded interface smoke."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.single_sample_interface_smoke --report-path $ReportPath
exit $LASTEXITCODE
