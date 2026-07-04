param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ContractReportPath = "reports\candidate_generation_contract_check_report.json",
    [string]$RuntimeDepsReportPath = "reports\smolvla_runtime_deps_report.json",
    [string]$LoadOnlyReportPath = "reports\smolvla_load_only_smoke_report.json",
    [string]$SingleSampleReportPath = "reports\smolvla_single_sample_interface_report.json",
    [string]$ReportPath = "reports\real_candidate_generation_smoke_plan_report.json",
    [string]$MarkdownReportPath = "reports\real_candidate_generation_smoke_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Real candidate-generation smoke plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not train, download, import heavy VLA models, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.real_candidate_generation_smoke_plan `
    --contract-report $ContractReportPath `
    --runtime-deps-report $RuntimeDepsReportPath `
    --load-only-report $LoadOnlyReportPath `
    --single-sample-report $SingleSampleReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
