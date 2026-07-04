param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SynthesisReportPath = "reports\scaleup_attribution_gap_synthesis_report.json",
    [string]$LoadOnlyReportPath = "reports\smolvla_load_only_smoke_report.json",
    [string]$SingleSampleReportPath = "reports\smolvla_single_sample_interface_report.json",
    [string]$FeatureCacheReportPath = "reports\feature_cache_eval_report.json",
    [string]$ReportPath = "reports\candidate_generation_readiness_plan_report.json",
    [string]$MarkdownReportPath = "reports\candidate_generation_readiness_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Learned-policy candidate-generation readiness plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is report-only. It does not train, download, import heavy VLA models, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.candidate_generation_readiness_plan `
    --synthesis-report $SynthesisReportPath `
    --load-only-report $LoadOnlyReportPath `
    --single-sample-report $SingleSampleReportPath `
    --feature-cache-report $FeatureCacheReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
