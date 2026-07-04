param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SynthesisReportPath = "reports\scaleup_attribution_gap_synthesis_report.json",
    [int]$MaxPairs = 16,
    [int]$MaxRecords = 64,
    [int]$CandidateCount = 8,
    [double]$Temperature = 0.5,
    [int]$MaxRuntimeSeconds = 300,
    [string]$ReportPath = "reports\tca_select_ambiguity_stress_plan_report.json",
    [string]$MarkdownReportPath = "reports\tca_select_ambiguity_stress_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "TCA-Select ambiguity stress-test plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not train, download, import heavy VLA models, load models, infer, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.tca_select_ambiguity_stress_plan `
    --synthesis-report $SynthesisReportPath `
    --max-pairs $MaxPairs `
    --max-records $MaxRecords `
    --candidate-count $CandidateCount `
    --temperature $Temperature `
    --max-runtime-seconds $MaxRuntimeSeconds `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
