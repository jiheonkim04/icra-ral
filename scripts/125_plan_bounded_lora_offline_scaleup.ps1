param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$EvidenceGapReportPath = "reports\offline_tca_lora_evidence_gap_report_runtime.json",
    [int]$MaxPairs = 16,
    [int]$MaxSamples = 64,
    [int]$MaxSteps = 64,
    [int]$MaxRuntimeMinutes = 20,
    [int]$LoraRank = 4,
    [string]$ReportPath = "reports\bounded_lora_offline_scaleup_plan_report.json",
    [string]$MarkdownReportPath = "reports\bounded_lora_offline_scaleup_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded LoRA / offline proxy scale-up plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It defines a future CPU-only offline LoRA scale-up runner budget from existing evidence reports."
Write-Host "It does not download, install, import heavy VLA models, load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, alter policy behavior, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.bounded_lora_offline_scaleup_plan `
    --evidence-gap-report $EvidenceGapReportPath `
    --max-pairs $MaxPairs `
    --max-samples $MaxSamples `
    --max-steps $MaxSteps `
    --max-runtime-minutes $MaxRuntimeMinutes `
    --lora-rank $LoraRank `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath
exit $LASTEXITCODE
