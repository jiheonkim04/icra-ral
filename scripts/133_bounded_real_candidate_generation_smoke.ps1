param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PlanReportPath = "reports\real_candidate_generation_smoke_plan_report.json",
    [string]$RuntimeDepsReportPath = "reports\smolvla_runtime_deps_report.json",
    [string]$LoadOnlyReportPath = "reports\smolvla_load_only_smoke_report.json",
    [string]$SingleSampleReportPath = "reports\smolvla_single_sample_interface_report.json",
    [string]$ReportPath = "reports\real_candidate_generation_smoke_report.json",
    [string]$MarkdownReportPath = "reports\real_candidate_generation_smoke_report.md",
    [int]$CandidateCount = 4,
    [int]$HeatmapGrid = 8,
    [double]$Temperature = 0.5,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [string]$Task = "pick up the object"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "Bounded real candidate-generation smoke"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is blocked by default. It requires ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1, ALLOW_HEAVY_IMPORT=1, and ALLOW_SINGLE_SAMPLE_INFERENCE=1."
Write-Host "It does not download, train, rollout, create simulator environments, execute OpenVLA-OFT, use external verifiers, use privileged state, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.real_candidate_generation_smoke `
    --plan-report $PlanReportPath `
    --runtime-deps-report $RuntimeDepsReportPath `
    --load-only-report $LoadOnlyReportPath `
    --single-sample-report $SingleSampleReportPath `
    --report-path $ReportPath `
    --markdown-report-path $MarkdownReportPath `
    --candidate-count $CandidateCount `
    --heatmap-grid $HeatmapGrid `
    --temperature $Temperature `
    --device $Device `
    --task $Task
exit $LASTEXITCODE
