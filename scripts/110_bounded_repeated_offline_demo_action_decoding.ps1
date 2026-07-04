param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$PlanReportPath = "reports\repeated_offline_demo_action_decoding_plan_report.json",
    [string]$JsonReportPath = "reports\repeated_offline_demo_action_decoding_report.json",
    [string]$MarkdownReportPath = "reports\repeated_offline_demo_action_decoding_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "Bounded repeated offline demonstration action decoding"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads local HDF5 observations/actions and runs at most three CPU SmolVLA action decodes. It does not create simulator environments, rollout, train, download, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

$savedGate = [Environment]::GetEnvironmentVariable("ALLOW_REPEATED_OFFLINE_DEMO_DECODING")
Remove-Item Env:\ALLOW_REPEATED_OFFLINE_DEMO_DECODING -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\109_plan_repeated_offline_demo_action_decoding.ps1") `
        -JsonReportPath $PlanReportPath `
        -MarkdownReportPath ($PlanReportPath -replace "\.json$", ".md") | Out-Null
} finally {
    if ($null -ne $savedGate) {
        $env:ALLOW_REPEATED_OFFLINE_DEMO_DECODING = $savedGate
    }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Resolve-RepoPath -Path $PlanReportPath))) {
    Write-Error "Repeated offline decoding planner failed before bounded runner."
    exit 11
}

& $Python -m tca_map.smolvla.repeated_offline_demo_action_decoding `
    --plan-report (Resolve-RepoPath -Path $PlanReportPath) `
    --report-path (Resolve-RepoPath -Path $JsonReportPath)
$exitCode = $LASTEXITCODE

if (Test-Path -LiteralPath (Resolve-RepoPath -Path $JsonReportPath)) {
    $report = Get-Content -Raw (Resolve-RepoPath -Path $JsonReportPath) | ConvertFrom-Json
    $lines = @(
        "# Repeated Offline Demonstration Action Decoding Report",
        "",
        "- decision: $($report.decision)",
        "- passed: $($report.repeated_offline_demo_action_decoding_passed)",
        "- model load performed: $($report.policy.model_load_performed)",
        "- model inference performed: $($report.policy.model_inference_performed)",
        "- simulator environment created: false",
        "- rollouts performed: false",
        "- training performed: false",
        "- paper-grade claim: false",
        "- sample count: $($report.metrics.sample_count)",
        "- mean action L1 to expert: $($report.metrics.mean_action_l1_to_expert)",
        "- mean action MSE to expert: $($report.metrics.mean_action_mse_to_expert)",
        "- offline alignment signal: $($report.metrics.offline_alignment_signal)",
        "",
        "This is tiny repeated offline diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath (Resolve-RepoPath -Path $MarkdownReportPath) -Encoding UTF8
}

exit $exitCode
