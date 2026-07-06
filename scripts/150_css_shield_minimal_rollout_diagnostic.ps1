param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports\css_shield_minimal_rollout_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports\css_shield_minimal_rollout_diagnostic_report.md",
    [string]$ProposalSource = "native_or_synthetic",
    [int]$MaxSteps = 10,
    [int]$CameraSize = 64,
    [double]$MaxTranslationNorm = 0.20
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "CSS-Shield minimal rollout diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is bounded: one task, exact-init, max 25 steps, CPU-only. It does not train, download, use GPU, run OpenVLA-OFT, or claim paper-grade evidence."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 25) {
    Write-Host "Refusing: MaxSteps must be between 1 and 25."
    exit 12
}
if ($ProposalSource -notin @("native_or_synthetic", "native_smolvla", "synthetic_counterfactual_probe")) {
    Write-Host "Refusing: ProposalSource must be native_or_synthetic, native_smolvla, or synthetic_counterfactual_probe."
    exit 13
}
if ($MaxTranslationNorm -le 0.0 -or $MaxTranslationNorm -gt 1.0) {
    Write-Host "Refusing: MaxTranslationNorm must be in (0, 1]."
    exit 14
}

& $Python -m tca_map.css_shield.minimal_rollout_diagnostic `
    --manifest (Resolve-RepoPath -Path $ManifestPath) `
    --report-json (Resolve-RepoPath -Path $JsonReportPath) `
    --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
    --proposal-source $ProposalSource `
    --max-steps $MaxSteps `
    --camera-size $CameraSize `
    --max-translation-norm $MaxTranslationNorm

exit $LASTEXITCODE
