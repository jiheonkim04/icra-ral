param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [int]$Trials = 60,
    [string]$Seeds = "11,23,37",
    [string]$JsonReportPath = "reports\amp_gd_minimal_probe_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports\amp_gd_minimal_probe_diagnostic_report.md",
    [string]$State1MarkdownPath = "reports\amp_gd_state1_minimal_probe_result.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "AMP-GD minimal active micro-probe diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is bounded: toy 2D control rollout, seeded trials, no training, no downloads, no GPU, no heavy VLA imports, no OpenVLA-OFT, and no paper-grade claim."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

if ($Trials -lt 20 -or $Trials -gt 300) {
    Write-Host "Refusing: Trials must be between 20 and 300."
    exit 12
}

& $Python -m tca_map.amp_gd.minimal_probe_diagnostic `
    --trials $Trials `
    --seeds $Seeds `
    --report-json (Resolve-RepoPath -Path $JsonReportPath) `
    --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
    --state1-md (Resolve-RepoPath -Path $State1MarkdownPath)

exit $LASTEXITCODE

