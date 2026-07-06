param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$State3ReportPath = "reports\execspec_state3_replay_validation.json",
    [string]$JsonReportPath = "reports\execspec_state3_5_baseline_dominance_audit.json",
    [string]$MarkdownReportPath = "reports\execspec_state3_5_baseline_dominance_audit.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "ExecSpec STATE 3.5 baseline dominance audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This is report-only analysis over the existing STATE 3 replay report."
Write-Host "It does not run replay, train, compute loss, download, use GPU, execute OpenVLA-OFT, or make paper claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}
if (-not (Test-Path -LiteralPath (Resolve-RepoPath -Path $State3ReportPath))) {
    Write-Host "Refusing: STATE 3 report not found: $State3ReportPath"
    exit 11
}

& $Python -m tca_map.execspec.baseline_dominance_audit `
    --state3-report (Resolve-RepoPath -Path $State3ReportPath) `
    --report-json (Resolve-RepoPath -Path $JsonReportPath) `
    --report-md (Resolve-RepoPath -Path $MarkdownReportPath)
exit $LASTEXITCODE
