param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$DemoPath = "",
    [string]$JsonReportPath = "reports\execspec_mismatch_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports\execspec_mismatch_diagnostic_report.md",
    [int]$MaxSteps = 300,
    [double]$SubstantialDriftThreshold = 0.10,
    [double]$GripperMismatchThreshold = 0.25
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "ExecSpec mismatch diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script reads bounded local HDF5 expert actions and computes action-space mismatch/repair metrics."
Write-Host "It does not download, install, load models, infer, train, use GPU, create a simulator environment, rollout, execute OpenVLA-OFT, or make paper claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}
if ($MaxSteps -lt 2 -or $MaxSteps -gt 512) {
    Write-Host "Refusing: MaxSteps must be between 2 and 512."
    exit 11
}
if ($SubstantialDriftThreshold -le 0 -or $SubstantialDriftThreshold -gt 10) {
    Write-Host "Refusing: SubstantialDriftThreshold must be in (0, 10]."
    exit 12
}
if ($GripperMismatchThreshold -lt 0 -or $GripperMismatchThreshold -gt 1) {
    Write-Host "Refusing: GripperMismatchThreshold must be in [0, 1]."
    exit 13
}
if ([string]::IsNullOrWhiteSpace($DemoPath) -and -not (Test-Path -LiteralPath (Resolve-RepoPath -Path $ManifestPath))) {
    Write-Host "Refusing: manifest not found: $ManifestPath"
    exit 14
}
if (-not [string]::IsNullOrWhiteSpace($DemoPath) -and -not (Test-Path -LiteralPath (Resolve-RepoPath -Path $DemoPath))) {
    Write-Host "Refusing: demo HDF5 not found: $DemoPath"
    exit 15
}

$argsList = @(
    "-m", "tca_map.execspec.mismatch_diagnostic",
    "--manifest", (Resolve-RepoPath -Path $ManifestPath),
    "--max-steps", "$MaxSteps",
    "--substantial-drift-threshold", "$SubstantialDriftThreshold",
    "--gripper-mismatch-threshold", "$GripperMismatchThreshold",
    "--report-json", (Resolve-RepoPath -Path $JsonReportPath),
    "--report-md", (Resolve-RepoPath -Path $MarkdownReportPath)
)
if (-not [string]::IsNullOrWhiteSpace($DemoPath)) {
    $argsList += @("--demo-path", (Resolve-RepoPath -Path $DemoPath))
}

& $Python @argsList
exit $LASTEXITCODE
