param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [switch]$SkipState15Execution,
    [switch]$IncludeNative,
    [switch]$RunState2IfGreen
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "CSS-Shield autopilot next controller"
Write-Host "Repo root: $RepoRoot"
Write-Host "This controller reads state, executes the next bounded diagnostic when safe, and updates persistent state files."

if (-not $SkipState15Execution) {
    $stateArgs = @("-ExecutionPolicy", "Bypass", "-File", "scripts\161_css_shield_state1_5_semantic_observability.ps1", "-RunState2IfGreen")
    if ($IncludeNative) {
        $stateArgs += "-IncludeNative"
    }
    powershell @stateArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$mainCommit = (git rev-parse HEAD).Trim()
& $Python -m tca_map.css_shield.autopilot_next --main-commit $mainCommit
exit $LASTEXITCODE
