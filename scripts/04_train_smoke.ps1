param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "TCA-Map dummy train smoke"
Write-Host "This updates reports/dummy_train_metrics.json and reports/smoke_report.json"
& $Python -m tca_map.launch.smoke_test --mode train
